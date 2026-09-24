"""Fixed-route risk evaluator for independent B/C0/C0V/D route searches.

The risk is the original full-network target-inventory loss with continuous
empty-to-empty truck recourse.  Optional, duplicate-free station visits are
supported.  All stations, including unvisited ones, remain in the loss.

No actual outcomes are read.  A maximization timeout yields an adversarial
lower bound and a valid risk upper bound, never a claimed exact objective.
All caches belong to one evaluator instance; no route or scenario pool is
shared across methods.  Final physical verification concerns the returned
worst-case witness, and is separate from global outer-route optimality.
"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import sys
import time
import warnings

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
sys.dont_write_bytecode = True
_STUDY = Path(__file__).resolve().parents[1]
for _folder in ("three_set_study", "conservation_exploration"):
    _path = str(_STUDY / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
from compact_oracle import CompactConservationOracle
from oracle import deterministic_value, solve_recourse


class RobustEvaluator:
    """Exact/anytime risk and distance for a single method's route search.

    Required data: demand_nominal, predicted, station_capacity,
    road_distance_km; C0/C0V/D also require an explicit scalar fleet_size.
    B intentionally relaxes system fleet closure while retaining all station
    and truck constraints. Required spec for robust arms: lower and upper;
    C arms may additionally have groups/linear_rows as in the compact oracle.
    The D arm evaluates error=0, irrespective of the supplied robust spec.

    evaluate(routes, time_limit=1., verify_physical=False) returns risk bounds
    value/lower_bound and upper_bound, objective_lower/objective_upper after
    adding distance cost, and certified=True only when the inner gap closes.
    It never claims global route optimality.  The outer search should compare
    certified values or explicitly use the upper bounds conservatively.
    """

    def __init__(self, data, cfg, spec, method):
        started = time.perf_counter()
        self.method = str(method).replace("_", "").upper()
        if self.method not in {"B", "C0", "C0V", "D"}:
            raise ValueError("Expected method B, C0, C0V, or D")
        self.cfg = dict(cfg)
        self.d = np.asarray(data["demand_nominal"], float)
        self.n = len(self.d)
        self.predicted = np.asarray(data["predicted"], float)
        self.station_capacity = np.asarray(data["station_capacity"], float)
        self.target = self.predicted + self.d
        self.road = np.asarray(data["road_distance_km"], float)
        self.capacity = float(cfg.get("truck_capacity", 30.))
        self.plus = float(cfg.get("shortage_penalty", 10.))
        self.minus = float(cfg.get("surplus_penalty", 10.))
        self.distance_cost = float(cfg.get("distance_cost_per_km", 1.))
        self.tolerance = float(cfg.get("oracle_absolute_tolerance", 1e-5))
        self.membership_tolerance = float(cfg.get("compact_membership_tolerance", 1e-8))
        self.fleet_closure_enforced = self.method != "B"
        if self.fleet_closure_enforced and "fleet_size" not in data:
            raise ValueError("C0/C0V/D require explicit fleet_size; missing node0 closure is not permitted")
        self.fleet_size = None
        if "fleet_size" in data:
            fleet = np.asarray(data["fleet_size"], float)
            if fleet.size != 1 or not np.isfinite(fleet).all() or float(fleet.item()) < 0:
                raise ValueError("fleet_size must be a finite nonnegative scalar")
            self.fleet_size = float(fleet.item())
        self.lo = np.zeros(self.n) if self.method == "D" else np.asarray(spec["lower"], float)
        self.hi = np.zeros(self.n) if self.method == "D" else np.asarray(spec["upper"], float)
        if any(x.shape != (self.n,) for x in (self.d, self.predicted, self.station_capacity, self.lo, self.hi)):
            raise ValueError("All station arrays must have identical one-dimensional shape")
        if not np.isfinite(np.r_[self.d, self.predicted, self.station_capacity, self.lo, self.hi]).all():
            raise ValueError("Station arrays must be finite")
        if self.road.shape != (self.n + 1, self.n + 1) or not np.isfinite(self.road).all() or np.any(self.road < 0):
            raise ValueError("road_distance_km must be a finite nonnegative depot-plus-station matrix")
        if np.any(self.lo > self.hi) or min(self.capacity, self.distance_cost) < 0 or min(self.plus, self.minus) <= 0:
            raise ValueError("Invalid box, capacity, or objective penalties")
        if np.any(self.predicted + self.lo < -self.membership_tolerance) or np.any(self.predicted + self.hi > self.station_capacity + self.membership_tolerance):
            raise ValueError("Error support violates physical station inventories")
        if np.any(self.target < -self.membership_tolerance) or np.any(self.target > self.station_capacity + self.membership_tolerance):
            raise ValueError("Target inventory violates physical station capacity")
        if self.method == "B" and (spec.get("groups") or spec.get("linear_rows") or spec.get("l1_budget") is not None):
            raise ValueError("B must be the ordinary box without aggregate constraints")
        if self.method == "D":
            node0 = self.fleet_size - float(self.predicted.sum())
            if node0 < -self.membership_tolerance or node0 > self.fleet_size + self.membership_tolerance:
                raise ValueError("Nominal D inventory implies a physically invalid node0 stock")
        self.compact = None
        if self.method.startswith("C"):
            # Every possible C witness must close to a nonnegative node0.
            # The lower system sum bound follows already from station stocks
            # being nonnegative.  Require the upper bound explicitly instead
            # of silently accepting old, incompletely closed input files.
            global_rows = [g for g in spec.get("groups", [])
                           if len(g["indices"]) == self.n and set(map(int, g["indices"])) == set(range(self.n))]
            physical_upper = self.fleet_size - float(self.predicted.sum())
            if not global_rows or min(float(g["upper"]) for g in global_rows) > physical_upper + self.membership_tolerance:
                raise ValueError("C0/C0V require a global error upper bound <= fleet_size - sum(predicted)")
            compact_data = {"demand_nominal": self.d, "predicted": self.predicted,
                            "station_capacity": self.station_capacity}
            self.compact = CompactConservationOracle(compact_data, dict(cfg, compact_verify_physical=False), spec)
        self.emission0 = -self.minus * (self.d - self.hi)
        self.emission1 = self.plus * (self.d - self.lo)
        self.standalone = np.maximum(self.emission0, self.emission1)
        self.transition = self.capacity * (self.plus + self.minus)
        self.cache_limit = max(1, int(cfg.get("oracle_cache_size", 2048)))
        self.cache = OrderedDict()
        self.stats = dict(calls=0, cache_hits=0, box_shortcuts=0, compact_calls=0,
                          uncertified_calls=0, physical_checks=0, seconds=0.,
                          cache_limit=self.cache_limit, cache_evictions=0,
                          setup_seconds=time.perf_counter() - started)

    def contains(self, error):
        e = np.asarray(error, float)
        if e.shape != (self.n,) or not np.isfinite(e).all():
            return False
        if self.fleet_closure_enforced:
            node0 = self.fleet_size - float((self.predicted + e).sum())
            if node0 < -self.membership_tolerance or node0 > self.fleet_size + self.membership_tolerance:
                return False
        if self.compact is not None:
            return self.compact.contains(e, tol=self.membership_tolerance)
        return bool(np.all(e >= self.lo - self.membership_tolerance) and
                    np.all(e <= self.hi + self.membership_tolerance))

    def _fleet_diagnostics(self, error):
        station_total = float((self.predicted + np.asarray(error, float)).sum())
        node0 = None if self.fleet_size is None else self.fleet_size - station_total
        valid = None if node0 is None else bool(-self.membership_tolerance <= node0 <= self.fleet_size + self.membership_tolerance)
        return dict(fleet_size=self.fleet_size, node0_stock=node0,
                    station_total_before=station_total, node0_physical=valid,
                    fleet_closure_enforced=self.fleet_closure_enforced,
                    fleet_closure_relaxed=self.method == "B",
                    fleet_closure_verified=False)

    def _routes(self, routes):
        routes = tuple(tuple(int(i) for i in route) for route in routes)
        flat = [i for route in routes for i in route]
        if len(flat) != len(set(flat)) or any(i < 0 or i >= self.n for i in flat):
            raise ValueError("Routes must be duplicate-free optional physical station visits")
        if "truck_count" in self.cfg and len(routes) != int(self.cfg["truck_count"]):
            raise ValueError("Wrong number of truck routes; unused trucks need empty routes")
        return routes

    def route_lengths(self, routes):
        result = []
        for route in routes:
            nodes = np.asarray([0] + [i + 1 for i in route] + [0], int)
            result.append(float(self.road[nodes[:-1], nodes[1:]].sum()))
        return np.asarray(result)

    def _box(self, routes):
        """Exact two-state path dual with its physical-box corner witness."""
        states = (self.emission1 > self.emission0).astype(np.int8)
        value = float(self.standalone.sum())
        for route in routes:
            a = b = 0.
            parents = []
            for i in route:
                parents.append((int(b > a), int(b > a - self.transition)))
                a, b = max(a, b) + self.emission0[i], max(a - self.transition, b) + self.emission1[i]
            value += max(a, b) - float(self.standalone[list(route)].sum())
            state = int(b > a)
            for k in range(len(route) - 1, -1, -1):
                states[route[k]] = state
                state = parents[k][state]
        error = np.where(states, self.lo, self.hi)
        return float(value), error, states

    def _verify_physical(self, routes, answer):
        error = np.asarray(answer["error"], float)
        if not self.contains(error):
            raise RuntimeError("Returned witness is outside its complete uncertainty set")
        # Keep the process-wide HiGHS scheduler consistent with the one-thread
        # compact MILP.  Initializing a default-thread physical LP first can
        # otherwise make a later explicit one-thread solve return Not Set.
        import oracle as physical_module
        original_lp = physical_module.linprog
        def strict_lp(*args, **kwargs):
            options = dict(kwargs.pop("options", {}) or {})
            options.update(threads=1, primal_feasibility_tolerance=1e-9,
                           dual_feasibility_tolerance=1e-9)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return original_lp(*args, options=options, **kwargs)
        physical_module.linprog = strict_lp
        try:
            primal = solve_recourse(routes, capacity=self.capacity, plus=self.plus, minus=self.minus,
                                    minimal_handling=False, inventory=self.predicted + error,
                                    station_capacity=self.station_capacity, targets=self.target)
        finally:
            physical_module.linprog = original_lp
        discrepancy = abs(float(primal["value"]) - float(answer["value"]))
        if discrepancy > self.tolerance:
            raise RuntimeError("Physical primal disagrees with risk lower-bound witness")
        fleet = self._fleet_diagnostics(error)
        fleet["station_total_after"] = float(np.asarray(primal["inventory_after"], float).sum())
        fleet["fleet_error"] = (None if self.fleet_size is None else
                                 fleet["station_total_after"] + fleet["node0_stock"] - self.fleet_size)
        # The endpoint trucks are empty, so the final fleet contains only
        # station stock and customer-transit stock.  Intermediate loads are
        # retained by the primal's route load-balance constraints.
        if self.fleet_closure_enforced:
            if not fleet["node0_physical"] or abs(fleet["fleet_error"]) > self.tolerance:
                raise RuntimeError("Final recourse violates complete station+node0 fleet closure")
            fleet["fleet_closure_verified"] = True
        self.stats["physical_checks"] += 1
        return dict(answer, physical_primal_verified=True, physical_primal_value=float(primal["value"]),
                    physical_discrepancy=discrepancy, physical_verification_seconds=float(primal["seconds"]),
                    **fleet)

    def evaluate(self, routes, time_limit=1., verify_physical=False):
        started = time.perf_counter()
        self.stats["calls"] += 1
        routes = self._routes(routes)
        old = self.cache.get(routes)
        if old is not None and old["certified"]:
            self.stats["cache_hits"] += 1
            answer = dict(old, cache_hit=True)
        else:
            box_value, box_error, states = self._box(routes)
            if self.contains(box_error):
                self.stats["box_shortcuts"] += 1
                answer = dict(value=box_value, upper_bound=box_value, error=box_error.tolist(),
                              gap=0., certified=True, strict_membership=True,
                              backend="exact_nominal_path_dp" if self.method == "D" else "exact_box_path_dp",
                              solver_status=0, solver_wall_seconds=0., box_upper_bound=box_value,
                              physical_primal_verified=False, cache_hit=False)
            else:
                self.stats["compact_calls"] += 1
                # The box value remains a valid complete-set upper bound.
                answer = self.compact.evaluate(routes, time_limit=max(.001, float(time_limit)))
                answer = dict(answer, upper_bound=min(float(answer["upper_bound"]), box_value),
                              box_upper_bound=box_value, cache_hit=False)
                # Retain previously established bounds on repeated evaluations.
                if old is not None:
                    answer["upper_bound"] = min(answer["upper_bound"], old["upper_bound"])
                    if old["value"] > answer["value"]:
                        answer["value"] = old["value"]
                        answer["error"] = old["error"]
                        answer["retained_previous_witness"] = True
                if answer["upper_bound"] < answer["value"] - self.tolerance:
                    raise RuntimeError("Risk bound interval is numerically inconsistent")
                answer["upper_bound"] = max(answer["upper_bound"], answer["value"])
                answer["gap"] = max(0., answer["upper_bound"] - answer["value"])
                answer["certified"] = answer["gap"] <= self.tolerance
                if not answer["certified"]:
                    self.stats["uncertified_calls"] += 1
            if not self.contains(answer["error"]):
                raise RuntimeError("Oracle witness violates the complete uncertainty set")
            if not np.isfinite([answer["value"], answer["upper_bound"]]).all():
                raise RuntimeError("Expected finite complete-set risk bounds")
            lengths = self.route_lengths(routes)
            travel_cost = self.distance_cost * float(lengths.sum())
            answer.update(method=self.method, lower_bound=float(answer["value"]),
                          route_distance_km=float(lengths.sum()), truck_distances_km=lengths.tolist(),
                          travel_cost=travel_cost, objective_lower=float(answer["value"] + travel_cost),
                          objective_upper=float(answer["upper_bound"] + travel_cost),
                          outer_global_optimality_proven=False,
                          physical_bounds_checked=True, all_station_losses_included=True)
            answer.update(self._fleet_diagnostics(answer["error"]))
        if verify_physical and not answer.get("physical_primal_verified", False):
            answer = self._verify_physical(routes, answer)
        elapsed = time.perf_counter() - started
        answer["seconds"] = elapsed
        self.stats["seconds"] += elapsed
        self.cache[routes] = dict(answer)
        self.cache.move_to_end(routes)
        if len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
            self.stats["cache_evictions"] += 1
        return dict(answer)
