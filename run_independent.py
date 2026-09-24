"""Independent regional ALNS on full-network snapshots; never opens actuals.

Every run owns one current and one incumbent solution. There is no policy pool,
no transfer of routes between arms, and no after-realization route selection.
Inner-risk bounds are not outer routing optimality bounds.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "three_set_study"))
from routing import route_lengths, _nearest_order, _two_opt
from robust_evaluator import RobustEvaluator

ARMS = ("B", "C0", "C0V", "D")
CONFIG = dict(truck_count=4, truck_capacity=30, shortage_penalty=10.,
              surplus_penalty=10., distance_cost_per_km=1., min_stops_per_truck=0,
              oracle_absolute_tolerance=1e-5, oracle_mip_relative_tolerance=1e-10,
              oracle_threads=1, compact_verify_physical=False)
OPERATORS = ("relocate", "swap", "reverse", "remove_optional",
             "insert_optional_pair", "regional_destroy_repair", "cross_region_block")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(routes):
    return tuple(sorted(tuple(map(int, route)) for route in routes))


def feasible(routes, road, cap):
    flat = [i for route in routes for i in route]
    return (len(routes) == 4 and len(flat) == len(set(flat)) and
            all(0 <= i < 567 for i in flat) and
            bool(np.all(route_lengths(routes, road) <= cap + 1e-8)))


def initial_routes(data, cap, seed):
    """One geography-guided seed, identical rule and inputs for every arm.

    Start from nominal nonzero needs, but no station is mandatory during search.
    Four groups guide construction; later moves freely cross their boundaries.
    Prune the smallest absolute nominal need when a route exceeds the common cap.
    """
    road, d = data["road_distance_km"], data["demand_nominal"]
    labels = data["region_labels"]
    rng = np.random.default_rng(seed)
    routes = []
    for region in range(4):
        nodes = np.flatnonzero((labels == region) & (np.abs(d) > 1e-8))
        # Stable random tie order is seed dependent, arm independent.
        nodes = nodes[rng.permutation(len(nodes))]
        route = _two_opt(_nearest_order(nodes, road), road, passes=12)
        while route and route_lengths([route], road)[0] > cap + 1e-8:
            remove = min(range(len(route)), key=lambda j: (abs(float(d[route[j]])), -float(road[0, route[j]+1])))
            route.pop(remove)
        routes.append(route)
    if not feasible(routes, road, cap): raise AssertionError("Invalid full-network initializer")
    return routes


def insert_near(route, node, road, rng):
    ids = np.asarray([0] + [i+1 for i in route] + [0])
    increments = road[ids[:-1], node+1] + road[node+1, ids[1:]] - road[ids[:-1], ids[1:]]
    if rng.random() < .8:
        at = int(np.argmin(increments))
    else:
        at = int(rng.integers(len(route)+1))
    route.insert(at, int(node))


def weighted_node(nodes, d, rng):
    if not len(nodes): return None
    # Positive floor ensures every unvisited B-class station is eligible.
    weights = 1. + np.abs(d[nodes])
    return int(rng.choice(nodes, p=weights/weights.sum()))


def proposal(current, operator, data, rng):
    routes = [list(route) for route in current]
    road, d, labels = data["road_distance_km"], data["demand_nominal"], data["region_labels"]
    occupied = [k for k, route in enumerate(routes) if route]
    if operator in ("relocate", "swap", "reverse", "remove_optional", "cross_region_block") and not occupied:
        return None
    if operator == "relocate":
        a = int(rng.choice(occupied)); node = routes[a].pop(int(rng.integers(len(routes[a]))))
        b = int(rng.integers(4)); insert_near(routes[b], node, road, rng)
    elif operator == "swap":
        a, b = [int(rng.choice(occupied)) for _ in range(2)]
        ia, ib = int(rng.integers(len(routes[a]))), int(rng.integers(len(routes[b])))
        routes[a][ia], routes[b][ib] = routes[b][ib], routes[a][ia]
    elif operator == "reverse":
        eligible = [k for k in occupied if len(routes[k]) > 1]
        if not eligible: return None
        k = int(rng.choice(eligible)); a, b = sorted(rng.choice(len(routes[k]), 2, replace=False))
        routes[k][a:b+1] = reversed(routes[k][a:b+1])
    elif operator == "remove_optional":
        k = int(rng.choice(occupied)); count = min(len(routes[k]), int(rng.integers(1, 5)))
        positions = sorted(rng.choice(len(routes[k]), count, replace=False), reverse=True)
        for position in positions: routes[k].pop(int(position))
    elif operator in ("insert_optional_pair", "regional_destroy_repair"):
        region = int(rng.integers(4))
        if operator == "regional_destroy_repair":
            visits = [(k, j) for k, route in enumerate(routes) for j, i in enumerate(route) if labels[i] == region]
            if visits:
                chosen = rng.choice(len(visits), min(len(visits), int(rng.integers(3, 9))), replace=False)
                for k, j in sorted([visits[int(q)] for q in chosen], reverse=True): routes[k].pop(j)
        visited = {i for route in routes for i in route}
        all_unvisited = np.asarray([i for i in range(567) if i not in visited], dtype=int)
        if not len(all_unvisited): return None
        regional = all_unvisited[labels[all_unvisited] == region]
        eligible = regional if len(regional) and rng.random() < .65 else all_unvisited
        first = weighted_node(eligible, d, rng)
        complement = all_unvisited[(all_unvisited != first) & (d[all_unvisited]*d[first] < 0)]
        if not len(complement): complement = all_unvisited[all_unvisited != first]
        second = weighted_node(complement, d, rng)
        k = int(rng.integers(4))
        pair = [first] + ([] if second is None else [second])
        if len(pair) == 2 and d[pair[0]] > d[pair[1]]: pair.reverse()
        # Paired insertion also works for an empty truck, where a singleton has no benefit.
        at = int(rng.integers(len(routes[k])+1))
        routes[k][at:at] = pair
    elif operator == "cross_region_block":
        a = int(rng.choice(occupied)); b = int(rng.choice([k for k in range(4) if k != a]))
        start = int(rng.integers(len(routes[a]))); size = min(int(rng.integers(1, 6)), len(routes[a])-start)
        block = routes[a][start:start+size]; del routes[a][start:start+size]
        at = int(rng.integers(len(routes[b])+1)); routes[b][at:at] = block
    return routes


def optimize(case, method, seed, settings, output):
    started = time.perf_counter()
    directory = Path(case["directory"])
    data = dict(np.load(directory/"instance.npz", allow_pickle=False))
    if any("actual" in key.lower() for key in data): raise AssertionError("Outcome entered optimization")
    metadata, specs = read(directory/"metadata.json"), read(directory/"uncertainty_specs.json")
    cap = float(metadata["distance_cap_km"])
    evaluator = RobustEvaluator(data, CONFIG, specs[method], method)
    current = initial_routes(data, cap, seed)
    initial = [route.copy() for route in current]
    initial_eval = evaluator.evaluate(current, time_limit=settings["initial_oracle_seconds"])
    current_eval = initial_eval
    best, best_eval = [r.copy() for r in current], initial_eval
    rng = np.random.default_rng(seed)
    weights = np.ones(len(OPERATORS)); rewards = np.zeros(len(OPERATORS)); uses = np.zeros(len(OPERATORS))
    counts = {op: dict(proposed=0, feasible=0, accepted=0, improved=0) for op in OPERATORS}
    trace = []; unresolved = 0; calls = 0
    search_start = time.perf_counter()
    for iteration in range(settings["max_iterations"]):
        elapsed = time.perf_counter()-search_start
        if elapsed >= settings["search_seconds"]: break
        op_index = int(rng.choice(len(OPERATORS), p=weights/weights.sum()))
        op = OPERATORS[op_index]; counts[op]["proposed"] += 1; uses[op_index] += 1
        candidate = proposal(current, op, data, rng)
        if candidate is None or canonical(candidate) == canonical(current) or not feasible(candidate, data["road_distance_km"], cap): continue
        counts[op]["feasible"] += 1
        limit = min(settings["candidate_oracle_seconds"], max(.005, settings["search_seconds"]-elapsed))
        evaluation = evaluator.evaluate(candidate, time_limit=limit)
        calls += 1
        if not evaluation["certified"]: unresolved += 1
        score, current_score = evaluation["objective_upper"], current_eval["objective_upper"]
        progress = max((iteration+1)/settings["max_iterations"], elapsed/settings["search_seconds"])
        temperature = max(.05, settings["initial_temperature"] * (1-progress)**2)
        accepted = score < current_score-1e-8 or rng.random() < math.exp(min(0., (current_score-score)/temperature))
        if accepted:
            current, current_eval = candidate, evaluation
            counts[op]["accepted"] += 1; rewards[op_index] += 1.
        if score < best_eval["objective_upper"]-1e-8:
            best, best_eval = [r.copy() for r in candidate], evaluation
            counts[op]["improved"] += 1; rewards[op_index] += 5.
            trace.append(dict(iteration=iteration, seconds=elapsed, upper=score,
                              lower=evaluation["objective_lower"], certified=evaluation["certified"]))
        if (iteration+1) % 50 == 0:
            weights = .8*weights + .2*(1+rewards/np.maximum(uses, 1))
            rewards[:] = 0.; uses[:] = 0.
    search_elapsed = time.perf_counter()-search_start
    final = evaluator.evaluate(best, time_limit=settings["final_oracle_seconds"], verify_physical=True)
    # Certification refines this one incumbent; it never chooses a different route.
    labels = data["region_labels"]
    flat = [i for route in best for i in route]
    crossings = sum(int(labels[a] != labels[b]) for route in best for a, b in zip(route, route[1:]))
    record = dict(case=case["case"], sample_id=case["sample_id"], origin=metadata["origin"], method=method, seed=seed,
                  station_count=567, routes=best, initial_routes_sha256=hashlib.sha256(json.dumps(initial).encode()).hexdigest(),
                  initial=initial_eval, risk=final, distance_cap_km=cap, route_lengths_km=route_lengths(best, data["road_distance_km"]),
                  visited_stations=len(flat), visited_original_B_stations=int(sum(not data["original_candidate_mask"][i] for i in flat)),
                  region_crossings=crossings, all_stations_optional=True, independent_search=True, policy_pool=False,
                  search_seconds=search_elapsed, total_seconds=time.perf_counter()-started, evaluated_proposals=calls,
                  uncertified_proposals=unresolved, operators=counts, improvement_trace=trace, evaluator_stats=evaluator.stats,
                  input_hashes=case["output_sha256"], actual_arrays_opened=False,
                  route_global_optimality_certified=False, outcome_scope="static full-network target-deviation recourse; no 10-minute execution claim")
    if not feasible(best, data["road_distance_km"], cap): raise AssertionError("Invalid final routes")
    write(output, record)
    return dict(case=case["case"], method=method, seed=seed, certified=final["certified"],
                proposals=calls, improved=float(initial_eval["objective_upper"]-final["objective_upper"]),
                seconds=record["total_seconds"], path=str(output))


def worker(job):
    case, method, seed, settings, output = job
    try: return optimize(case, method, seed, settings, output)
    except Exception as exc:
        import traceback
        failure = dict(case=case["case"], method=method, seed=seed, failure=repr(exc), traceback=traceback.format_exc())
        write(Path(output).with_suffix(".failure.json"), failure)
        return failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "main"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--search-seconds", type=float, default=12.)
    parser.add_argument("--max-iterations", type=int, default=1000000)
    args = parser.parse_args()
    index = read(HERE/"inputs/index.json")
    settings = dict(search_seconds=args.search_seconds, max_iterations=args.max_iterations,
                    candidate_oracle_seconds=.15, initial_oracle_seconds=3., final_oracle_seconds=10., initial_temperature=12.)
    # Preflight is full-size technical verification, not an extra small experiment.
    selected = index["cases"] if args.stage == "main" else index["cases"][::72]
    out = HERE/"results"/args.stage
    sources = [Path(__file__), HERE/"robust_evaluator.py", HERE/"prepare_inputs.py", HERE/"inputs/protocol_freeze.json", HERE/"inputs/index.json"]
    protocol = dict(stage=args.stage, config=CONFIG, settings=settings, arms=ARMS, main_seed=17,
                    parallel_workers=args.workers,
                    repeat_seeds=[29,43] if args.stage == "main" else [], repeat_origins="every 2 hours: sample 176,188,...,308",
                    selected_cases=[case["case"] for case in selected], source_sha256={str(p): sha(p) for p in sources},
                    selection="same input/initializer/operator rules; independent current and incumbent per run; upper-bound score under timeouts",
                    accounting="common search wall limits per run; initial/final inner certification reported separately; parallel worker count recorded",
                    routing_optimality="heuristic outer solution; final certificates apply only to inner adversary for the fixed route",
                    source_outcomes_read=False)
    frozen = out/"protocol.json"
    if frozen.exists() and read(frozen) != clean(protocol): raise RuntimeError("Protocol changed; retain old stage and use a new explicitly named campaign")
    write(frozen, protocol)
    jobs = []
    for case in selected:
        seeds = [17]
        if args.stage == "main" and (case["sample_id"]-176) % 12 == 0: seeds += [29,43]
        for seed in seeds:
            for method in ARMS:
                output = out/"runs"/f"{case['case']}__{method}__s{seed}.json"
                if not output.exists(): jobs.append((case, method, seed, settings, str(output)))
    write(out/"progress.json", dict(expected_runs=sum(4*(3 if args.stage=="main" and (c["sample_id"]-176)%12==0 else 1) for c in selected), pending=len(jobs), status="running"))
    completed, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result(); completed.append(result)
            if "failure" in result: failures.append(result)
            print(json.dumps(clean(result), ensure_ascii=False), flush=True)
            write(out/"progress.json", dict(expected_runs=672 if args.stage=="main" else 8,
                    completed_on_this_invocation=len(completed), remaining_on_this_invocation=len(jobs)-len(completed),
                    existing_run_files=len(list((out/"runs").glob("*.json"))) if (out/"runs").exists() else 0,
                    failures=failures, status="running"))
    paths = sorted((out/"runs").glob("*.json"))
    paths = [p for p in paths if not p.name.endswith(".failure.json")]
    expected = 672 if args.stage=="main" else 8
    manifest = dict(stage=args.stage, expected=expected, completed=len(paths), failures=failures,
                    complete=len(paths)==expected and not failures, routes_frozen_before_replay=True,
                    hashes={str(path): sha(path) for path in paths}, source_sha256=protocol["source_sha256"])
    write(out/"optimization_freeze.json", manifest)
    write(out/"progress.json", dict(status="complete" if manifest["complete"] else "incomplete", **manifest))
    if not manifest["complete"]: raise RuntimeError("Campaign incomplete; failures retained")


if __name__ == "__main__": main()
