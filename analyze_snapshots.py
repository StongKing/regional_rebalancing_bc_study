"""Frozen-route large-network snapshot replay, never a closed-loop service test.

The complete optimization manifest, all run/input hashes, and expected arm/seed
combinations must pass before the actual member is opened. The same-route
mechanism check only evaluates the already frozen B route; it selects no route.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from pathlib import Path
import time
import warnings

import numpy as np

from run_independent import ARMS, CONFIG, HERE, canonical, feasible, read, sha, write
from robust_evaluator import RobustEvaluator
import oracle
import compact_oracle
import routing

MAIN = HERE / "results/main"
OUT = HERE / "results/snapshot_analysis"
EXPORT = HERE.parents[1] / "outputs/01a0a96b-0035-7f22-b360-8fc2b56e2ddf/rebalancing_epoch_86/all_predictions.npz"
METRICS = ("target_deviation_bikes", "distance_km", "weighted_total_cost")
PAIRS = (("B", "C0"), ("C0", "C0V"), ("B", "C0V"), ("D", "B"), ("D", "C0"), ("D", "C0V"))
TOL = 1e-6


def csv_write(path, rows):
    if not rows: raise ValueError("Required output table is empty")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def check_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError("Frozen file changed or missing: " + name)


def verify_prediction_archive(index):
    """Read only prediction/identifier/time members; explicitly never actual."""
    allowed = ("sample_id", "prediction", "origin_time_ns", "target_time_ns")
    with np.load(EXPORT, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in allowed}
    ids = np.asarray(values["sample_id"]).reshape(-1)
    checked = 0
    for case in index["cases"]:
        matches = np.flatnonzero(ids == case["sample_id"])
        if len(matches) != 1: raise RuntimeError("Archive sample identifier is ambiguous")
        position = int(matches[0])
        directory = Path(case["directory"])
        metadata = read(directory/"metadata.json")
        with np.load(directory/"instance.npz", allow_pickle=False) as archive:
            prediction = archive["predicted_original"].copy()
        np.testing.assert_array_equal(values["prediction"][position, 1:, 0], prediction)
        for field, key in (("origin", "origin_time_ns"), ("target", "target_time_ns")):
            expected_ns = int(np.datetime64(metadata[field].replace(" ", "T"), "ns").astype(np.int64))
            actual_ns = int(np.asarray(values[key][position]).reshape(-1)[0])
            if actual_ns != expected_ns:
                raise AssertionError(f"Archive {field} timestamp differs for {case['case']}")
        checked += 1
    return dict(checked_cases=checked, original_prediction_matches=True,
                origin_target_times_match=True, opened_members=list(allowed), actual_opened=False)


def preflight():
    freeze_path = MAIN/"optimization_freeze.json"
    if not freeze_path.exists(): raise RuntimeError("Optimization not frozen; outcome access prohibited")
    freeze = read(freeze_path)
    if not freeze.get("complete") or freeze.get("expected") != 672 or freeze.get("completed") != 672 or freeze.get("failures"):
        raise RuntimeError("All 672 frozen optimization records are required before outcome access")
    if not freeze.get("routes_frozen_before_replay"): raise RuntimeError("Route freeze not declared")
    check_hashes(freeze["hashes"]); check_hashes(freeze["source_sha256"])
    index = read(HERE/"inputs/index.json")
    if len(index["cases"]) != 144: raise RuntimeError("Expected all 144 large instances")
    expected = {(c["case"], arm, seed) for c in index["cases"]
                for seed in ([17, 29, 43] if (c["sample_id"]-176) % 12 == 0 else [17]) for arm in ARMS}
    runs = {}
    for filename in freeze["hashes"]:
        row = read(filename)
        key = (row["case"], row["method"], row["seed"])
        if key in runs: raise RuntimeError("Duplicate frozen run")
        if row.get("actual_arrays_opened") or not row.get("independent_search") or row.get("policy_pool"):
            raise RuntimeError("Unexpected information pattern in optimizer")
        runs[key] = row
    if set(runs) != expected: raise RuntimeError("Frozen arm/seed/origin combinations are incomplete")
    for case in index["cases"]:
        directory = Path(case["directory"])
        check_hashes({str(directory/name): value for name, value in case["output_sha256"].items()})
        for key, row in runs.items():
            if key[0] == case["case"] and row["input_hashes"] != case["output_sha256"]:
                raise RuntimeError("Optimizer input hashes differ from prepared data")
    original = read(HERE/"inputs/protocol_freeze.json")["original_protocol"]["source_sha256"]
    source_hash = next((value for name, value in original.items() if Path(name).resolve() == EXPORT.resolve()), None)
    if source_hash is None or sha(EXPORT) != source_hash:
        raise RuntimeError("Original prediction/outcome archive fingerprint changed")
    prediction_audit = verify_prediction_archive(index)
    return index, freeze, runs, source_hash, prediction_audit


def actual_inventory_snapshot(actual, position, data):
    """Validate the archived 0-node plus stations against the frozen fleet."""
    full = np.asarray(actual[position, :, 0], float)
    fleet = float(np.asarray(data["fleet_size"]))
    if full.shape != (568,) or not np.isfinite(full).all() or np.any(full < -TOL):
        raise ValueError("Actual full-system inventory is malformed or negative")
    if abs(float(full.sum())-fleet) > TOL:
        raise ValueError("Actual node0 plus station inventory does not close to frozen fleet")
    if not -TOL <= float(full[0]) <= fleet+TOL:
        raise ValueError("Actual node0 inventory outside [0, fleet]")
    return full[1:].copy(), float(full[0])


def physical_check(routes, inventory, data, cap):
    """Full 567-station physical LP plus independent inventory/load/region audits."""
    inventory = np.asarray(inventory, float)
    target, capacities = np.asarray(data["target_inventory"]), np.asarray(data["station_capacity"])
    fleet = float(np.asarray(data["fleet_size"]))
    node0 = fleet-float(inventory.sum())
    if inventory.shape != (567,) or not np.isfinite(inventory).all(): raise ValueError("Bad actual snapshot")
    if np.any(inventory < -TOL) or np.any(inventory > capacities+TOL):
        raise ValueError("Actual snapshot violates inherited physical station capacities; do not silently clip")
    if not np.isfinite(fleet) or not -TOL <= node0 <= fleet+TOL:
        raise ValueError("Actual snapshot implies invalid 0-node inventory")
    if not feasible(routes, data["road_distance_km"], cap): raise ValueError("Frozen route is infeasible")
    original_lp = oracle.linprog
    def strict_lp(*args, **kwargs):
        options = dict(kwargs.pop("options", {}) or {})
        options.update(threads=1, primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return original_lp(*args, options=options, **kwargs)
    oracle.linprog = strict_lp
    try:
        # Equal-optimum tie-breaking minimizes total handling for every arm.
        primal = oracle.solve_recourse(routes, inventory=inventory, station_capacity=capacities,
                    targets=target, capacity=CONFIG["truck_capacity"], plus=CONFIG["shortage_penalty"],
                    minus=CONFIG["surplus_penalty"], minimal_handling=True)
    finally:
        oracle.linprog = original_lp
    y = np.asarray(primal["y"])
    after = inventory+y
    shortage, surplus = np.maximum(target-after, 0.), np.maximum(after-target, 0.)
    if not np.allclose(shortage, primal["shortage"], atol=TOL, rtol=0) or not np.allclose(surplus, primal["surplus"], atol=TOL, rtol=0):
        raise AssertionError("Reported slacks differ from actual full-network target deviations")
    visited = np.zeros(567, bool); labels = np.asarray(data["region_labels"], int)
    flows = np.zeros((4, 4)); load_violation = 0.; endpoint_violation = 0.
    for route, reported in zip(routes, primal["route_loads"]):
        ids = np.asarray(route, int); visited[ids] = True
        loads = -np.cumsum(y[ids])
        if not np.allclose(loads, reported, atol=TOL, rtol=0): raise AssertionError("Truck load path mismatch")
        if len(loads):
            load_violation = max(load_violation, float(np.maximum(-loads, 0).max()),
                                 float(np.maximum(loads-CONFIG["truck_capacity"], 0).max()))
            endpoint_violation = max(endpoint_violation, abs(float(loads[-1])))
        # Empty depot departure/return carry zero, hence only station-to-station
        # crossings contribute to interregional inventory transfer.
        for j, (i, k) in enumerate(zip(route, route[1:])):
            if labels[i] != labels[k]: flows[labels[i], labels[k]] += loads[j]
    regional_net = np.array([y[labels == g].sum() for g in range(4)])
    regional_flow_net = flows.sum(axis=0)-flows.sum(axis=1)
    region_balance_error = float(np.max(np.abs(regional_net-regional_flow_net)))
    inventory_conservation_error = abs(float(after.sum()-inventory.sum()))
    global_fleet_closure_error = abs(float(after.sum()+node0-fleet))
    unvisited_handling = float(np.max(np.abs(y[~visited]), initial=0.))
    negative = float(np.maximum(-after, 0.).sum()); overflow = float(np.maximum(after-capacities, 0.).sum())
    violations = [load_violation, endpoint_violation, region_balance_error,
                  inventory_conservation_error, global_fleet_closure_error, unvisited_handling, negative, overflow]
    if max(violations) > TOL: raise AssertionError("Physical/region conservation verification failed")
    dual = oracle.deterministic_value(routes, target-inventory, CONFIG["truck_capacity"],
                                      CONFIG["shortage_penalty"], CONFIG["surplus_penalty"])
    value = float(CONFIG["shortage_penalty"]*shortage.sum()+CONFIG["surplus_penalty"]*surplus.sum())
    if max(abs(value-primal["value"]), abs(value-dual)) > 1e-5:
        raise AssertionError("Full-network physical primal/dual disagree")
    lengths = routing.route_lengths(routes, data["road_distance_km"])
    travel_cost = float(lengths.sum())*CONFIG["distance_cost_per_km"]
    return dict(shortage_bikes=float(shortage.sum()), surplus_bikes=float(surplus.sum()),
                target_deviation_bikes=float(shortage.sum()+surplus.sum()), distance_km=float(lengths.sum()),
                weighted_total_cost=value+travel_cost, weighted_target_loss=value,
                regional_net_delivery=regional_net, cross_region_load_flow=flows,
                negative_inventory=negative, capacity_overflow=overflow,
                load_violation=load_violation, endpoint_load_violation=endpoint_violation,
                unvisited_handling=unvisited_handling, regional_balance_error=region_balance_error,
                inventory_conservation_error=inventory_conservation_error,
                node0_inventory=node0, fleet_size=fleet, global_fleet_closure_error=global_fleet_closure_error,
                net_delivery=y, inventory_after=after, route_loads=primal["route_loads"])


def mechanism_worker(job):
    case, routes, limit, output = job
    directory = Path(case["directory"])
    with np.load(directory/"instance.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    specs = read(directory/"uncertainty_specs.json")
    results = {}
    for method in ("B", "C0", "C0V"):
        evaluator = RobustEvaluator(data, CONFIG, specs[method], method)
        results[method] = evaluator.evaluate(routes, time_limit=limit, verify_physical=True)
    for parent, child in (("B", "C0"), ("C0", "C0V")):
        if results[parent]["upper_bound"] < results[child]["lower_bound"]-1e-5:
            raise AssertionError("Fixed-route nested risk bound intervals are inconsistent")
    record = dict(case=case["case"], sample_id=case["sample_id"], source_method="B", source_seed=17,
                  route_canonical=canonical(routes), risk=results, actual_outcomes_read=False,
                  route_selection_performed=False, outer_optimality_claim=False)
    write(output, record)
    return case["case"]


def summarize(rows):
    result = []
    for arm in ARMS:
        selected = [r for r in rows if r["method"] == arm]
        result.append(dict(method=arm, count=len(selected),
                           **{metric: float(np.mean([r[metric] for r in selected])) for metric in METRICS},
                           maximum_weighted_total_cost=max(r["weighted_total_cost"] for r in selected),
                           shortage_bikes=float(np.mean([r["shortage_bikes"] for r in selected])),
                           surplus_bikes=float(np.mean([r["surplus_bikes"] for r in selected]))))
    return result


def paired(rows):
    lookup = {(r["case"], r["seed"], r["method"]): r for r in rows}
    origins = sorted(set((r["case"], r["seed"]) for r in rows))
    result = []
    for baseline, challenger in PAIRS:
        for metric in METRICS:
            delta = np.array([lookup[(case, seed, challenger)][metric]-lookup[(case, seed, baseline)][metric]
                              for case, seed in origins])
            result.append(dict(baseline=baseline, challenger=challenger, metric=metric, count=len(delta),
                               mean_challenger_minus_baseline=float(delta.mean()),
                               wins=int((delta < -TOL).sum()), ties=int((np.abs(delta) <= TOL).sum()),
                               losses=int((delta > TOL).sum()), tolerance=TOL))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mechanism-seconds", type=float, default=10.)
    args = parser.parse_args()
    started = time.perf_counter()
    index, freeze, runs, source_hash, prediction_audit = preflight()
    protocol = dict(optimization_freeze_sha256=sha(MAIN/"optimization_freeze.json"),
                    expected_runs=672, main_origins=144, main_seed=17,
                    stability_origins=[c["sample_id"] for c in index["cases"] if (c["sample_id"]-176)%12 == 0],
                    stability_seeds=[17,29,43], main_metrics=list(METRICS),
                    total_cost_descriptive_statistics=["mean", "observed_maximum"], comparisons=PAIRS,
                    physics="full 567-station physical continuous recourse, fixed route, empty truck start/end, minimum-handling tie break",
                    mechanism="only frozen B seed17 route evaluated in B/C0/C0V, no route reselection or pooling",
                    mechanism_seconds_per_arm=args.mechanism_seconds,
                    source_sha256={str(path): sha(path) for path in [Path(__file__), Path(oracle.__file__), Path(compact_oracle.__file__), Path(routing.__file__)]},
                    outcome_archive_sha256=source_hash,
                    prediction_identity_audit=prediction_audit,
                    outcome_scope="retrospective target-inventory snapshot deviation, not customer unmet demand or continuous closed loop")
    frozen_analysis = OUT/"analysis_freeze.json"
    if frozen_analysis.exists() and read(frozen_analysis) != __import__("run_independent").clean(protocol):
        raise RuntimeError("Analysis protocol changed after freeze; retain previous analysis")
    write(frozen_analysis, protocol)
    jobs = []
    for case in index["cases"]:
        routes = runs[(case["case"], "B", 17)]["routes"]
        path = OUT/"mechanism"/(case["case"]+".json")
        if path.exists():
            record = read(path)
            if canonical(record["route_canonical"]) != canonical(routes): raise RuntimeError("Mechanism route drift")
        else: jobs.append((case, routes, args.mechanism_seconds, str(path)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(mechanism_worker, job) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            case_name = future.result()
            print(f"mechanism {number}/{len(jobs)} {case_name}", flush=True)
    # Recheck all decisions and analysis code immediately before first outcome read.
    check_hashes(freeze["hashes"]); check_hashes(freeze["source_sha256"]); check_hashes(protocol["source_sha256"])
    if sha(EXPORT) != source_hash: raise RuntimeError("Outcome source changed before access")
    access = dict(first_access_utc=datetime.now(timezone.utc).isoformat(), member="actual",
                  optimization_complete=True, frozen_run_count=672, all_hashes_verified=True,
                  mechanism_routes_already_frozen=True, retrospective_outcomes_known_from_earlier_studies=True)
    if not (OUT/"actual_access.json").exists(): write(OUT/"actual_access.json", access)
    with np.load(EXPORT, allow_pickle=False) as archive:
        sample_ids = archive["sample_id"].copy()
        actual = archive["actual"].copy()
    rows, physical_details, coverage = [], [], []
    for case in index["cases"]:
        directory = Path(case["directory"])
        with np.load(directory/"instance.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        positions = np.flatnonzero(sample_ids == case["sample_id"])
        if len(positions) != 1: raise RuntimeError("Actual sample identifier is ambiguous")
        inventory, actual_node0 = actual_inventory_snapshot(actual, int(positions[0]), data)
        specs = read(directory/"uncertainty_specs.json")
        for arm in ("B", "C0", "C0V"):
            evaluator = RobustEvaluator(data, CONFIG, specs[arm], arm)
            coverage.append(dict(case=case["case"], method=arm,
                                 observed_error_inside=evaluator.contains(inventory-data["predicted"])))
        for seed in ([17,29,43] if (case["sample_id"]-176)%12 == 0 else [17]):
            for arm in ARMS:
                frozen_run = runs[(case["case"], arm, seed)]
                replay = physical_check(frozen_run["routes"], inventory, data, case["distance_cap_km"])
                if abs(replay["node0_inventory"]-actual_node0) > TOL:
                    raise AssertionError("Explicit actual node0 differs from physical fleet residual")
                compact = {key: replay[key] for key in METRICS+("shortage_bikes", "surplus_bikes")}
                row = dict(case=case["case"], sample_id=case["sample_id"], origin=case["origin"],
                           method=arm, seed=seed, **compact,
                           inner_lower=frozen_run["risk"]["lower_bound"], inner_upper=frozen_run["risk"]["upper_bound"],
                           inner_certified=bool(frozen_run["risk"]["certified"]),
                           search_seconds=frozen_run["search_seconds"], total_seconds=frozen_run["total_seconds"],
                           evaluated_proposals=frozen_run["evaluated_proposals"])
                rows.append(row)
                physical_details.append(dict(case=case["case"], method=arm, seed=seed, **replay))
        print(f"replay {case['sample_id']-175}/144 {case['case']}", flush=True)
    main_rows = [r for r in rows if r["seed"] == 17]
    stability_rows = [r for r in rows if (r["sample_id"]-176)%12 == 0]
    means, comparisons = summarize(main_rows), paired(main_rows)
    stability = []
    for case in index["cases"]:
        if (case["sample_id"]-176)%12: continue
        for arm in ARMS:
            sample = [r for r in stability_rows if r["case"] == case["case"] and r["method"] == arm]
            if len(sample) != 3: raise AssertionError("Seed stability block incomplete")
            for metric in METRICS:
                values = [r[metric] for r in sample]
                stability.append(dict(case=case["case"], method=arm, metric=metric, seed_count=3,
                                       mean=float(np.mean(values)), minimum=min(values), maximum=max(values),
                                       seed_range=max(values)-min(values)))
    mechanism_rows, mechanism_differences = [], []
    for case in index["cases"]:
        record = read(OUT/"mechanism"/(case["case"]+".json"))
        for arm, risk in record["risk"].items():
            mechanism_rows.append(dict(case=case["case"], evaluation_arm=arm, route_source="B_seed17",
                                       lower_bound=risk["lower_bound"], upper_bound=risk["upper_bound"],
                                       certified=risk["certified"]))
        for parent, child in (("B", "C0"), ("C0", "C0V"), ("B", "C0V")):
            a, b = record["risk"][parent], record["risk"][child]
            mechanism_differences.append(dict(case=case["case"], parent=parent, child=child,
                    reduction_lower=max(0., a["lower_bound"]-b["upper_bound"]),
                    reduction_upper=max(0., a["upper_bound"]-b["lower_bound"]),
                    both_inner_certified=bool(a["certified"] and b["certified"])))
    mechanism_summary = []
    for arm in ("B", "C0", "C0V"):
        selected = [r for r in mechanism_rows if r["evaluation_arm"] == arm]
        mechanism_summary.append(dict(evaluation_arm=arm, count=144,
                                      mean_lower=float(np.mean([r["lower_bound"] for r in selected])),
                                      mean_upper=float(np.mean([r["upper_bound"] for r in selected])),
                                      inner_certified_count=sum(r["certified"] for r in selected)))
    mechanism_difference_summary = []
    for parent, child in (("B", "C0"), ("C0", "C0V"), ("B", "C0V")):
        selected = [r for r in mechanism_differences if r["parent"]==parent and r["child"]==child]
        mechanism_difference_summary.append(dict(parent=parent, child=child,
            mean_reduction_lower=float(np.mean([r["reduction_lower"] for r in selected])),
            mean_reduction_upper=float(np.mean([r["reduction_upper"] for r in selected])),
            both_inner_certified_count=sum(r["both_inner_certified"] for r in selected)))
    algorithm = []
    for arm in ARMS:
        selected = [r for r in main_rows if r["method"]==arm]
        algorithm.append(dict(method=arm, count=144,
            mean_search_seconds=float(np.mean([r["search_seconds"] for r in selected])),
            mean_total_seconds=float(np.mean([r["total_seconds"] for r in selected])),
            mean_evaluated_proposals=float(np.mean([r["evaluated_proposals"] for r in selected])),
            final_inner_certified_count=sum(r["inner_certified"] for r in selected),
            global_route_optimality_certified=False))
    summary = dict(main=means, paired=comparisons,
                   algorithm_stability=dict(distinct_origins=12, seeds=[17,29,43],
                        not_independent_observation_count=True, mean_seed_range={arm: {
                            metric: float(np.mean([r["seed_range"] for r in stability if r["method"]==arm and r["metric"]==metric]))
                            for metric in METRICS} for arm in ARMS}),
                   mechanism=mechanism_summary, mechanism_reductions=mechanism_difference_summary,
                   algorithm=algorithm,
                   uncertainty_coverage={arm: dict(count=sum(r["observed_error_inside"] for r in coverage if r["method"]==arm), denominator=144)
                                         for arm in ("B", "C0", "C0V")},
                   optimized_route_inner_certification={arm: dict(certified=sum(r["inner_certified"] for r in main_rows if r["method"]==arm), denominator=144)
                                                         for arm in ARMS},
                   physical_checks=dict(count=672, passed=True, maximum_violation=max(float(r[k]) for r in physical_details for k in
                       ("negative_inventory", "capacity_overflow", "load_violation", "endpoint_load_violation", "unvisited_handling", "regional_balance_error", "inventory_conservation_error", "global_fleet_closure_error"))),
                   outer_routing_optimality_claim=False, seconds=time.perf_counter()-started)
    csv_write(OUT/"all_snapshot_results.csv", rows)
    csv_write(OUT/"main_144_seed17.csv", main_rows)
    csv_write(OUT/"main_summary.csv", means)
    csv_write(OUT/"paired_main.csv", comparisons)
    csv_write(OUT/"seed_stability_12_origins.csv", stability)
    csv_write(OUT/"fixed_B_route_mechanism.csv", mechanism_rows)
    csv_write(OUT/"fixed_B_route_mechanism_reductions.csv", mechanism_differences)
    csv_write(OUT/"algorithm_summary.csv", algorithm)
    csv_write(OUT/"uncertainty_coverage.csv", coverage)
    write(OUT/"physical_recourse_details.json", physical_details)
    write(OUT/"summary.json", summary)
    lines = ["# 分区域再平衡：144个大规模快照结果", "",
             "本轮共完成672次独立求解。主结果仅使用144个时点各自seed17的四组结果；另12个固定时点使用17/29/43三个种子检查算法稳定性，不能把重复种子当成新的独立样本。每个实例均包含567站。", "",
             "全部路线和输入哈希在读取本轮实际库存前冻结核验。随后固定路线，在库存揭示后求解满足站容、卡车容量和空车起讫约束的装卸量，完整计算567站目标偏差。该信息模式属于静态两阶段追索，不是到站时才观察库存的动态调度。", "",
             "B使用忽略全网耦合的盒松弛，可能包含无法满足车队总量的联合误差组合；C₀在共同盒上加入统计全网误差带与0节点物理库存范围的交集，C₀V再加固定区域合计带。四组在实际库存回放中均使用同一真实物理约束。首批遗漏0节点非负边界的运行已在实际数据读取前废止存档，不进入本报告。", "",
             "## 主结果", "", "目标偏差和里程报告144个时点的平均值；同一加权总成本同时报告均值与观察最大值。观察最大值仅为这144个历史截面内的样本最坏，不是未知需求分布下的保证。总成本沿用每辆目标偏差罚10、每公里成本1；它是模型加权目标，不是经过运营资料标定的货币成本。", "",
             "|组别|平均目标库存偏差（辆）|平均总里程（km）|平均加权总成本|观察最大总成本|", "|---|---:|---:|---:|---:|"]
    for row in means:
        lines.append(f"|{row['method']}|{row['target_deviation_bikes']:.4f}|{row['distance_km']:.4f}|{row['weighted_total_cost']:.4f}|{row['maximum_weighted_total_cost']:.4f}|")
    lines += ["", "## 配对比较", "", "胜/平/负均以挑战组数值更小为胜，平局阈值为1e-6。分别比较目标偏差、里程和总成本，不将它们合成为新指标。", "",
              "|基线→挑战组|指标|挑战组−基线的均值|胜/平/负|", "|---|---|---:|---:|"]
    titles = dict(target_deviation_bikes="目标偏差（辆）", distance_km="里程（km）", weighted_total_cost="加权总成本")
    for row in comparisons:
        lines.append(f"|{row['baseline']}→{row['challenger']}|{titles[row['metric']]}|{row['mean_challenger_minus_baseline']:.4f}|{row['wins']}/{row['ties']}/{row['losses']}|")
    lines += ["", "## 守恒机制与算法边界", "",
              "这里只把每例已经冻结的B路线放入B、C₀、C₀V三个集合重新评价，没有重新选路、没有共享候选池，也没有把诊断结果回送给主搜索。下表为同一路线最坏目标偏差罚损的均值上下界。", "",
              "|评价集合|平均下界|平均上界|内层闭合例数|", "|---|---:|---:|---:|"]
    for row in mechanism_summary:
        lines.append(f"|{row['evaluation_arm']}|{row['mean_lower']:.4f}|{row['mean_upper']:.4f}|{row['inner_certified_count']}/144|")
    lines += ["", "守恒使同一B路线最坏罚损减少的均值范围（已利用集合嵌套保证不为负）：", ""]
    for row in mechanism_difference_summary:
        lines.append(f"- {row['parent']}→{row['child']}：[{row['mean_reduction_lower']:.4f}, {row['mean_reduction_upper']:.4f}]；两端内层同时闭合{row['both_inner_certified_count']}/144例。")
    lines += ["", "上下界针对固定路线的最坏追索，不是整个车辆路径问题的全局最优证明。未闭合的内层不当作精确最坏值；模型内最坏罚损下降，也不能替代实际快照目标偏差改善。", "",
              "算法稳定性只对12个时点逐时点计算三个种子结果范围，详见 `seed_stability_12_origins.csv`。这些是同一实例的重复搜索，不与144个主时点合并推断。", "",
              f"全部672次实际快照追索通过物理核验；最大数值违例为{summary['physical_checks']['maximum_violation']:.3g}。核验包括原归档0节点加实体站等于冻结车队、0节点非负、未访问站零装卸、实体库存非负和不超站容、车载量上下界、空车返回、全网库存守恒以及各区净送入等于实际跨区车载量净流。", "",
              "## 数据与解释限制", "",
              "这是历史快照回放，目标库存偏差不是顾客失单；144个分别求解的快照也不是连续闭环。相邻时点可能存在序列相关，不将144个时点假定为独立同分布样本。十分钟是预测步长，不能解释为全部路线十分钟完成。原数据曾用于诊断，且继承动态图前瞻、成功骑行回溯库存、全期站容估计以及验证残差参与checkpoint选择的限制，不能称为未见样本外证据。", "",
              "误差集合实际覆盖只用于有效性说明，详见 `uncertainty_coverage.csv`。超出统计区间不代表违反物理守恒。区域虚站点是联合误差记账，不是额外供车库存。"]
    lines += ["", "主种子算法统计见 `algorithm_summary.csv`，包含各组实际评价提案数量的平均值；相同秒数预算不保证相同评价次数。其中总耗时包含初始化、搜索和最终内层核验；搜索预算与内层核验预算分开记录，不把内层闭合证书当作外层路径全局最优证书。分析阶段另外冻结底层LP/最坏情景依赖文件哈希，此举只核验后续文件一致性，不冒充主求解开始时已经冻结这些依赖的证据。"]
    (HERE/"SNAPSHOT_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    check_hashes(freeze["hashes"]); check_hashes(freeze["source_sha256"]); check_hashes(protocol["source_sha256"])
    output_paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "analysis_complete.json")
    output_paths.append(HERE/"SNAPSHOT_REPORT.md")
    write(OUT/"analysis_complete.json", dict(complete=True, actual_replays=672, main_origins=144,
          sources_and_frozen_routes_unchanged=True, output_sha256={str(path): sha(path) for path in output_paths}))
    print("SNAPSHOT_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
