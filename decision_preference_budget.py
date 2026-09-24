"""Separate fixed-decision preference audit of the existing 60-second runs."""
from pathlib import Path
import csv
import json
import time

import numpy as np
from decision_preference import (HERE, CONFIG, TOLERANCE, RobustEvaluator, canonical,
                                 read, sha, write, check_hashes, objective_interval, preference)


def main():
    started = time.perf_counter()
    source = HERE / "results/budget_check"
    out = HERE / "results/decision_preference_budget"
    frozen_path = source / "optimization_freeze.json"
    freeze = read(frozen_path)
    if not freeze.get("complete") or freeze.get("completed") != 48:
        raise RuntimeError("All 48 longer-budget routes must be frozen")
    check_hashes(freeze["source_sha256"])
    index_path = HERE / "inputs/index.json"
    selected = [c for c in read(index_path)["cases"] if (c["sample_id"] - 176) % 12 == 0]
    if len(selected) != 12:
        raise AssertionError("Expected the same 12 preregistered two-hour origins")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("A previous separate budget preference audit exists")
    sources = {str(frozen_path): sha(frozen_path), str(index_path): sha(index_path),
               str(Path(__file__)): sha(__file__),
               str(HERE / "decision_preference.py"): sha(HERE / "decision_preference.py")}
    rows, details = [], []
    for case in selected:
        name = case["case"]
        bp = source / "runs" / (name + "__B__s17.json")
        vp = source / "runs" / (name + "__C0V__s17.json")
        for path in [bp, vp]:
            expected = freeze["hashes"].get(str(path))
            if expected is None or sha(path) != expected:
                raise AssertionError("Budget decision record missing or changed")
            sources[str(path)] = expected
        b, v = read(bp), read(vp)
        for record, method in [(b, "B"), (v, "C0V")]:
            assert (record["case"], record["method"], record["seed"]) == (name, method, 17)
            assert record["independent_search"] and not record["actual_arrays_opened"]
            assert record["input_hashes"] == case["output_sha256"]
        folder = Path(case["directory"])
        hashes = {str(folder / file): digest for file, digest in case["output_sha256"].items()}
        check_hashes(hashes)
        sources.update(hashes)
        with np.load(folder / "instance.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in ["demand_nominal", "predicted", "station_capacity",
                                                "road_distance_km", "fleet_size"]}
        specs = read(folder / "uncertainty_specs.json")
        box = RobustEvaluator(data, CONFIG, specs["B"], "B")
        conserved = RobustEvaluator(data, CONFIG, specs["C0V"], "C0V")
        b_of_v = box.evaluate(v["routes"])
        v_of_b = conserved.evaluate(b["routes"], time_limit=10., verify_physical=True)
        assert b_of_v["certified"] and b_of_v["backend"] == "exact_box_path_dp"
        distance_b = float(box.route_lengths(b["routes"]).sum())
        distance_v = float(box.route_lengths(v["routes"]).sum())
        for record, distance in [(b, distance_b), (v, distance_v)]:
            assert abs(record["risk"]["route_distance_km"] - distance) <= TOLERANCE
        cost_b, cost_v = CONFIG["distance_cost_per_km"] * distance_b, CONFIG["distance_cost_per_km"] * distance_v
        bb = objective_interval(b["risk"], cost_b)
        bv = objective_interval(b_of_v, cost_v)
        vb = objective_interval(v_of_b, cost_b)
        vv = objective_interval(v["risk"], cost_v)
        bpref, bdiff = preference(bb, bv)
        vpref, vdiff = preference(vb, vv)
        row = dict(case=name, sample_id=int(case["sample_id"]), seed=17, search_seconds=60,
                   identical_routes=canonical(b["routes"]) == canonical(v["routes"]),
                   B_prefers=bpref, C0V_prefers=vpref,
                   strict_preference_reversal=bpref == "B_route" and vpref == "C0V_route",
                   C0V_search_shortfall_evidence=vpref == "B_route",
                   B_search_shortfall_evidence=bpref == "C0V_route",
                   B_xB_lower=bb[0], B_xB_upper=bb[1], B_xCV_lower=bv[0], B_xCV_upper=bv[1],
                   C0V_xB_lower=vb[0], C0V_xB_upper=vb[1], C0V_xCV_lower=vv[0], C0V_xCV_upper=vv[1],
                   B_difference_lower=bdiff[0], B_difference_upper=bdiff[1],
                   C0V_difference_lower=vdiff[0], C0V_difference_upper=vdiff[1],
                   additional_C0V_inner_certified=v_of_b["certified"])
        rows.append(row)
        details.append(dict(**row, new_box_evaluation=b_of_v, new_C0V_of_B_evaluation=v_of_b,
                            frozen_route_record_paths=[str(bp), str(vp)]))
        print(json.dumps(row), flush=True)
    labels = ["B_route", "C0V_route", "tie_within_tolerance", "unresolved_bounds"]
    summary = dict(origins=12, seed=17, source_search_seconds=60, strict_difference_tolerance=TOLERANCE,
                   strict_preference_reversal_count=sum(r["strict_preference_reversal"] for r in rows),
                   C0V_search_shortfall_count=sum(r["C0V_search_shortfall_evidence"] for r in rows),
                   B_search_shortfall_count=sum(r["B_search_shortfall_evidence"] for r in rows),
                   identical_routes_count=sum(r["identical_routes"] for r in rows),
                   preference_counts={arm: {label: sum(r[arm + "_prefers"] == label for r in rows)
                                           for label in labels} for arm in ["B", "C0V"]},
                   new_C0V_inner_certified=sum(r["additional_C0V_inner_certified"] for r in rows),
                   actual_outcomes_read=False, new_outer_optimization_runs=0, new_routes_selected=0,
                   search_feedback=False, new_exact_box_evaluations=12, new_fixed_route_C0V_evaluations=12,
                   fixed_route_C0V_time_limit_seconds=10.,
                   strict_reversal_definition="UB_B(xB)+1e-6 < LB_B(xCV) and UB_C0V(xCV)+1e-6 < LB_C0V(xB)",
                   separate_from_main_144=True, seconds=time.perf_counter() - started)
    check_hashes(sources)
    check_hashes(freeze["source_sha256"])
    out.mkdir(parents=True, exist_ok=True)
    write(out / "protocol.json", dict(source_sha256=sources, configuration=CONFIG,
                                      tolerance=TOLERANCE, actual_outcomes_read=False,
                                      selection="same previously registered 12 longer-budget origins; frozen seed17 B/C0V routes only"))
    write(out / "summary.json", summary)
    write(out / "details.json", details)
    with (out / "per_origin.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 60秒预算：独立的冻结决策偏好核验", "",
             "仅使用原先登记的12个两小时间隔时点及seed17。两组路线来自已完成的60秒预算实验；没有再次搜索或替换路线，也不与144个主时点合并。", "",
             "记x_B、x_C为两组冻结路线，J_B、J_C均包含全网最坏目标罚损和相同里程成本。严格反转要求UB[J_B(x_B)]+1e-6<LB[J_B(x_C)]，且UB[J_C(x_C)]+1e-6<LB[J_C(x_B)]。只对冻结B路线增加一次10秒上限的C₀V风险认证、对冻结C₀V路线补一次精确盒DP；未读取实际库存。", "",
             "|核验项|例数（共12）|", "|---|---:|",
             f"|严格偏好反转|{summary['strict_preference_reversal_count']}|",
             f"|C₀V自己的目标下，B路线严格更便宜|{summary['C0V_search_shortfall_count']}|",
             f"|B自己的目标下，C₀V路线严格更便宜|{summary['B_search_shortfall_count']}|",
             f"|新增C₀V固定路线内层认证闭合|{summary['new_C0V_inner_certified']}|", "",
             "严格反转证明两条路线排序受到守恒影响；相反，另一组路线在本组目标下更便宜，是本组启发式搜索不足的证据。未出现反转不等于证明守恒仅产生相同常数变化。诊断不提供顾客服务改善或外层全局最优证明，不向任何搜索回送结果。", "",
             "|评价集合|严格偏好B路线|严格偏好C₀V路线|阈值内持平|区间尚不能判定|", "|---|---:|---:|---:|---:|"]
    for arm in ["B", "C0V"]:
        counts = summary["preference_counts"][arm]
        lines.append(f"|{arm}|{counts['B_route']}|{counts['C0V_route']}|{counts['tie_within_tolerance']}|{counts['unresolved_bounds']}|")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(out / "complete.json", dict(complete=True, frozen_sources_unchanged=True,
                                      actual_outcomes_read=False,
                                      output_sha256={str(p): sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
