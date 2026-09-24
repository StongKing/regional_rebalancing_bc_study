"""Post hoc preference audit of two independently frozen full-network routes.

Only seed-17 B and C0V routes are examined.  Three of the four risk values
already exist; the fourth is an exact box-path DP evaluation.  No optimization,
route selection, outcome access, or feedback to any search is performed.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

from robust_evaluator import RobustEvaluator
from run_independent import CONFIG, canonical

HERE = Path(__file__).resolve().parent
MAIN = HERE / "results/main"
MECHANISM = HERE / "results/snapshot_analysis/mechanism"
OUT = HERE / "results/decision_preference"
TOLERANCE = 1e-6  # Same fixed strict-difference tolerance as snapshot comparisons.


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def check_hashes(hashes):
    for name, digest in hashes.items():
        if sha(name) != digest:
            raise RuntimeError("Frozen source changed: " + str(name))


def objective_interval(risk, distance_cost):
    lower = float(risk.get("lower_bound", risk["value"])) + distance_cost
    upper = float(risk["upper_bound"]) + distance_cost
    if not np.isfinite([lower, upper]).all() or upper < lower:
        raise ValueError("Invalid certified risk interval")
    return lower, upper


def preference(first, second):
    """Compare xB (first) and xCV (second), using bounds without midpoints."""
    difference = (second[0] - first[1], second[1] - first[0])
    if difference[0] > TOLERANCE:
        result = "B_route"
    elif difference[1] < -TOLERANCE:
        result = "C0V_route"
    elif difference[0] >= -TOLERANCE and difference[1] <= TOLERANCE:
        result = "tie_within_tolerance"
    else:
        result = "unresolved_bounds"
    return result, difference


def main():
    started = time.perf_counter()
    freeze_path = MAIN / "optimization_freeze.json"
    freeze = read(freeze_path)
    if not freeze.get("complete") or freeze.get("completed") != 672:
        raise RuntimeError("All independent optimization decisions must be frozen first")
    check_hashes(freeze["source_sha256"])
    index_path = HERE / "inputs/index.json"
    cases = read(index_path)["cases"]
    if len(cases) != 144:
        raise ValueError("Expected all 144 main origins")
    missing = [case["case"] for case in cases if not (MECHANISM / (case["case"] + ".json")).exists()]
    if missing:
        raise RuntimeError("Wait for all existing B-route mechanism records; missing " + str(len(missing)))
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("A prior preference audit exists; do not overwrite it")

    sources = {str(freeze_path): sha(freeze_path), str(index_path): sha(index_path),
               str(Path(__file__)): sha(__file__)}
    rows = []
    detailed = []
    for case in cases:
        name = case["case"]
        b_path = MAIN / "runs" / (name + "__B__s17.json")
        cv_path = MAIN / "runs" / (name + "__C0V__s17.json")
        mechanism_path = MECHANISM / (name + ".json")
        for path in [b_path, cv_path]:
            expected = freeze["hashes"].get(str(path))
            if expected is None or sha(path) != expected:
                raise RuntimeError("Final route record is absent from freeze or changed: " + str(path))
            sources[str(path)] = expected
        sources[str(mechanism_path)] = sha(mechanism_path)
        b, cv, mechanism = read(b_path), read(cv_path), read(mechanism_path)
        for record, method in [(b, "B"), (cv, "C0V")]:
            if (record["case"], record["method"], record["seed"]) != (name, method, 17):
                raise AssertionError("Unexpected source decision")
            if not record.get("independent_search") or record.get("actual_arrays_opened"):
                raise AssertionError("Unexpected source decision information pattern")
        if mechanism.get("actual_outcomes_read") or mechanism.get("route_selection_performed"):
            raise AssertionError("Unexpected mechanism information pattern")
        if canonical(mechanism["route_canonical"]) != canonical(b["routes"]):
            raise AssertionError("Mechanism does not evaluate the fixed B route")

        folder = Path(case["directory"])
        input_hashes = {str(folder / filename): digest for filename, digest in case["output_sha256"].items()}
        check_hashes(input_hashes)
        sources.update(input_hashes)
        if b["input_hashes"] != case["output_sha256"] or cv["input_hashes"] != case["output_sha256"]:
            raise AssertionError("Source routes use different inputs")
        with np.load(folder / "instance.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in ["demand_nominal", "predicted", "station_capacity",
                                                "road_distance_km", "fleet_size"]}
        specs = read(folder / "uncertainty_specs.json")
        evaluator = RobustEvaluator(data, CONFIG, specs["B"], "B")
        b_of_cv = evaluator.evaluate(cv["routes"])
        if not b_of_cv["certified"] or b_of_cv["backend"] != "exact_box_path_dp":
            raise AssertionError("The sole new evaluation must be the exact box DP")
        distance_b = float(evaluator.route_lengths(b["routes"]).sum())
        distance_cv = float(evaluator.route_lengths(cv["routes"]).sum())
        for source, distance in [(b, distance_b), (cv, distance_cv)]:
            if abs(source["risk"]["route_distance_km"] - distance) > TOLERANCE:
                raise AssertionError("Frozen route distance differs from recomputation")
        cost_b = CONFIG["distance_cost_per_km"] * distance_b
        cost_cv = CONFIG["distance_cost_per_km"] * distance_cv
        bb = objective_interval(mechanism["risk"]["B"], cost_b)
        vb = objective_interval(mechanism["risk"]["C0V"], cost_b)
        bv = objective_interval(b_of_cv, cost_cv)
        vv = objective_interval(cv["risk"], cost_cv)
        original_bb = objective_interval(b["risk"], cost_b)
        if bb[0] > original_bb[1] + TOLERANCE or original_bb[0] > bb[1] + TOLERANCE:
            raise AssertionError("Repeated fixed B-route box intervals are inconsistent")
        b_preference, b_difference = preference(bb, bv)
        cv_preference, cv_difference = preference(vb, vv)
        reversal = b_preference == "B_route" and cv_preference == "C0V_route"
        same = canonical(b["routes"]) == canonical(cv["routes"])
        if same and (reversal or b_preference in ["B_route", "C0V_route"] or cv_preference in ["B_route", "C0V_route"]):
            raise AssertionError("Identical routes cannot exhibit strict preference")
        row = dict(case=name, sample_id=int(case["sample_id"]), seed=17,
                   identical_routes=same, B_prefers=b_preference, C0V_prefers=cv_preference,
                   strict_preference_reversal=reversal,
                   C0V_search_shortfall_evidence=cv_preference == "B_route",
                   B_search_shortfall_evidence=b_preference == "C0V_route",
                   B_xB_lower=bb[0], B_xB_upper=bb[1], B_xCV_lower=bv[0], B_xCV_upper=bv[1],
                   C0V_xB_lower=vb[0], C0V_xB_upper=vb[1], C0V_xCV_lower=vv[0], C0V_xCV_upper=vv[1],
                   B_difference_CVroute_minus_Broute_lower=b_difference[0],
                   B_difference_CVroute_minus_Broute_upper=b_difference[1],
                   C0V_difference_CVroute_minus_Broute_lower=cv_difference[0],
                   C0V_difference_CVroute_minus_Broute_upper=cv_difference[1],
                   B_route_distance_km=distance_b, C0V_route_distance_km=distance_cv)
        rows.append(row)
        detailed.append(dict(**row, new_box_evaluation=b_of_cv,
                             frozen_route_record_paths=[str(b_path), str(cv_path)],
                             existing_B_route_mechanism=str(mechanism_path)))

    preference_labels = ["B_route", "C0V_route", "tie_within_tolerance", "unresolved_bounds"]
    summary = dict(origins=144, seed=17, strict_difference_tolerance=TOLERANCE,
                   strict_preference_reversal_count=sum(r["strict_preference_reversal"] for r in rows),
                   C0V_search_shortfall_count=sum(r["C0V_search_shortfall_evidence"] for r in rows),
                   B_search_shortfall_count=sum(r["B_search_shortfall_evidence"] for r in rows),
                   identical_routes_count=sum(r["identical_routes"] for r in rows),
                   preference_counts={arm: {label: sum(r[arm + "_prefers"] == label for r in rows)
                                           for label in preference_labels} for arm in ["B", "C0V"]},
                   actual_outcomes_read=False, new_optimization_runs=0, new_routes_selected=0,
                   search_feedback=False, new_exact_box_evaluations=144,
                   cost_definition="complete-network worst-case target loss plus the same distance cost",
                   strict_reversal_definition="UB_B(xB)+1e-6 < LB_B(xCV) and UB_C0V(xCV)+1e-6 < LB_C0V(xB)",
                   C0V_shortfall_definition="UB_C0V(xB)+1e-6 < LB_C0V(xCV)",
                   B_shortfall_definition="UB_B(xCV)+1e-6 < LB_B(xB)",
                   limitations="two frozen heuristic routes per origin; no global-route optimality, causal service benefit, or constant-shift conclusion from an absent reversal",
                   seconds=time.perf_counter() - started)
    check_hashes(sources)
    check_hashes(freeze["source_sha256"])
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "protocol.json", dict(source_sha256=sources, configuration=CONFIG,
                                      tolerance=TOLERANCE, actual_outcomes_read=False,
                                      timing="post hoc mechanism diagnosis; no route selection or search feedback"))
    write(OUT / "details.json", detailed)
    write(OUT / "summary.json", summary)
    with (OUT / "per_origin.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 守恒是否改变路线偏好：冻结决策机理核验", "",
             "仅检查144个主时点、种子17。每例使用B与C₀V各自独立搜索后已经冻结的两条路线。仅读已有最坏风险记录，并补算一次精确盒DP；没有新增优化、选路、实际库存读取或向搜索回送结果。", "",
             "记x_B、x_C为两组冻结路线，J_B(x)、J_C(x)为相应不确定集合下的完整567站最坏目标罚损加同一里程成本。所有比较都包含各自路线里程，严格差阈值固定为1e-6；不以区间中点替代未闭合风险。", "",
             "严格偏好反转要求同时证明：UB[J_B(x_B)]+1e-6<LB[J_B(x_C)]，且UB[J_C(x_C)]+1e-6<LB[J_C(x_B)]。若成立，说明守恒改变了这两条路线的排序，不能解释为对每条路线减去相同常数。", "",
             "|核验项|例数（共144）|", "|---|---:|",
             f"|严格偏好反转|{summary['strict_preference_reversal_count']}|",
             f"|C₀V自己的目标下，冻结B路线严格更便宜|{summary['C0V_search_shortfall_count']}|",
             f"|B自己的目标下，冻结C₀V路线严格更便宜|{summary['B_search_shortfall_count']}|",
             f"|两组冻结路线相同（忽略同质卡车编号）|{summary['identical_routes_count']}|", "",
             "后两项说明相应启发式搜索至少漏掉了一个已经独立找到的更好可行路线，不代表另一组具有真实运营优势。本诊断不会据此替换任何主结果中的路线。", "",
             "|评价集合|严格偏好B路线|严格偏好C₀V路线|阈值内持平|区间尚不能判定|", "|---|---:|---:|---:|---:|"]
    for arm in ["B", "C0V"]:
        counts = summary["preference_counts"][arm]
        lines.append(f"|{arm}|{counts['B_route']}|{counts['C0V_route']}|{counts['tie_within_tolerance']}|{counts['unresolved_bounds']}|")
    lines += ["", "没有观察到严格反转不等于证明守恒只产生常数平移，也不等于证明守恒没有决策价值；可能受这两条已找到路线以及尚未闭合风险区间限制。该核验不是顾客服务或实际库存效果评价。逐时点四个成本区间和判定见per_origin.csv。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = {str(path): sha(path) for path in OUT.iterdir() if path.is_file()}
    write(OUT / "complete.json", dict(complete=True, frozen_sources_unchanged=True,
                                      actual_outcomes_read=False, output_sha256=outputs))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
