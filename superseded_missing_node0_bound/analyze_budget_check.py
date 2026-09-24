"""Predeclared 12-origin 12s/60s comparison; never replaces main routes.

Both campaigns (672 + 48 routes) must be complete and hash verified before
actual inventory is opened. Results concern fixed-route snapshot recourse.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
import sys
sys.dont_write_bytecode = True
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

import analyze_snapshots as snapshots
from run_independent import ARMS, HERE, clean, read, sha, write

BUDGET = HERE/"results/budget_check"
OUT = HERE/"results/budget_check_analysis"
REPORT = HERE/"BUDGET_CHECK_REPORT.md"
METRICS = snapshots.METRICS


def preflight():
    index, main_freeze, main_runs, source_hash, prediction_audit = snapshots.preflight()
    path = BUDGET/"optimization_freeze.json"
    if not path.exists(): raise RuntimeError("Budget campaign incomplete; actual remains unopened")
    freeze = read(path)
    if (not freeze.get("complete") or freeze.get("expected") != 48 or freeze.get("completed") != 48
            or freeze.get("failures") or freeze.get("actual_arrays_opened") or not freeze.get("main_routes_unchanged")):
        raise RuntimeError("All 48 budget runs must be complete before actual access")
    snapshots.check_hashes(freeze["hashes"]); snapshots.check_hashes(freeze["source_sha256"])
    selected = [c for c in index["cases"] if (c["sample_id"]-176)%12 == 0]
    if len(selected) != 12: raise RuntimeError("Wrong frozen two-hour origin block")
    main_protocol = read(snapshots.MAIN/"protocol.json")
    budget_protocol = read(BUDGET/"protocol.json")
    if budget_protocol["selected_cases"] != [c["case"] for c in selected]: raise RuntimeError("Budget cases drifted")
    if main_protocol["config"] != budget_protocol["config"]: raise RuntimeError("Physical model differs by budget")
    main_settings, budget_settings = main_protocol["settings"].copy(), budget_protocol["settings"].copy()
    if main_settings.pop("search_seconds") != 12. or budget_settings.pop("search_seconds") != 60.:
        raise RuntimeError("Expected predeclared 12s and 60s budgets")
    if main_settings != budget_settings: raise RuntimeError("A solver setting other than time budget changed")
    budget_runs = {}
    for filename in freeze["hashes"]:
        row = read(filename)
        key = (row["case"], row["method"], row["seed"])
        if key in budget_runs or row.get("actual_arrays_opened") or not row.get("independent_search") or row.get("policy_pool"):
            raise RuntimeError("Budget run violates independent outcome-blind design")
        budget_runs[key] = row
    expected = {(c["case"], arm, 17) for c in selected for arm in ARMS}
    if set(budget_runs) != expected: raise RuntimeError("Budget arm/origin combinations incomplete")
    for case in selected:
        for arm in ARMS:
            key = (case["case"], arm, 17)
            if budget_runs[key]["input_hashes"] != case["output_sha256"]:
                raise RuntimeError("Budget inputs differ from main inputs")
            if budget_runs[key]["initial_routes_sha256"] != main_runs[key]["initial_routes_sha256"]:
                raise RuntimeError("12s/60s runs have different initial routes")
    return selected, main_freeze, freeze, main_runs, budget_runs, source_hash, prediction_audit


def main():
    selected, main_freeze, budget_freeze, main_runs, budget_runs, source_hash, prediction_audit = preflight()
    protocol = dict(selected_cases=[c["case"] for c in selected], distinct_origins=12, seed=17,
        budgets_seconds=[12,60], metrics=list(METRICS),
        total_cost_descriptive_statistics=["mean", "observed_maximum"],
        estimate="paired 60s-minus-12s within each arm; C0V-minus-B at each budget on the same 12 origins",
        main_144_mean_is_not_replaced=True, best_route_selection_prohibited=True,
        identical_inputs_initial_routes_and_non_time_solver_settings_verified=True,
        complete_optimization_run_count=720,
        main_freeze_sha256=sha(snapshots.MAIN/"optimization_freeze.json"),
        budget_freeze_sha256=sha(BUDGET/"optimization_freeze.json"),
        source_sha256={str(path): sha(path) for path in (Path(__file__), Path(snapshots.__file__),
            Path(snapshots.oracle.__file__), Path(snapshots.routing.__file__))},
        prediction_identity_audit=prediction_audit, outcome_archive_sha256=source_hash,
        scope="retrospective full-network fixed-route inventory-target recourse; not closed-loop service")
    if (OUT/"analysis_freeze.json").exists() and read(OUT/"analysis_freeze.json") != clean(protocol):
        raise RuntimeError("Budget analysis protocol changed after freeze")
    write(OUT/"analysis_freeze.json", protocol)
    snapshots.check_hashes(main_freeze["hashes"]); snapshots.check_hashes(budget_freeze["hashes"])
    snapshots.check_hashes(main_freeze["source_sha256"]); snapshots.check_hashes(budget_freeze["source_sha256"])
    if sha(snapshots.EXPORT) != source_hash: raise RuntimeError("Prediction/outcome archive changed")
    if not (OUT/"actual_access.json").exists():
        write(OUT/"actual_access.json", dict(first_access_utc=datetime.now(timezone.utc).isoformat(),
            complete_frozen_routes=720, all_run_hashes_verified=True, member="actual"))
    with np.load(snapshots.EXPORT, allow_pickle=False) as archive:
        ids = archive["sample_id"].copy().reshape(-1)
        actual = archive["actual"].copy()
    rows, details = [], []
    for case in selected:
        with np.load(Path(case["directory"])/"instance.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        position = np.flatnonzero(ids == case["sample_id"])
        if len(position) != 1: raise RuntimeError("Outcome sample id not unique")
        inventory = np.asarray(actual[int(position[0]), 1:, 0], float)
        for budget, runs in ((12, main_runs), (60, budget_runs)):
            for arm in ARMS:
                run = runs[(case["case"], arm, 17)]
                replay = snapshots.physical_check(run["routes"], inventory, data, case["distance_cap_km"])
                rows.append(dict(case=case["case"], sample_id=case["sample_id"], method=arm, seed=17,
                    budget_seconds=budget, **{metric: replay[metric] for metric in METRICS},
                    evaluated_proposals=run["evaluated_proposals"], search_seconds=run["search_seconds"],
                    total_seconds=run["total_seconds"], inner_certified=run["risk"]["certified"],
                    inner_lower=run["risk"]["lower_bound"], inner_upper=run["risk"]["upper_bound"]))
                details.append(dict(case=case["case"], method=arm, budget_seconds=budget, **replay))
    means = []
    for budget in (12,60):
        for arm in ARMS:
            subset = [r for r in rows if r["budget_seconds"]==budget and r["method"]==arm]
            means.append(dict(budget_seconds=budget, method=arm, distinct_origins=12,
                **{metric: float(np.mean([r[metric] for r in subset])) for metric in METRICS},
                maximum_weighted_total_cost=max(r["weighted_total_cost"] for r in subset),
                mean_evaluated_proposals=float(np.mean([r["evaluated_proposals"] for r in subset])),
                mean_search_seconds=float(np.mean([r["search_seconds"] for r in subset])),
                mean_total_seconds=float(np.mean([r["total_seconds"] for r in subset])),
                inner_certified_count=sum(r["inner_certified"] for r in subset)))
    lookup = {(r["case"], r["method"], r["budget_seconds"]): r for r in rows}
    budget_changes, effects, effect_changes = [], [], []
    def describe(values):
        values = np.asarray(values)
        return dict(mean_delta=float(values.mean()), negative=int((values < -snapshots.TOL).sum()),
                    tied=int((np.abs(values)<=snapshots.TOL).sum()), positive=int((values > snapshots.TOL).sum()))
    for metric in METRICS:
        for arm in ARMS:
            changes = [lookup[(c["case"], arm, 60)][metric]-lookup[(c["case"], arm, 12)][metric] for c in selected]
            budget_changes.append(dict(method=arm, metric=metric, contrast="60s-minus-12s", **describe(changes)))
        for budget in (12,60):
            differences = [lookup[(c["case"], "C0V", budget)][metric]-lookup[(c["case"], "B", budget)][metric] for c in selected]
            effects.append(dict(budget_seconds=budget, metric=metric, contrast="C0V-minus-B", **describe(differences)))
        differences = [(lookup[(c["case"], "C0V", 60)][metric]-lookup[(c["case"], "B", 60)][metric])-
                       (lookup[(c["case"], "C0V", 12)][metric]-lookup[(c["case"], "B", 12)][metric]) for c in selected]
        effect_changes.append(dict(metric=metric, contrast="C0V-minus-B at60s minus same contrast at12s", **describe(differences)))
    summary = dict(distinct_origins=12, seed=17, snapshot_replays=96, means=means,
        paired_budget_changes=budget_changes, C0V_minus_B_effects=effects, paired_effect_changes=effect_changes,
        main_144_results_replaced=False, main_routes_unchanged=True,
        physical_checks=dict(passed=True, replay_count=96, maximum_violation=max(float(r[k]) for r in details for k in
            ("negative_inventory", "capacity_overflow", "load_violation", "endpoint_load_violation", "unvisited_handling", "regional_balance_error", "inventory_conservation_error"))),
        no_outer_optimality_claim=True, temporal_IID_not_assumed=True)
    snapshots.csv_write(OUT/"budget_snapshot_results.csv", rows)
    snapshots.csv_write(OUT/"means_same_12_origins.csv", means)
    snapshots.csv_write(OUT/"paired_budget_changes.csv", budget_changes)
    snapshots.csv_write(OUT/"C0V_minus_B_effects.csv", effects)
    snapshots.csv_write(OUT/"paired_effect_changes.csv", effect_changes)
    write(OUT/"physical_recourse_details.json", details)
    write(OUT/"summary.json", summary)
    labels = dict(target_deviation_bikes="目标库存偏差（辆）", distance_km="里程（km）", weighted_total_cost="加权总成本")
    lines = ["# 计算预算核验：固定12个大规模时点", "",
        "在读取本轮实际库存之前，因发现不同鲁棒集合的内层评价速度差异而预先登记本项核验。只对原先固定的12个整两小时时点、seed17，将四组搜索预算统一从12秒增至60秒；实例、误差集合、初始化规则和其他求解设置完全相同。新增48条独立路线，保留全部原主结果，不从两种预算中事后择优。", "",
        "两批720条优化路线全部完成并核对哈希后，才读取实际库存。下文所有均值仅对应同一组12个时点，不与144个主时点均值混用；96次快照回放也不等于96个独立样本。", "",
        "## 相同12时点的结果", "",
        "|预算|组别|平均目标偏差（辆）|平均里程（km）|平均加权总成本|观察最大总成本|平均评价提案数|最终内层闭合|", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in means:
        lines.append(f"|{r['budget_seconds']}秒|{r['method']}|{r['target_deviation_bikes']:.4f}|{r['distance_km']:.4f}|{r['weighted_total_cost']:.4f}|{r['maximum_weighted_total_cost']:.4f}|{r['mean_evaluated_proposals']:.1f}|{r['inner_certified_count']}/12|")
    lines += ["", "观察最大总成本仅为相同12个历史截面的样本最坏，不代表未知需求分布保证。相同墙钟时间不保证相同评价次数；这里检验增加时间后模型比较是否改变，而不是以60秒结果替换12秒主结果。总成本沿用每辆目标偏差罚10、每公里成本1，不是标定后的货币成本。", "",
        "## C₀V相对B的效果是否随预算改变", "",
        "数值为C₀V减B，负值有利于C₀V；胜/平/负以该差值小于/等于/大于零统计，阈值1e-6。", "",
        "|预算|指标|C₀V−B均值|C₀V胜/平/负|", "|---:|---|---:|---:|"]
    for r in effects:
        lines.append(f"|{r['budget_seconds']}秒|{labels[r['metric']]}|{r['mean_delta']:.4f}|{r['negative']}/{r['tied']}/{r['positive']}|")
    lines += ["", "将60秒下的C₀V−B减去12秒下的同一差值，可直接判断有限搜索预算对结论的影响；负值表示增加预算后比较向C₀V方向移动：", ""]
    for r in effect_changes:
        lines.append(f"- {labels[r['metric']]}：配对差值变化均值 {r['mean_delta']:.4f}。")
    lines += ["", "各组自身的60秒−12秒配对结果见 `paired_budget_changes.csv`。即使60秒有所改善，外层ALNS仍是启发式搜索，不能据此声称模型全局最优或计算预算影响已完全排除。", "",
        "全部96次完整567站追索均通过站容、车载量、空车起讫、未访问站零装卸、库存守恒及区域跨界载量核验。库存揭示后只改变装卸量，不改变已冻结路线。", "",
        "这是回顾性快照目标库存偏差，不能称为真实失单或连续闭环；固定12个时点仍有序列相关，不假定独立同分布。原数据的前瞻预测图、回溯库存重构、全期站容和验证资料复用限制仍然存在。"]
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf-8")
    snapshots.check_hashes(main_freeze["hashes"]); snapshots.check_hashes(budget_freeze["hashes"])
    snapshots.check_hashes(main_freeze["source_sha256"]); snapshots.check_hashes(budget_freeze["source_sha256"])
    snapshots.check_hashes(protocol["source_sha256"])
    paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "analysis_complete.json") + [REPORT]
    write(OUT/"analysis_complete.json", dict(complete=True, optimized_routes_frozen=720, actual_replays=96,
        distinct_origins=12, main_routes_unchanged=True, output_sha256={str(path): sha(path) for path in paths}))
    print("BUDGET_CHECK_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
