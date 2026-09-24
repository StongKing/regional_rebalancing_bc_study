"""Regression checks on existing 567-station inputs; never reads actuals."""
from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
import numpy as np

from robust_evaluator import RobustEvaluator
from oracle import Oracle, deterministic_value

HERE = Path(__file__).resolve().parent
CONFIG = dict(truck_count=4, truck_capacity=30., shortage_penalty=10.,
              surplus_penalty=10., distance_cost_per_km=1.,
              oracle_absolute_tolerance=1e-5)
KEYS = ["demand_nominal", "predicted", "station_capacity", "road_distance_km"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_data(folder, require_fleet=True):
    with np.load(folder / "instance.npz", allow_pickle=False) as saved:
        data = {key: saved[key] for key in KEYS}
        if require_fleet:
            data["fleet_size"] = saved["fleet_size"]
    if len(data["predicted"]) != 567:
        raise AssertionError("This verification only uses existing full-network instances")
    return data


def must_reject(data, spec, method):
    try:
        RobustEvaluator(data, CONFIG, spec, method)
    except ValueError as exc:
        return dict(method=method, rejected=True, reason=str(exc))
    raise AssertionError("An incompletely closed input was silently accepted")


def main():
    folders = sorted((HERE / "inputs").glob("network_*"))
    if len(folders) != 144:
        raise ValueError("Wait for all 144 corrected full-network inputs")
    details = []
    for position in [0, 48, 96]:
        folder = folders[position]
        data = load_data(folder)
        specs = read(folder / "uncertainty_specs.json")
        baseline = read(folder / "metadata.json")["baseline_routes"]
        optional = [route[::2] for route in baseline]
        if optional[0]:
            optional[1].insert(0, optional[0].pop())
        for label, routes in [("baseline", baseline), ("optional_cross_region", optional),
                              ("idle", [[], [], [], []])]:
            answers = {}
            for method in ["B", "C0", "C0V", "D"]:
                evaluator = RobustEvaluator(data, CONFIG, specs[method], method)
                answer = evaluator.evaluate(routes, time_limit=10., verify_physical=True)
                assert evaluator.contains(answer["error"])
                assert answer["physical_primal_verified"] and answer["all_station_losses_included"]
                if method == "B":
                    value, _, _ = Oracle(data, CONFIG, specs["B"])._box(routes)
                    assert abs(value - answer["value"]) < 1e-5
                    assert answer["fleet_closure_relaxed"] and not answer["fleet_closure_verified"]
                elif method == "D":
                    value = deterministic_value(routes, data["demand_nominal"], 30., 10., 10.)
                    assert abs(value - answer["value"]) < 1e-5
                if method != "B":
                    assert answer["fleet_closure_verified"] and answer["node0_physical"]
                    assert -1e-8 <= answer["node0_stock"] <= float(data["fleet_size"]) + 1e-8
                    assert abs(answer["fleet_error"]) <= 1e-5
                cached = evaluator.evaluate(routes, time_limit=.001, verify_physical=True)
                if answer["certified"]:
                    assert cached["cache_hit"]
                answers[method] = answer
                row = dict(case=folder.name, routes=label, method=method,
                           **{key: answer[key] for key in ["value", "upper_bound", "certified", "seconds",
                               "backend", "physical_discrepancy", "node0_stock", "fleet_error",
                               "fleet_closure_verified", "fleet_closure_relaxed"]})
                details.append(row)
                print(json.dumps(row), flush=True)
            assert answers["C0"]["value"] <= answers["B"]["upper_bound"] + 1e-5
            assert answers["C0V"]["value"] <= answers["C0"]["upper_bound"] + 1e-5

    first = folders[0]
    data = load_data(first)
    specs = read(first / "uncertainty_specs.json")
    baseline = read(first / "metadata.json")["baseline_routes"]
    missing = {key: value for key, value in data.items() if key != "fleet_size"}
    rejection_checks = [must_reject(missing, specs[arm], arm) for arm in ["C0", "C0V", "D"]]
    wrong_fleet = dict(data, fleet_size=float(data["predicted"].sum()) - 1.)
    rejection_checks.append(must_reject(wrong_fleet, specs["D"], "D"))
    old = HERE / "superseded_missing_node0_bound" / "inputs" / first.name
    if old.exists():
        old_data = load_data(old, require_fleet=False)
        old_data["fleet_size"] = data["fleet_size"]
        old_specs = read(old / "uncertainty_specs.json")
        for arm in ["C0", "C0V"]:
            rejection_checks.append(must_reject(old_data, old_specs[arm], arm))

    timeout_checks = []
    for folder in [folders[0], folders[96]]:
        case_data = load_data(folder)
        case_specs = read(folder / "uncertainty_specs.json")
        routes = read(folder / "metadata.json")["baseline_routes"]
        evaluator = RobustEvaluator(case_data, CONFIG, case_specs["C0V"], "C0V")
        early = evaluator.evaluate(routes, time_limit=.001, verify_physical=True)
        final = evaluator.evaluate(routes, time_limit=10., verify_physical=True)
        assert early["value"] <= final["value"] + 1e-5 <= early["upper_bound"] + 2e-5
        assert final["upper_bound"] <= early["upper_bound"] + 1e-5
        assert final["certified"] and early["fleet_closure_verified"] and final["fleet_closure_verified"]
        timeout_checks.append(dict(case=folder.name,
                                   early={k: early[k] for k in ["value", "upper_bound", "certified", "node0_stock"]},
                                   final={k: final[k] for k in ["value", "upper_bound", "certified", "node0_stock"]},
                                   interval_contains_certified_value=True))
    evaluator = RobustEvaluator(data, dict(CONFIG, oracle_cache_size=2), specs["B"], "B")
    before = evaluator.evaluate(baseline)
    evaluator.evaluate([route[::2] for route in baseline])
    evaluator.evaluate([[], [], [], []])
    after = evaluator.evaluate(baseline, verify_physical=True)
    assert len(evaluator.cache) == 2 and evaluator.stats["cache_evictions"] == 2
    assert before["value"] == after["value"]

    report = dict(scope="existing complete 567-station instances only; corrected node0 physical closure",
                  actual_outcomes_read=False, checks=len(details),
                  certified=sum(row["certified"] for row in details),
                  max_physical_discrepancy=max(row["physical_discrepancy"] for row in details),
                  max_seconds=max(row["seconds"] for row in details),
                  all_enforced_fleet_checks_passed=True, cache_bound_verified=True,
                  rejection_checks=rejection_checks, timeout_refinement_checks=timeout_checks,
                  details=details)
    (HERE / "evaluator_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}), flush=True)


if __name__ == "__main__":
    main()
