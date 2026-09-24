"""Prepare the fixed 144 large instances without reading their actual outcomes.

All writes are confined to this study. Existing forecasts, targets, capacities,
and residual histories are reused verbatim. The four geographic groups are
fixed once from coordinates. The robust arms share one box and one calibration
radius; only signed aggregate constraints differ.
"""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "conservation_large_scale_study/results/instances"
OUT = HERE / "inputs"
ROBUST_ARMS = ("B", "C0", "C0V")
TOL = 1e-7


def clean(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def geographic_regions(coords):
    """Recursive longest projected axis median; deterministic equal-size halves."""
    coords = np.asarray(coords, float)
    if coords.shape != (567, 2) or not np.isfinite(coords).all():
        raise ValueError("Expected all 567 finite station coordinates")
    metric = coords.copy()
    metric[:, 1] *= np.cos(np.deg2rad(coords[:, 0].mean()))
    trace = []

    def split(indices, depth, name):
        if depth == 0:
            return [(name, indices)]
        span = np.ptp(metric[indices], axis=0)
        axis = int(np.argmax(span))
        order = indices[np.lexsort((indices, metric[indices, axis]))]
        middle = len(order) // 2
        trace.append(dict(node=name, axis=axis, count=len(indices),
                          lower_max=float(metric[order[middle-1], axis]),
                          upper_min=float(metric[order[middle], axis])))
        return split(order[:middle], depth-1, name+"0") + split(order[middle:], depth-1, name+"1")

    parts = split(np.arange(len(coords)), 2, "geo_")
    labels = np.empty(len(coords), dtype=int)
    groups = []
    for k, (name, indices) in enumerate(parts):
        indices = np.sort(indices)
        labels[indices] = k
        groups.append(dict(name=name, indices=indices.tolist()))
    return groups, labels, dict(
        rule="two recursive median splits on the longest geographic axis; longitude scaled by cos(mean latitude); station index breaks ties",
        data_used="fixed station coordinates only", counts=[len(g["indices"]) for g in groups],
        split_trace=trace, no_forecasts_or_outcomes_used=True)


def center_scale(values):
    lower, upper = values.min(axis=0), values.max(axis=0)
    return (lower+upper)/2, np.maximum(1., (upper-lower)/2)


def inside(spec, errors):
    errors = np.atleast_2d(errors)
    yes = ((errors >= np.asarray(spec["lower"])-TOL) &
           (errors <= np.asarray(spec["upper"])+TOL)).all(axis=1)
    for group in spec["groups"]:
        values = errors[:, group["indices"]].sum(axis=1)
        yes &= (values >= group["lower"]-TOL) & (values <= group["upper"]+TOL)
    return yes


def distribute(values, lower, upper, target):
    """Reach a feasible sum along the box from a deterministic starting vector."""
    values = np.clip(np.asarray(values, float), lower, upper)
    delta = float(target-values.sum())
    room = upper-values if delta >= 0 else values-lower
    if abs(delta) > room.sum()+TOL:
        raise ValueError("Infeasible aggregate bound after physical intersection")
    if abs(delta) > 1e-12 and room.sum() > 0:
        values += np.sign(delta)*room*(abs(delta)/room.sum())
    return values


def feasible_witness(spec, regions):
    lo, hi = np.asarray(spec["lower"]), np.asarray(spec["upper"])
    if np.any(lo > hi+TOL):
        raise ValueError("Empty coordinate interval")
    values = np.clip(np.zeros(len(lo)), lo, hi)
    if not spec["groups"]:
        return values
    global_band = spec["groups"][0]
    if len(spec["groups"]) == 1:
        return distribute(values, lo, hi, np.clip(values.sum(), global_band["lower"], global_band["upper"]))
    regional = spec["groups"][1:]
    group_lo = np.array([max(lo[g["indices"]].sum(), g["lower"]) for g in regional])
    group_hi = np.array([min(hi[g["indices"]].sum(), g["upper"]) for g in regional])
    if np.any(group_lo > group_hi+TOL):
        raise ValueError("Empty regional support")
    total_lo = max(group_lo.sum(), global_band["lower"])
    total_hi = min(group_hi.sum(), global_band["upper"])
    if total_lo > total_hi+TOL:
        raise ValueError("Global and regional physical supports do not intersect")
    group_start = np.array([values[g["indices"]].sum() for g in regional])
    group_start = np.clip(group_start, group_lo, group_hi)
    group_target = distribute(group_start, group_lo, group_hi,
                              np.clip(group_start.sum(), total_lo, total_hi))
    for group, target in zip(regional, group_target):
        indices = np.asarray(group["indices"])
        values[indices] = distribute(values[indices], lo[indices], hi[indices], target)
    return values


def implied_marginals(spec, regions):
    """Exact projected bounds for a box plus disjoint regional/global sums."""
    lo, hi = np.asarray(spec["lower"]), np.asarray(spec["upper"])
    effective_lo, effective_hi = lo.copy(), hi.copy()
    if not spec["groups"]: return effective_lo, effective_hi
    global_band = spec["groups"][0]
    if len(spec["groups"]) == 1:
        return (np.maximum(lo, global_band["lower"]-(hi.sum()-hi)),
                np.minimum(hi, global_band["upper"]-(lo.sum()-lo)))
    regional = spec["groups"][1:]
    group_lo = np.array([max(lo[g["indices"]].sum(), g["lower"]) for g in regional])
    group_hi = np.array([min(hi[g["indices"]].sum(), g["upper"]) for g in regional])
    for k, group in enumerate(regional):
        idx = np.asarray(group["indices"])
        gmin = max(group_lo[k], global_band["lower"]-(group_hi.sum()-group_hi[k]))
        gmax = min(group_hi[k], global_band["upper"]-(group_lo.sum()-group_lo[k]))
        effective_lo[idx] = np.maximum(lo[idx], gmin-(hi[idx].sum()-hi[idx]))
        effective_hi[idx] = np.minimum(hi[idx], gmax-(lo[idx].sum()-lo[idx]))
    return effective_lo, effective_hi


def build_specs(data, metadata, regions):
    historical = np.asarray(data["calibration_errors"], float)
    times = pd.DatetimeIndex(data["calibration_times_ns"].astype("datetime64[ns]"))
    if historical.shape != (len(times), 568) or not np.isfinite(historical).all():
        raise ValueError("Malformed historical residuals")
    if not times.max() < pd.Timestamp(metadata["origin"]):
        raise ValueError("Calibration includes a non-past time")
    if np.max(np.abs(historical.sum(axis=1))) > TOL:
        raise ValueError("Historical node0 identity fails")
    errors = historical[:, 1:]
    dates = np.asarray(times.strftime("%Y-%m-%d"))
    days = sorted(set(dates))
    if len(days) < 3: raise ValueError("Need fit dates and two calibration dates")
    fit = np.isin(dates, days[:-2]); calibration = ~fit
    if fit.sum() < 2 or calibration.sum() == 0: raise ValueError("Insufficient residual history")
    total = errors.sum(axis=1)
    regional = np.column_stack([errors[:, g["indices"]].sum(axis=1) for g in regions])
    station_center, station_scale = center_scale(errors[fit])
    global_center, global_scale = center_scale(total[fit])
    region_center, region_scale = center_scale(regional[fit])
    station_score = np.max(np.abs((errors-station_center)/station_scale), axis=1)
    global_score = np.abs((total-global_center)/global_scale)
    region_score = np.max(np.abs((regional-region_center)/region_scale), axis=1)
    base_score = np.maximum(station_score, global_score)
    joint_score = np.maximum(base_score, region_score)
    base_radius = max(1., float(base_score[calibration].max()))
    radius = max(1., float(joint_score[calibration].max()))
    shift = np.asarray(data["center_shift"], float)
    raw_lo = station_center-radius*station_scale+shift
    raw_hi = station_center+radius*station_scale+shift
    lower = np.maximum(raw_lo, -data["predicted"])
    upper = np.minimum(raw_hi, data["station_capacity"]-data["predicted"])
    global_band = [float(global_center-radius*global_scale+shift.sum()),
                   float(global_center+radius*global_scale+shift.sum())]
    region_shift = np.array([shift[g["indices"]].sum() for g in regions])
    region_lower = region_center-radius*region_scale+region_shift
    region_upper = region_center+radius*region_scale+region_shift
    global_group = dict(name="global_signed_sum", indices=list(range(567)),
                        lower=global_band[0], upper=global_band[1])
    regional_groups = [dict(name="regional_signed_sum_"+g["name"], indices=g["indices"],
                           lower=float(region_lower[k]), upper=float(region_upper[k]))
                       for k, g in enumerate(regions)]
    specs = {}
    for arm, groups in (("B", []), ("C0", [global_group]),
                        ("C0V", [global_group]+regional_groups)):
        specs[arm] = dict(name=arm, lower=lower.copy(), upper=upper.copy(), groups=groups,
                          linear_rows=[], l1_budget=None)
    specs["D"] = dict(name="D", lower=np.zeros(567), upper=np.zeros(567), groups=[],
                       linear_rows=[], l1_budget=None)
    shifted_history = errors+shift
    cover = {}
    for arm, spec in specs.items():
        witness = feasible_witness(spec, regions)
        if not inside(spec, witness).all(): raise AssertionError("Feasible witness failed")
        spec["feasible_error"] = witness
        if arm == "D": continue
        raw_spec = dict(spec, lower=raw_lo, upper=raw_hi)
        statistical = inside(raw_spec, shifted_history)
        if not statistical[calibration].all(): raise AssertionError("Joint calibration coverage failed")
        physical = inside(spec, shifted_history)
        effective_lo, effective_hi = implied_marginals(spec, regions)
        marginal_change = int(np.count_nonzero((effective_lo > lower+TOL) | (effective_hi < upper-TOL)))
        cover[arm] = dict(statistical_calibration_count=int(statistical[calibration].sum()),
                          calibration_denominator=int(calibration.sum()),
                          statistical_history_count=int(statistical.sum()),
                          history_denominator=len(errors),
                          current_physical_calibration_count=int(physical[calibration].sum()),
                          effective_marginals_changed_coordinates=marginal_change,
                          same_explicit_box=True)
    detail = dict(
        rule="shared max joint station/global/four-regional score; fit earlier dates and calibrate last two matched dates",
        fit_days=days[:-2], calibration_days=days[-2:], fit_rows=int(fit.sum()),
        calibration_rows=int(calibration.sum()), history_rows=len(errors),
        historical_first=str(times.min()), historical_last=str(times.max()),
        base_station_global_radius=base_radius, common_joint_radius=radius,
        radius_increased_due_to_regions=radius > base_radius+TOL,
        fit_geometry=dict(station_center=station_center, station_scale=station_scale,
                          global_center=global_center, global_scale=global_scale,
                          regional_center=region_center, regional_scale=region_scale),
        raw_station_lower=raw_lo, raw_station_upper=raw_hi,
        global_band=global_band, regional_bands=np.column_stack([region_lower, region_upper]),
        calibration_scores=dict(station=station_score[calibration], global_sum=global_score[calibration],
                                region=region_score[calibration], joint=joint_score[calibration]),
        membership=cover, history_conservation_max_abs=float(np.abs(historical.sum(axis=1)).max()),
        coordinates="actual inventory minus common physically clipped nominal forecast",
        statistical_coverage_is_not_a_future_probability_guarantee=True,
        historical_vectors_intersected_with_current_physical_box_are_not_historical_inventory_validation=True,
        region_interval_is_statistical_not_exact_zero_conservation=True,
        virtual_node_identity="raw epsilon_Vg = - sum(raw station epsilon in g); virtual complement is bookkeeping only",
        current_center_shift=shift,
        test_actual_read=False)
    return specs, detail


def main():
    manifest = read(SOURCE / "manifest.json")
    cases = manifest["cases"]
    if [c["sample_id"] for c in cases] != list(range(176, 320)):
        raise AssertionError("The exact 144 origins must be retained in order")
    source_paths = [SOURCE/"manifest.json", SOURCE/"protocol_freeze.json"]
    for case in cases:
        source_paths += [SOURCE/case["case"]/name for name in
                         ("instance.npz", "metadata.json", "uncertainty_specs.json", "calibration.json")]
    hashes_before = {str(path): sha(path) for path in source_paths}
    first = SOURCE / cases[0]["case"] / "instance.npz"
    with np.load(first, allow_pickle=False) as archive:
        coords = archive["coords"].copy()
        station_ids = archive["station_ids"].copy()
    regions, labels, geography = geographic_regions(coords)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT/"regions.json", dict(regions=regions, labels=labels, geography=geography,
                                    station_ids=station_ids))
    protocol = dict(
        sample_ids=list(range(176, 320)), date="2017-07-27", arms=["B", "C0", "C0V", "D"],
        station_count=567, vehicle_count=4, truck_capacity=30,
        actual_arrays_allowed=False, small_instances_created=False,
        target_and_capacity_rule="reuse each original instance exactly",
        service_rule="every station optional, at most once; unvisited full target-deviation penalty",
        geography=geography,
        uncertainty_rule="nested shared box; C0 adds global sum; C0V adds four disjoint signed regional sums",
        calibration_rule="one joint max radius for all robust arms, using station/global/regional history scores; 100 percent empirical calibration coverage before physical clipping",
        nominal_rule="clip original forecast once to [0, capacity] for ALL arms; translate all robust error bounds and aggregate bands by original minus clipped forecast; targets unchanged",
        nominal_clip_reason="D must use a physically feasible inventory; translation does not change robust physical states at a fixed radius",
        fixed_radius_contains_regional_information_for_all_arms=True,
        predictor_retrained=False,
        distance_cap_rule="reuse previously frozen per-instance distance cap, shared across arms",
        inherited_limitations=[
            "retrospective snapshots previously exposed in coverage diagnostics, not independent holdout",
            "prediction dynamic graph lookahead in inherited export",
            "inventory reconstructed retrospectively from completed trips",
            "station capacities estimated using full period",
            "validation residuals overlap checkpoint selection data",
            "ten-minute forecast horizon is not full route execution time",
            "snapshot target deviation is not customer unmet demand or closed-loop service performance"],
        source_sha256=hashes_before, original_protocol=read(SOURCE/"protocol_freeze.json"),
        prepare_script_sha256=sha(Path(__file__)), failure_rule="retain failure report and stop; never delete a failed origin")
    write(OUT/"protocol_freeze.json", protocol)
    records = []
    for case in cases:
        name = case["case"]
        try:
            source_dir = SOURCE/name
            metadata = read(source_dir/"metadata.json")
            with np.load(source_dir/"instance.npz", allow_pickle=False) as archive:
                if any("actual" in key.lower() for key in archive.files):
                    raise ValueError("Actual outcome member unexpectedly present in source instance")
                data = {key: archive[key].copy() for key in archive.files}
            if not np.array_equal(data["coords"], coords) or not np.array_equal(data["station_ids"], station_ids):
                raise AssertionError("Coordinate or station ordering drift")
            old_predicted = data["predicted"].copy()
            clipped = np.clip(old_predicted, 0, data["station_capacity"])
            data["predicted_original"] = old_predicted
            data["center_shift"] = old_predicted-clipped
            data["predicted"] = clipped
            data["original_candidate_mask"] = data["candidate_mask"].copy()
            data["candidate_mask"] = np.ones(567, dtype=bool)
            data["demand_nominal_before_center_shift"] = data["demand_nominal"].copy()
            data["demand_nominal"] = data["target_inventory"]-clipped
            data["region_labels"] = labels.copy()
            specs, calibration = build_specs(data, metadata, regions)
            metadata.update(
                original_dispatch_candidates=int(data["original_candidate_mask"].sum()),
                dispatch_candidates=567, service_policy="all 567 optional; full loss retained for all stations",
                region_counts=geography["counts"], center_shift_nonzero_count=int(np.count_nonzero(data["center_shift"])),
                center_shift_total=float(data["center_shift"].sum()),
                center_shift_rule=protocol["nominal_rule"],
                original_source=str(source_dir), raw_source_sha256=hashes_before[str(source_dir/"instance.npz")],
                arms=["B", "C0", "C0V", "D"],
                inherited_limitations=protocol["inherited_limitations"])
            directory = OUT/name
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory/"instance.npz", **data)
            write(directory/"metadata.json", metadata)
            write(directory/"uncertainty_specs.json", specs)
            write(directory/"calibration.json", calibration)
            files = [directory/file for file in ("instance.npz", "metadata.json", "uncertainty_specs.json", "calibration.json")]
            records.append(dict(case=name, sample_id=case["sample_id"], origin=metadata["origin"],
                                directory=str(directory), path=str(directory),
                                distance_cap_km=metadata["distance_cap_km"],
                                radius=calibration["common_joint_radius"],
                                base_radius=calibration["base_station_global_radius"],
                                center_shift_nonzero_count=metadata["center_shift_nonzero_count"],
                                output_sha256={path.name: sha(path) for path in files},
                                marginal_changes={arm: calibration["membership"][arm]["effective_marginals_changed_coordinates"] for arm in ROBUST_ARMS}))
        except Exception as exc:
            write(OUT/"failure.json", dict(case=name, exception=repr(exc), completed=len(records)))
            raise
    hashes_after = {str(path): sha(path) for path in source_paths}
    if hashes_before != hashes_after: raise AssertionError("Original input files changed during preparation")
    index = dict(cases=records, count=len(records), station_count=567,
                 sample_ids=list(range(176, 320)), regions=regions,
                 source_hashes_unchanged=True, actual_arrays_opened=False,
                 region_enlarged_radius_cases=sum(r["radius"] > r["base_radius"]+TOL for r in records),
                 physically_clipped_nominal_cases=sum(r["center_shift_nonzero_count"] > 0 for r in records),
                 marginal_changed_case_counts={arm: sum(r["marginal_changes"][arm] > 0 for r in records) for arm in ROBUST_ARMS})
    write(OUT/"index.json", index)
    print(json.dumps({key: value for key, value in index.items() if key not in ("cases", "regions", "sample_ids")}, indent=2))


if __name__ == "__main__":
    main()
