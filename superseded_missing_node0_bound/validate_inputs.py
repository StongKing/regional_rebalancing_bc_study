"""Audit all 144 real-network prepared inputs, with no evaluation outcomes."""
from pathlib import Path
import json
import numpy as np
import prepare_inputs as prep


def main():
    index = prep.read(prep.OUT / "index.json")
    protocol = prep.read(prep.OUT / "protocol_freeze.json")
    geography = prep.read(prep.OUT / "regions.json")
    groups = geography["regions"]
    assert index["count"] == 144
    assert sorted(i for g in groups for i in g["indices"]) == list(range(567))
    maximum_inventory_translation_error = 0.
    for row in index["cases"]:
        directory = Path(row["directory"])
        for name, expected in row["output_sha256"].items():
            assert prep.sha(directory/name) == expected
        with np.load(directory/"instance.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        with np.load(prep.SOURCE/row["case"]/"instance.npz", allow_pickle=False) as archive:
            for key in ("target_inventory", "station_capacity", "road_distance_km", "coords",
                        "station_ids", "calibration_errors", "calibration_times_ns"):
                assert np.array_equal(data[key], archive[key]), (row["case"], key)
            assert np.array_equal(data["original_candidate_mask"], archive["candidate_mask"])
            assert np.array_equal(data["predicted_original"], archive["predicted"])
        assert data["candidate_mask"].all()
        assert np.allclose(data["demand_nominal"], data["target_inventory"]-data["predicted"])
        assert np.all(data["predicted"] >= 0)
        assert np.all(data["predicted"] <= data["station_capacity"])
        specs = prep.read(directory/"uncertainty_specs.json")
        calibration = prep.read(directory/"calibration.json")
        assert len(specs["B"]["groups"]) == 0
        assert len(specs["C0"]["groups"]) == 1
        assert len(specs["C0V"]["groups"]) == 5
        assert specs["C0"]["groups"][0] == specs["C0V"]["groups"][0]
        for arm in prep.ROBUST_ARMS:
            assert specs[arm]["lower"] == specs["B"]["lower"]
            assert specs[arm]["upper"] == specs["B"]["upper"]
            witness = np.asarray(specs[arm]["feasible_error"])
            assert prep.inside(specs[arm], witness).all()
            realized = data["predicted"]+witness
            assert np.all(realized >= -prep.TOL)
            assert np.all(realized <= data["station_capacity"]+prep.TOL)
            coverage = calibration["membership"][arm]
            assert coverage["statistical_calibration_count"] == coverage["calibration_denominator"]
            lo, hi = prep.implied_marginals(specs[arm], groups)
            assert np.allclose(lo, specs["B"]["lower"], atol=prep.TOL, rtol=0)
            assert np.allclose(hi, specs["B"]["upper"], atol=prep.TOL, rtol=0)
        geometry = calibration["fit_geometry"]
        rho = calibration["common_joint_radius"]
        rawlo = np.asarray(geometry["station_center"])-rho*np.asarray(geometry["station_scale"])
        rawhi = np.asarray(geometry["station_center"])+rho*np.asarray(geometry["station_scale"])
        expected_low_inventory = np.maximum(data["predicted_original"]+rawlo, 0)
        expected_high_inventory = np.minimum(data["predicted_original"]+rawhi, data["station_capacity"])
        low_inventory = data["predicted"]+np.asarray(specs["B"]["lower"])
        high_inventory = data["predicted"]+np.asarray(specs["B"]["upper"])
        error = max(float(np.max(np.abs(expected_low_inventory-low_inventory))),
                    float(np.max(np.abs(expected_high_inventory-high_inventory))))
        maximum_inventory_translation_error = max(maximum_inventory_translation_error, error)
        assert error < prep.TOL
        assert prep.inside(specs["D"], np.zeros(567)).all()
    for source, expected in protocol["source_sha256"].items():
        assert prep.sha(source) == expected, source
    report = dict(status="passed", real_network_instances=144, station_count=567,
                  small_instances_used=False, actual_outcomes_read=False,
                  sources_unchanged=True, nested_sets_verified=True,
                  exact_coordinate_projection_bounds_equal_across_robust_arms=True,
                  statistical_joint_calibration_coverage=1.,
                  maximum_inventory_translation_error=maximum_inventory_translation_error)
    prep.write(prep.OUT/"validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
