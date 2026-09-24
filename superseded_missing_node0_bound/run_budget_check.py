"""Pre-outcome registered computation-budget check on 12 full-size origins.

These runs never replace main routes. All four arms receive 60 seconds and
the same seed/initialization. The only changed factor is search time.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
from run_independent import HERE, ARMS, CONFIG, read, write, sha, worker


def main():
    index = read(HERE/"inputs/index.json")
    selected = [c for c in index["cases"] if (c["sample_id"]-176) % 12 == 0]
    out = HERE/"results/budget_check"
    settings = dict(search_seconds=60., max_iterations=1000000,
                    candidate_oracle_seconds=.15, initial_oracle_seconds=3.,
                    final_oracle_seconds=10., initial_temperature=12.)
    protocol = dict(purpose="check finite-search-budget sensitivity; no change to main estimates or route selection",
                    selected_cases=[c["case"] for c in selected], methods=ARMS, seed=17,
                    selection="same predeclared 12 two-hour origins as main seed stability",
                    expected_runs=48, settings=settings, config=CONFIG,
                    only_changed_factor="12 to 60 seconds of search",
                    trigger="C0V evaluates materially fewer route proposals at equal CPU wall budgets; inspected before actual outcomes",
                    actual_read_at_registration=False,
                    source_sha256={str(p): sha(p) for p in [Path(__file__), HERE/"run_independent.py", HERE/"robust_evaluator.py", HERE/"inputs/index.json"]})
    frozen = out/"protocol.json"
    from run_independent import clean
    if frozen.exists() and read(frozen) != clean(protocol): raise RuntimeError("Budget protocol drift")
    write(frozen, protocol)
    write(out/"progress.json", dict(status="registered_waiting_for_main", expected_runs=48))
    # Register now, but use the same worker count and avoid concurrent campaigns.
    while not (HERE/"results/main/optimization_freeze.json").exists(): time.sleep(2)
    main_freeze = read(HERE/"results/main/optimization_freeze.json")
    if not main_freeze["complete"]: raise RuntimeError("Main campaign incomplete")
    jobs = [(case, method, 17, settings, str(out/"runs"/f"{case['case']}__{method}__s17.json"))
            for case in selected for method in ARMS]
    jobs = [job for job in jobs if not Path(job[-1]).exists()]
    failures = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for count, future in enumerate(as_completed([pool.submit(worker, job) for job in jobs]), 1):
            result = future.result()
            if "failure" in result: failures.append(result)
            print(__import__("json").dumps(clean(result)), flush=True)
            write(out/"progress.json", dict(status="running", completed_on_this_invocation=count,
                                           expected_runs=48, failures=failures))
    paths = sorted(p for p in (out/"runs").glob("*.json") if not p.name.endswith(".failure.json"))
    freeze = dict(complete=len(paths)==48 and not failures, expected=48, completed=len(paths),
                  hashes={str(p): sha(p) for p in paths}, source_sha256=protocol["source_sha256"],
                  failures=failures, actual_arrays_opened=False, main_routes_unchanged=True)
    write(out/"optimization_freeze.json", freeze)
    write(out/"progress.json", dict(status="complete" if freeze["complete"] else "incomplete", **freeze))
    if not freeze["complete"]: raise RuntimeError("Budget check incomplete")


if __name__ == "__main__": main()
