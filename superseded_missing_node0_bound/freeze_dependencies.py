"""Record read-only algorithm dependencies during computation, before replay."""
from pathlib import Path
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def main():
    import numpy
    import scipy
    path = HERE / "results/dependency_freeze.json"
    dependencies = [STUDY/"conservation_exploration/compact_oracle.py",
                    STUDY/"three_set_study/oracle.py", STUDY/"three_set_study/routing.py",
                    STUDY/"three_set_study/uncertainty.py"]
    dependencies += list((STUDY/"conservation_joint_study").glob("*.py"))
    dependencies += list((STUDY/"b_elimination_full_study").glob("*.py"))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(dependencies))}
    if path.exists():
        original = json.loads(path.read_text(encoding="utf-8"))
        changed = [p for p, digest in original["sha256"].items() if hashes.get(p) != digest]
        result = dict(files=len(hashes), changed=changed, unchanged=not changed)
        (HERE/"results/dependency_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if changed: raise AssertionError(changed)
    else:
        result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                      timing="during optimization, before outcome replay; complements campaign source hashes",
                      python=sys.version, executable=sys.executable, platform=platform.platform(),
                      numpy=numpy.__version__, scipy=scipy.__version__, sha256=hashes)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(files=len(hashes), frozen=True)))


if __name__ == "__main__": main()
