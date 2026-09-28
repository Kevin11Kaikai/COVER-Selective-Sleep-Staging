"""Three-route FIT development. Replay is default; no CERT/OUTER entrypoint."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key, rel in (("MPLCONFIGDIR", "matplotlib"), ("TEMP", "tmp"), ("TMP", "tmp")):
    path = ROOT / "outputs/bceanchor_upgrade/cache" / rel
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "src"),
               str(ROOT.parent / "AutoML_Claude")]


def main():
    import argparse
    from threadpoolctl import threadpool_limits
    from new_objective import bceanchor_upgrade as experiment
    from new_objective.bceanchor_upgrade_core import UpgradeBudget, UOUT
    from new_objective.storage import write, utc
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=["preflight", "pilot", "fit", "replay", "report"],
                        default="replay")
    parser.add_argument("--branch", choices=["selection", "averaging", "nested", "all"],
                        default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # One process owns the incremental ledger and fitting outputs at a time.
    import msvcrt
    lock = (UOUT / "run.lock").open("a+b")
    lock.seek(0)
    if not lock.read(1):
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        raise RuntimeError("Another upgrade command owns the study lock")
    experiment.install_guards()
    budget = UpgradeBudget(args.mode)
    try:
        with threadpool_limits(limits=2):
            if args.mode in ("replay", "report"):
                from new_objective import bceanchor_upgrade_report as reporting
                getattr(reporting, args.mode)(budget, args.branch)
            else:
                getattr(experiment, args.mode)(budget, args.resume, args.branch)
        budget.finish("COMPLETED")
    except BaseException as error:
        budget.finish(type(error).__name__)
        write(UOUT / "last_failure.json", {"time": utc(), "mode": args.mode,
              "branch": args.branch, "error": type(error).__name__, "message": str(error)})
        raise
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


if __name__ == "__main__":
    main()
