"""Explicit lightweight fitting or portable replay of the COVER selection ablation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fit", "replay"], default="replay")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Complete outer-00 after all FIT selections")
    parser.add_argument("--run-dir", default="outputs/strengthening/stage1_strengthening_v1")
    args = parser.parse_args()
    from stage1_analysis.selection_ablation import fit, replay
    if args.mode == "fit":
        fit(ROOT, resume=args.resume, pilot=args.pilot)
    if not args.pilot:
        out = (ROOT / args.run_dir).resolve()
        out.relative_to(ROOT)
        replay(ROOT, out)


if __name__ == "__main__":
    main()
