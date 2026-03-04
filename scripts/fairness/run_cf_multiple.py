#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def find_run_dirs(outputs_root: Path, state: str, prefixes: List[str]) -> List[Path]:
    state_dir = outputs_root / state
    if not state_dir.exists():
        return []

    dirs = [p for p in state_dir.iterdir() if p.is_dir()]
    out: List[Path] = []
    for d in dirs:
        name = d.name
        if any(name.startswith(pref) for pref in prefixes):
            out.append(d)

    # stable ordering
    return sorted(out, key=lambda p: p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="apps/web/public/outputs", help="Root outputs folder")
    ap.add_argument("--states", nargs="+", required=True, help="State keys, e.g. il ny")
    ap.add_argument(
        "--prefixes",
        nargs="+",
        default=["current_congress", "kmeans", "hillclimb_dem", "hillclimb_rep"],
        help="Run folder name prefixes to include",
    )
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if fairness.json already exists")
    ap.add_argument("--area-epsg", type=int, default=5070)
    ap.add_argument("--votes-to-win-mode", default="majority_plus_one", choices=["majority_plus_one", "half"])
    ap.add_argument("--competitive-band", type=float, default=0.05)
    ap.add_argument("--packed-margin", type=float, default=0.10)
    ap.add_argument("--cracked-margin", type=float, default=0.05)
    args = ap.parse_args()

    outputs_root = Path(args.outputs_root)
    compute_script = Path("scripts/fairness/compute_run_fairness.py")

    if not compute_script.exists():
        raise FileNotFoundError(f"Missing: {compute_script}")

    total = 0
    ran = 0

    for state in args.states:
        run_dirs = find_run_dirs(outputs_root, state, args.prefixes)
        if not run_dirs:
            print(f"[warn] no matching run folders for state={state} under {outputs_root/state}")
            continue

        for run_dir in run_dirs:
            total += 1
            out_path = run_dir / "fairness.json"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {run_dir} (fairness.json exists)")
                continue

            cmd = [
                sys.executable,
                str(compute_script),
                "--run-dir", str(run_dir),
                "--area-epsg", str(args.area_epsg),
                "--votes-to-win-mode", args.votes_to_win_mode,
                "--competitive-band", str(args.competitive_band),
                "--packed-margin", str(args.packed_margin),
                "--cracked-margin", str(args.cracked_margin),
            ]
            print(f"[run]  {run_dir}")
            subprocess.check_call(cmd)
            ran += 1

    print(f"\nDone. processed={total}, computed={ran}")


if __name__ == "__main__":
    main()