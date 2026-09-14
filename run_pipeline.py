#!/usr/bin/env python3
"""Run the full DAIC-WOZ reproduction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Smoke test with 2 participants, 3 epochs")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    py = sys.executable
    extra = ["--max-participants", "2"] if args.quick else []

    if not args.skip_preprocess:
        run([py, "scripts/01_preprocess.py", *extra], root)

    if not args.skip_features:
        feat_args = []
        if args.quick:
            feat_args = ["--embedding", "ecapa", "--skip-covarep"]
        run([py, "scripts/02_extract_features.py", *feat_args], root)

    exp_args = ["--feature", "ecapa", "--model", "lstm", "--task", "detection"]
    if args.quick:
        exp_args.append("--quick")
    else:
        exp_args = ["--run-all"]
    run([py, "scripts/03_run_experiments.py", *exp_args], root)


if __name__ == "__main__":
    main()
