#!/usr/bin/env python
"""Legacy wrapper: compare baseline daily models via forecast_daily runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy entrypoint. Delegates to the canonical forecast_daily baseline comparison pipeline."
        )
    )
    parser.add_argument("--csv", default=None, help="Optional CSV with columns: date, airnow_no2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--train-end", type=str, default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["transformer", "mamba", "gnn"],
        default=["transformer", "mamba", "gnn"],
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    target = root / "forecast_daily" / "generate_results.py"

    cmd = [
        sys.executable,
        str(target),
        "--models",
        *args.models,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--results-dir",
        str(args.results_dir),
        "--train-end",
        str(args.train_end),
    ]
    if args.csv:
        cmd.extend(["--csv", args.csv])

    print("Legacy wrapper active: delegating to forecast_daily/generate_results.py")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
