#!/usr/bin/env python
"""Legacy wrapper: route root training to forecast_daily baseline runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASELINE_MODELS = ["transformer", "mamba", "gnn"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy entrypoint. Use forecast_daily baseline pipeline with K=7 and direct t+1 forecasting."
        )
    )
    parser.add_argument("--model", choices=BASELINE_MODELS, default="transformer")
    parser.add_argument("--csv", default=None, help="Optional CSV with columns: date, airnow_no2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--train-end", type=str, default="auto")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    target = root / "forecast_daily" / "generate_results.py"
    cmd = [
        sys.executable,
        str(target),
        "--models",
        args.model,
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
