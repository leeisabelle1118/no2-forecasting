from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Run daily forecasting training by model")
    p.add_argument("--model", choices=["transformer", "mamba", "gnn"], required=True)
    p.add_argument("--csv", default=None, help="Optional CSV with columns: date, airnow_no2")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-end", type=str, default="auto", help="Last date included in training. Use 'auto' for first full-year chronological split.")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    script_map = {
        "transformer": root / "transformer" / "train_transformer_daily.py",
        "mamba": root / "mamba" / "train_mamba_daily.py",
        "gnn": root / "gnn" / "train_gnn_daily.py",
    }

    target = script_map[args.model]
    cmd = [
        sys.executable,
        str(target),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--train-end", str(args.train_end),
    ]
    if args.csv:
        cmd.extend(["--csv", args.csv])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
