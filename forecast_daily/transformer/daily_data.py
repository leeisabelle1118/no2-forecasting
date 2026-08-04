"""Compatibility shim for the canonical daily dataset contract.

The single source of truth is forecast_daily/daily_data_contract.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from daily_data_contract import *  # noqa: F401,F403
