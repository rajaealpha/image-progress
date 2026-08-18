"""
Last-known-score store for the stateless /analyze API.

The API endpoint scores each request independently (no shared history), so
without this, the same zone can return wildly different % values between
calls (AI re-reads the scene differently run to run). This store persists
the last accepted score per zone_id and lets the caller clamp regressions
the same way the batch pipeline's ProgressScorer does.
"""

import json
import os
import threading
from pathlib import Path
from typing import Optional

from construction_progress.config import HISTORY_DIR

SCORES_PATH = Path(HISTORY_DIR) / "api_zone_scores.json"

_lock = threading.Lock()


def _load() -> dict:
    if not SCORES_PATH.exists():
        return {}
    try:
        with open(SCORES_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict):
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCORES_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, SCORES_PATH)  # atomic on same filesystem


def get_last_score(zone_id: str) -> Optional[dict]:
    with _lock:
        return _load().get(zone_id)


def set_last_score(zone_id: str, progress_pct: float, confidence: float, run_timestamp: str):
    with _lock:
        data = _load()
        data[zone_id] = {
            "progress_pct": progress_pct,
            "confidence": confidence,
            "run_timestamp": run_timestamp,
        }
        _save(data)
