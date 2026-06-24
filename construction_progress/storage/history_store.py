"""
History Store — persists snapshots, zone features, and scores to disk.
Provides image pair selection (current vs. nearest matching angle).
"""

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from construction_progress.config import HISTORY_DIR, OUTPUT_DIR, MAX_HISTORY_FRAMES

logger = logging.getLogger(__name__)


# ── Snapshot record ───────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    snapshot_id: str          # ISO timestamp slug
    timestamp: str            # ISO 8601
    image_path: str           # absolute path to stored image
    camera_id: str
    zone_features: dict       # zone_id -> ZoneFeatures as dict
    zone_scores: dict         # zone_id -> ZoneScore as dict
    overall_progress: float
    report_path: str          # path to JSON report


class HistoryStore:
    """
    File-based persistence layer.

    Layout:
        data/history/
            <camera_id>/
                index.json               — ordered list of snapshot records
                <timestamp>/
                    image.jpg
                    zone_features.json
                    zone_scores.json
    """

    def __init__(self, base_dir: str = HISTORY_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _camera_dir(self, camera_id: str) -> Path:
        d = self.base_dir / camera_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _index_path(self, camera_id: str) -> Path:
        return self._camera_dir(camera_id) / "index.json"

    def _load_index(self, camera_id: str) -> list[dict]:
        p = self._index_path(camera_id)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return []

    def _save_index(self, camera_id: str, index: list[dict]):
        with open(self._index_path(camera_id), "w") as f:
            json.dump(index, f, indent=2)

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_snapshot(
        self,
        camera_id: str,
        image: np.ndarray,
        zone_features: dict,   # zone_id -> ZoneFeatures
        zone_scores: dict,     # zone_id -> ZoneScore
        overall_progress: float,
        report: dict,
    ) -> Snapshot:
        ts = datetime.utcnow()
        slug = ts.strftime("%Y%m%dT%H%M%SZ")
        snap_dir = self._camera_dir(camera_id) / slug
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        img_path = snap_dir / "image.jpg"
        cv2.imwrite(str(img_path), image)

        # Serialise features (dataclasses → dict, drop crop_bytes)
        def clean_features(feat_map: dict) -> dict:
            out = {}
            for zid, f in feat_map.items():
                d = asdict(f) if hasattr(f, "__dataclass_fields__") else dict(f)
                d.pop("crop_bytes", None)
                out[zid] = d
            return out

        def clean_scores(score_map: dict) -> dict:
            out = {}
            for zid, s in score_map.items():
                out[zid] = asdict(s) if hasattr(s, "__dataclass_fields__") else dict(s)
            return out

        feats_path = snap_dir / "zone_features.json"
        with open(feats_path, "w") as f:
            json.dump(clean_features(zone_features), f, indent=2)

        scores_path = snap_dir / "zone_scores.json"
        with open(scores_path, "w") as f:
            json.dump(clean_scores(zone_scores), f, indent=2)

        # Save report
        report_path = snap_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        snap = Snapshot(
            snapshot_id=slug,
            timestamp=ts.isoformat() + "Z",
            image_path=str(img_path),
            camera_id=camera_id,
            zone_features=clean_features(zone_features),
            zone_scores=clean_scores(zone_scores),
            overall_progress=overall_progress,
            report_path=str(report_path),
        )

        # Update index
        index = self._load_index(camera_id)
        index.append(asdict(snap))
        self._save_index(camera_id, index)

        logger.info("Snapshot saved: %s/%s", camera_id, slug)
        return snap

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_latest_snapshot(self, camera_id: str) -> Optional[dict]:
        index = self._load_index(camera_id)
        if not index:
            return None
        return index[-1]

    def get_history(self, camera_id: str, limit: int = MAX_HISTORY_FRAMES) -> list[dict]:
        index = self._load_index(camera_id)
        return index[-limit:]

    def get_reference_image(self, camera_id: str) -> Optional[np.ndarray]:
        """Return the most recent stored image for alignment reference."""
        snap = self.get_latest_snapshot(camera_id)
        if snap is None:
            return None
        img = cv2.imread(snap["image_path"])
        if img is None:
            logger.warning("Reference image missing: %s", snap["image_path"])
        return img

    def get_zone_features_history(self, camera_id: str, zone_id: str) -> list[dict]:
        """Return ordered list of zone feature records for a given zone."""
        history = self.get_history(camera_id)
        result = []
        for snap in history:
            feats = snap.get("zone_features", {}).get(zone_id)
            if feats:
                result.append({"timestamp": snap["timestamp"], **feats})
        return result

    def get_latest_zone_features(self, camera_id: str) -> dict:
        """Return zone_id -> feature dict from the most recent snapshot."""
        snap = self.get_latest_snapshot(camera_id)
        if snap is None:
            return {}
        return snap.get("zone_features", {})

    def get_latest_zone_scores(self, camera_id: str) -> dict:
        """Return zone_id -> score dict from the most recent snapshot."""
        snap = self.get_latest_snapshot(camera_id)
        if snap is None:
            return {}
        return snap.get("zone_scores", {})

    def get_progress_timeline(self, camera_id: str) -> list[dict]:
        """Return list of {timestamp, overall_progress, zone_scores} for charting."""
        history = self.get_history(camera_id, limit=1000)
        return [
            {
                "timestamp": s["timestamp"],
                "overall_progress": s["overall_progress"],
                "zones": {
                    zid: scores.get("progress_pct", 0)
                    for zid, scores in s.get("zone_scores", {}).items()
                },
            }
            for s in history
        ]

    # ── Image pair selector ───────────────────────────────────────────────────

    def select_best_reference(
        self,
        camera_id: str,
        current_image: np.ndarray,
        max_candidates: int = 5,
    ) -> Optional[np.ndarray]:
        """
        From recent history, pick the stored image most similar to current
        (using ORB feature matching score) for best alignment quality.
        """
        history = self.get_history(camera_id, limit=max_candidates)
        if not history:
            return None

        detector = cv2.ORB_create(nfeatures=1000)
        matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        gray_cur = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        _, des_cur = detector.detectAndCompute(gray_cur, None)

        best_score = -1
        best_img   = None

        for snap in reversed(history):
            ref = cv2.imread(snap["image_path"])
            if ref is None:
                continue
            gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
            _, des_ref = detector.detectAndCompute(gray_ref, None)
            if des_cur is None or des_ref is None:
                continue
            matches = matcher.match(des_cur, des_ref)
            score = len([m for m in matches if m.distance < 60])
            if score > best_score:
                best_score = score
                best_img   = ref

        if best_img is not None:
            logger.info("Best reference selected with match score %d.", best_score)
        return best_img
