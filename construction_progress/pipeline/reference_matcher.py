"""
Reference image matcher.

Compares the current zone crop against a bank of reference images
(0%.jpg, 10%.jpg ... 100%.jpg) stored per zone.

The pipeline uses this INSTEAD of (or alongside) the AI scorer when
reference images are available — giving a much more accurate % because
the AI is comparing against known ground-truth milestones.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REFERENCE_BASE = Path(__file__).parent.parent / "data" / "reference_images"


@dataclass
class ReferenceMatch:
    matched_pct: float           # e.g. 40.0
    lower_pct: float             # closest reference below
    upper_pct: float             # closest reference above
    lower_similarity: float      # 0-1
    upper_similarity: float      # 0-1
    interpolated_pct: float      # fine-grained estimate between two refs
    method: str                  # "reference_bank" or "ai_only"
    confidence: float            # 0-1


def _load_reference_images(zone_id: str) -> dict[float, np.ndarray]:
    """
    Load all reference images for a zone.
    Returns {percentage: cropped_image} sorted by percentage.
    """
    zone_dir = REFERENCE_BASE / zone_id
    if not zone_dir.exists():
        return {}

    refs = {}
    for f in zone_dir.iterdir():
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            pct = float(f.stem)
            img = cv2.imread(str(f))
            if img is not None:
                refs[pct] = img
        except ValueError:
            logger.warning("Skipping non-numeric reference file: %s", f.name)

    if refs:
        logger.info("Loaded %d reference images for zone '%s'", len(refs), zone_id)
    return dict(sorted(refs.items()))


def _crop_polygon(image: np.ndarray, polygon_frac: list[list[float]]) -> np.ndarray:
    """Crop polygon region from image."""
    h, w = image.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in polygon_frac], dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    x = max(0, x); y = max(0, y)
    x2 = min(w, x + bw); y2 = min(h, y + bh)
    return image[y:y2, x:x2]


def _image_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Multi-metric similarity between two images (0=different, 1=identical).
    Combines:
      - Histogram correlation (colour distribution)
      - ORB feature match ratio (structural similarity)
      - Normalised pixel diff (texture change)
    """
    # Resize both to same size for fair comparison
    target = (256, 256)
    a = cv2.resize(img_a, target, interpolation=cv2.INTER_AREA)
    b = cv2.resize(img_b, target, interpolation=cv2.INTER_AREA)

    scores = []

    # 1. Histogram correlation (per channel)
    hist_scores = []
    for ch in range(3):
        ha = cv2.calcHist([a], [ch], None, [64], [0, 256])
        hb = cv2.calcHist([b], [ch], None, [64], [0, 256])
        cv2.normalize(ha, ha); cv2.normalize(hb, hb)
        score = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
        hist_scores.append(max(0, score))
    scores.append(np.mean(hist_scores) * 0.35)

    # 2. ORB feature matching
    try:
        orb = cv2.ORB_create(nfeatures=300)
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        _, da = orb.detectAndCompute(ga, None)
        _, db = orb.detectAndCompute(gb, None)
        if da is not None and db is not None and len(da) > 5 and len(db) > 5:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(da, db)
            good = [m for m in matches if m.distance < 60]
            ratio = min(1.0, len(good) / max(len(da), len(db)))
            scores.append(ratio * 0.40)
        else:
            scores.append(0.0)
    except Exception:
        scores.append(0.0)

    # 3. Normalised absolute difference (inverted — lower diff = higher score)
    diff = cv2.absdiff(a.astype(np.float32), b.astype(np.float32))
    norm_diff = diff.mean() / 255.0
    pixel_sim = max(0.0, 1.0 - norm_diff * 4)
    scores.append(pixel_sim * 0.25)

    return min(1.0, sum(scores))


def _interpolate_pct(
    lower_pct: float, lower_sim: float,
    upper_pct: float, upper_sim: float,
) -> float:
    """
    Interpolate between two reference percentages based on similarity scores.
    If current is closer to upper → score is higher within the range.
    """
    total = lower_sim + upper_sim
    if total == 0:
        return (lower_pct + upper_pct) / 2
    # Weight toward whichever reference is more similar
    weight_upper = upper_sim / total
    return round(lower_pct + (upper_pct - lower_pct) * weight_upper, 1)


class ReferenceMatcher:
    """
    Matches a zone crop against the reference image bank to estimate % complete.
    Falls back gracefully if no reference images exist for the zone.
    """

    def __init__(self):
        self._cache: dict[str, dict[float, np.ndarray]] = {}

    def _get_refs(self, zone_id: str) -> dict[float, np.ndarray]:
        if zone_id not in self._cache:
            self._cache[zone_id] = _load_reference_images(zone_id)
        return self._cache[zone_id]

    def has_references(self, zone_id: str) -> bool:
        return bool(self._get_refs(zone_id))

    def match(
        self,
        zone_id: str,
        current_crop: np.ndarray,
        polygon_frac: Optional[list[list[float]]] = None,
    ) -> Optional[ReferenceMatch]:
        """
        Compare current_crop against all reference images for this zone.
        Returns None if no reference images are available.

        If polygon_frac is provided, crops reference images to the same zone
        before comparing (use this when reference images are full-frame shots).
        """
        refs = self._get_refs(zone_id)
        if not refs:
            logger.info("No reference images for zone '%s' — skipping reference match.", zone_id)
            return None

        logger.info("Matching zone '%s' against %d reference images...", zone_id, len(refs))

        # Compute similarity to every reference
        similarities: dict[float, float] = {}
        for pct, ref_img in refs.items():
            # Crop reference to same polygon if full-frame reference provided
            if polygon_frac and ref_img.shape[:2] != current_crop.shape[:2]:
                ref_crop = _crop_polygon(ref_img, polygon_frac)
            else:
                ref_crop = ref_img

            sim = _image_similarity(current_crop, ref_crop)
            similarities[pct] = sim
            logger.debug("  Ref %3.0f%%: similarity=%.3f", pct, sim)

        # Find the best matching reference
        best_pct = max(similarities, key=similarities.get)
        best_sim = similarities[best_pct]

        pcts = sorted(similarities.keys())

        # Find the two references that bracket the current state
        # Strategy: find lower (best match below best_pct) and upper (above)
        idx = pcts.index(best_pct)
        lower_pct = pcts[idx - 1] if idx > 0 else best_pct
        upper_pct = pcts[idx + 1] if idx < len(pcts) - 1 else best_pct

        lower_sim = similarities[lower_pct]
        upper_sim = similarities[upper_pct]

        # Interpolate for finer-grained estimate
        if lower_pct == upper_pct:
            interpolated = best_pct
        else:
            interpolated = _interpolate_pct(lower_pct, lower_sim, upper_pct, upper_sim)

        # Confidence: how clearly does one reference win?
        sorted_sims = sorted(similarities.values(), reverse=True)
        if len(sorted_sims) >= 2:
            confidence = min(1.0, (sorted_sims[0] - sorted_sims[1]) * 5 + 0.5)
        else:
            confidence = best_sim

        logger.info(
            "Zone '%s' matched: best=%.0f%% (sim=%.3f), interpolated=%.1f%%, confidence=%.2f",
            zone_id, best_pct, best_sim, interpolated, confidence,
        )

        return ReferenceMatch(
            matched_pct=best_pct,
            lower_pct=lower_pct,
            upper_pct=upper_pct,
            lower_similarity=lower_sim,
            upper_similarity=upper_sim,
            interpolated_pct=interpolated,
            method="reference_bank",
            confidence=round(confidence, 3),
        )
