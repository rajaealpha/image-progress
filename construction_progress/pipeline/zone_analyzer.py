"""
Stage 3: Zone segmentation + feature extraction + semantic diff.

Each named zone is cropped from the clean image and sent to Azure Vision
for structural analysis and embedding-style description.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from construction_progress.core.azure_client import AzureVisionClient
from construction_progress.config import DEFAULT_ZONES

logger = logging.getLogger(__name__)


@dataclass
class ZoneFeatures:
    zone_id: str
    zone_name: str
    edge_density: float          # 0-1, proxy for structural complexity
    texture_variance: float      # pixel variance, proxy for material richness
    brightness: float            # mean brightness 0-255
    structural_description: str  # Azure's natural-language description
    visible_elements: list[str]  # ["rebar", "formwork", "concrete", ...]
    occlusion_flag: bool = False
    crop_bytes: Optional[bytes] = None


@dataclass
class ZoneDiff:
    zone_id: str
    zone_name: str
    current_features: ZoneFeatures
    history_features: Optional[ZoneFeatures]
    change_summary: str          # Azure-generated change description
    structural_change_score: float  # 0-1 how much changed
    new_elements: list[str]      # elements present now but not before
    removed_elements: list[str]  # elements gone (scaffolding removed etc.)


# ── Zone crop helper ──────────────────────────────────────────────────────────

def crop_zone(image: np.ndarray, polygon_frac: list[list[float]]) -> np.ndarray:
    """
    Crop a zone from the image using fractional polygon coordinates.
    Returns a rectangular crop of the bounding box of the polygon.
    """
    h, w = image.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in polygon_frac], dtype=np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    x = max(0, x); y = max(0, y)
    x2 = min(w, x + bw); y2 = min(h, y + bh)
    return image[y:y2, x:x2]


def polygon_mask(image: np.ndarray, polygon_frac: list[list[float]]) -> np.ndarray:
    """Return masked image with only the polygon region visible."""
    h, w = image.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in polygon_frac], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    result = image.copy()
    result[mask == 0] = 0
    return result


def image_to_bytes(img: np.ndarray, quality: int = 85) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ── Feature extractor ─────────────────────────────────────────────────────────

FEATURE_EXTRACTION_PROMPT = """
You are an expert construction site analyst.
Analyse this image of a specific construction zone and respond with JSON:

{
  "structural_description": "<one sentence describing what structural stage this zone is at>",
  "visible_elements": ["<element1>", "<element2>", ...],
  "construction_stage": "<one of: site_cleared | excavation | pcc_blinding | rebar_placement | formwork | concrete_poured | curing | deshuttered | finishing | complete>",
  "occlusion": <true if workers/cranes/dust/rain significantly block the structural view>,
  "occlusion_reason": "<reason if occluded, else null>",
  "quality_issues": ["<any image quality issues like darkness, blur, rain etc>"]
}

visible_elements should include any of:
bare_earth, compacted_fill, pcc_blinding, rebar, rebar_mesh, column_starter_bars,
formwork_timber, formwork_metal, concrete_wet, concrete_cured, brick_work,
block_work, plaster, tile, scaffolding, shoring_props, deck_plate, waterproofing,
any other construction element you can identify.
"""

DIFF_PROMPT = """
You are a construction progress analyst.

You are given TWO images of the same construction zone:
- Image 1 (HISTORICAL): taken earlier
- Image 2 (CURRENT): taken now

Compare them and respond with JSON:
{
  "change_summary": "<one sentence describing what construction progress occurred>",
  "structural_change_score": <0.0 to 1.0, where 0=no change, 1=complete transformation>,
  "new_elements": ["<element present in current but not historical>"],
  "removed_elements": ["<element present in historical but now gone, e.g. formwork stripped>"],
  "progress_indicators": ["<specific observations supporting progress>"],
  "regression_indicators": ["<anything suggesting work was undone or damaged>"]
}
"""


class ZoneAnalyzer:
    """
    Analyses each defined zone in a clean image.
    Extracts features and compares against historical zone features.
    """

    def __init__(self, client: AzureVisionClient, zones: list[dict] = None):
        self.client = client
        self.zones = zones or DEFAULT_ZONES

    def _compute_cv_features(self, crop: np.ndarray) -> dict:
        """Compute OpenCV-based features (fast, no API cost)."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Edge density via Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.sum() / 255) / max(1, edges.size)

        # Texture variance
        texture_variance = float(gray.var())

        # Brightness
        brightness = float(gray.mean())

        return {
            "edge_density": round(edge_density, 4),
            "texture_variance": round(texture_variance, 2),
            "brightness": round(brightness, 2),
        }

    def extract_features(self, clean_image: np.ndarray, zone: dict) -> ZoneFeatures:
        """Extract features for a single zone using CV + Azure Vision."""
        crop = crop_zone(clean_image, zone["polygon"])
        if crop.size == 0:
            logger.warning("Empty crop for zone %s", zone["id"])
            return ZoneFeatures(
                zone_id=zone["id"],
                zone_name=zone["name"],
                edge_density=0, texture_variance=0, brightness=0,
                structural_description="Zone crop failed",
                visible_elements=[],
                occlusion_flag=True,
            )

        cv_feats = self._compute_cv_features(crop)
        crop_bytes = image_to_bytes(crop)

        logger.info("Analysing zone '%s' with Azure Vision...", zone["name"])
        try:
            result = self.client.ask_json(
                prompt=FEATURE_EXTRACTION_PROMPT,
                image_bytes_list=[crop_bytes],
                max_tokens=1000,
            )
        except Exception as e:
            logger.error("Feature extraction failed for zone %s: %s", zone["id"], e)
            result = {
                "structural_description": "Analysis failed",
                "visible_elements": [],
                "construction_stage": "unknown",
                "occlusion": False,
                "quality_issues": [str(e)],
            }

        return ZoneFeatures(
            zone_id=zone["id"],
            zone_name=zone["name"],
            edge_density=cv_feats["edge_density"],
            texture_variance=cv_feats["texture_variance"],
            brightness=cv_feats["brightness"],
            structural_description=result.get("structural_description", ""),
            visible_elements=result.get("visible_elements", []),
            occlusion_flag=result.get("occlusion", False),
            crop_bytes=crop_bytes,
        )

    def diff_zones(
        self,
        current_features: ZoneFeatures,
        history_features: Optional[ZoneFeatures],
        zone: dict,
    ) -> ZoneDiff:
        """Compare current zone features against historical ones."""
        if history_features is None or history_features.crop_bytes is None:
            return ZoneDiff(
                zone_id=zone["id"],
                zone_name=zone["name"],
                current_features=current_features,
                history_features=None,
                change_summary="No historical data — treating current as baseline.",
                structural_change_score=0.0,
                new_elements=current_features.visible_elements,
                removed_elements=[],
            )

        logger.info("Diffing zone '%s' against history...", zone["name"])
        try:
            result = self.client.ask_json(
                prompt=DIFF_PROMPT,
                image_bytes_list=[history_features.crop_bytes, current_features.crop_bytes],
                max_tokens=1500,
            )
        except Exception as e:
            logger.error("Zone diff failed for %s: %s", zone["id"], e)
            result = {
                "change_summary": f"Diff failed: {e}",
                "structural_change_score": 0.0,
                "new_elements": [],
                "removed_elements": [],
            }

        # Also compute numerical similarity as backup signal
        if current_features.crop_bytes and history_features.crop_bytes:
            cur_img = cv2.imdecode(np.frombuffer(current_features.crop_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            hist_img = cv2.imdecode(np.frombuffer(history_features.crop_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if cur_img is not None and hist_img is not None:
                try:
                    h = min(cur_img.shape[0], hist_img.shape[0])
                    w = min(cur_img.shape[1], hist_img.shape[1])
                    diff = cv2.absdiff(cur_img[:h, :w], hist_img[:h, :w])
                    pixel_change = float(diff.mean()) / 255
                    # Blend AI score and pixel score
                    ai_score = float(result.get("structural_change_score", 0.0))
                    blended = 0.7 * ai_score + 0.3 * min(1.0, pixel_change * 5)
                    result["structural_change_score"] = round(blended, 3)
                except Exception:
                    pass

        return ZoneDiff(
            zone_id=zone["id"],
            zone_name=zone["name"],
            current_features=current_features,
            history_features=history_features,
            change_summary=result.get("change_summary", ""),
            structural_change_score=float(result.get("structural_change_score", 0.0)),
            new_elements=result.get("new_elements", []),
            removed_elements=result.get("removed_elements", []),
        )

    def analyse_all_zones(
        self,
        clean_image: np.ndarray,
        history_features_map: dict[str, ZoneFeatures],
    ) -> list[ZoneDiff]:
        """Run feature extraction and diff for all configured zones."""
        results = []
        for zone in self.zones:
            current_feats = self.extract_features(clean_image, zone)
            history_feats = history_features_map.get(zone["id"])
            diff = self.diff_zones(current_feats, history_feats, zone)
            results.append(diff)
        return results
