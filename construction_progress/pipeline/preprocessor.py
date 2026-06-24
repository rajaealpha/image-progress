"""
Stage 1 + 2: Camera alignment and moving-object removal.

Instead of YOLO/LaMa we use the Azure Vision model to:
  1. Detect bounding regions of dynamic objects (workers, cranes, vehicles).
  2. Generate an inpainting description so OpenCV can patch those regions.

OpenCV handles the geometric alignment (homography) and pixel-level inpainting.
"""

import io
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from construction_progress.core.azure_client import AzureVisionClient
from construction_progress.config import API_IMAGE_SIZE

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    clean_image: np.ndarray          # BGR numpy array — static scene only
    dynamic_mask: np.ndarray         # uint8 mask, 255 = dynamic region removed
    detected_objects: list[dict]     # raw detections from Azure
    alignment_homography: Optional[np.ndarray] = None
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def _resize_for_api(img: np.ndarray, target: tuple[int, int] = API_IMAGE_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= target[0] and h <= target[1]:
        return img
    scale = min(target[0] / w, target[1] / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _ndarray_to_bytes(img: np.ndarray, quality: int = 85) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _scale_boxes_to_original(boxes: list[dict], orig_shape: tuple, api_shape: tuple) -> list[dict]:
    """Scale normalised or api-sized boxes back to original image dimensions."""
    oh, ow = orig_shape[:2]
    ah, aw = api_shape[:2]
    scaled = []
    for b in boxes:
        scaled.append({
            **b,
            "x1": int(b["x1"] * ow / aw),
            "y1": int(b["y1"] * oh / ah),
            "x2": int(b["x2"] * ow / aw),
            "y2": int(b["y2"] * oh / ah),
        })
    return scaled


# ── Camera alignment ──────────────────────────────────────────────────────────

class CameraAligner:
    """
    Aligns current frame to reference frame using ORB feature matching + homography.
    Falls back gracefully if not enough features are found.
    """

    def __init__(self, max_features: int = 5000, match_ratio: float = 0.75):
        self.detector = cv2.ORB_create(nfeatures=max_features)
        self.matcher  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_ratio = match_ratio

    def align(self, current: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Returns (aligned_current, homography_matrix).
        If alignment fails returns (current, None).
        """
        gray_cur = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

        kp_cur, des_cur = self.detector.detectAndCompute(gray_cur, None)
        kp_ref, des_ref = self.detector.detectAndCompute(gray_ref, None)

        if des_cur is None or des_ref is None or len(kp_cur) < 10 or len(kp_ref) < 10:
            logger.warning("Insufficient features for alignment — using original frame.")
            return current, None

        matches = self.matcher.knnMatch(des_cur, des_ref, k=2)
        good = [m for m, n in matches if m.distance < self.match_ratio * n.distance]

        if len(good) < 10:
            logger.warning("Too few good matches (%d) — skipping alignment.", len(good))
            return current, None

        src_pts = np.float32([kp_cur[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            logger.warning("Homography estimation failed — using original frame.")
            return current, None

        h, w = reference.shape[:2]
        aligned = cv2.warpPerspective(current, H, (w, h))
        logger.info("Alignment succeeded with %d inliers.", int(mask.sum()))
        return aligned, H


# ── Dynamic object detector (Azure Vision) ────────────────────────────────────

DETECTION_PROMPT = """
You are a construction site image analyser.

Identify ALL dynamic (moving) objects in this image: workers (people), cranes,
excavators, concrete trucks, forklifts, scaffolding workers, and any vehicles.

For each detected object return a JSON array with objects in this format:
{
  "label": "<object type>",
  "confidence": <0.0-1.0>,
  "x1": <left pixel>,
  "y1": <top pixel>,
  "x2": <right pixel>,
  "y2": <bottom pixel>
}

Image dimensions will be provided in the prompt.
If no dynamic objects are found return an empty array [].
Only output the JSON array.
"""


class DynamicObjectDetector:
    """Uses Azure Vision model to detect workers, cranes, and vehicles."""

    def __init__(self, client: AzureVisionClient):
        self.client = client

    def detect(self, image: np.ndarray) -> list[dict]:
        api_img = _resize_for_api(image)
        h, w = api_img.shape[:2]
        prompt = DETECTION_PROMPT + f"\nImage dimensions: width={w}, height={h} pixels."

        img_bytes = _ndarray_to_bytes(api_img)
        try:
            detections = self.client.ask_json(
                prompt=prompt,
                image_bytes_list=[img_bytes],
                max_tokens=1500,
            )
            if not isinstance(detections, list):
                detections = detections.get("objects", detections.get("detections", []))
        except Exception as e:
            logger.error("Object detection call failed: %s", e)
            return []

        # Scale back to original image size
        return _scale_boxes_to_original(detections, image.shape, api_img.shape)


# ── Inpainter ─────────────────────────────────────────────────────────────────

class RegionInpainter:
    """
    Fills dynamic-object regions using OpenCV inpainting (Navier-Stokes).
    For large occluded areas also queries Azure for a texture description
    to guide the fill (informational — OpenCV does the actual pixel work).
    """

    INPAINT_RADIUS = 15

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.max() == 0:
            return image
        # Dilate mask slightly to cover object edges
        kernel = np.ones((12, 12), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        result = cv2.inpaint(image, dilated, self.INPAINT_RADIUS, cv2.INPAINT_TELEA)
        return result


# ── Background model (temporal averaging) ─────────────────────────────────────

class TemporalBackgroundModel:
    """
    Maintains a running median of clean frames to produce a crowd-free background.
    Used as a fallback / supplement when detection misses something.
    """

    def __init__(self, max_frames: int = 20):
        self.max_frames = max_frames
        self._frames: list[np.ndarray] = []

    def update(self, clean_frame: np.ndarray):
        self._frames.append(clean_frame.astype(np.float32))
        if len(self._frames) > self.max_frames:
            self._frames.pop(0)

    def get_background(self) -> Optional[np.ndarray]:
        if not self._frames:
            return None
        stack = np.stack(self._frames, axis=0)
        median = np.median(stack, axis=0).astype(np.uint8)
        return median


# ── Main preprocessor ─────────────────────────────────────────────────────────

class ConstructionPreprocessor:
    """
    Orchestrates alignment → detection → masking → inpainting.
    Returns a clean static-scene image ready for progress analysis.
    """

    def __init__(self, client: AzureVisionClient):
        self.aligner   = CameraAligner()
        self.detector  = DynamicObjectDetector(client)
        self.inpainter = RegionInpainter()
        self.bg_model  = TemporalBackgroundModel()

    def process(
        self,
        current_path: str,
        reference_path: Optional[str] = None,
    ) -> PreprocessResult:
        current = _load_image(current_path)
        warnings = []
        H = None

        # Stage 1 — alignment
        if reference_path:
            reference = _load_image(reference_path)
            current, H = self.aligner.align(current, reference)
            if H is None:
                warnings.append("Camera alignment skipped — insufficient features.")
        else:
            warnings.append("No reference image provided — alignment skipped.")

        # Stage 2a — detect dynamic objects via Azure
        logger.info("Detecting dynamic objects with Azure Vision...")
        detections = self.detector.detect(current)
        logger.info("Detected %d dynamic objects.", len(detections))

        # Stage 2b — build mask
        h, w = current.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for det in detections:
            x1, y1 = max(0, det["x1"]), max(0, det["y1"])
            x2, y2 = min(w, det["x2"]), min(h, det["y2"])
            mask[y1:y2, x1:x2] = 255

        # Stage 2c — inpaint
        clean = self.inpainter.inpaint(current, mask)

        # Update background model
        self.bg_model.update(clean)

        # Confidence: lower if many/large dynamic regions
        masked_ratio = mask.sum() / 255 / (h * w)
        confidence = max(0.1, 1.0 - masked_ratio * 3)
        if masked_ratio > 0.3:
            warnings.append(
                f"Large dynamic area ({masked_ratio*100:.1f}% of frame) — "
                "consider using temporal background model."
            )

        return PreprocessResult(
            clean_image=clean,
            dynamic_mask=mask,
            detected_objects=detections,
            alignment_homography=H,
            confidence=confidence,
            warnings=warnings,
        )
