"""
FastAPI — Construction Progress Monitor

POST /analyze
GET  /health
GET  /docs  (Swagger UI)
"""

import asyncio
import base64
import logging
import os
import tempfile
import warnings
from datetime import datetime, timezone
from typing import Optional

import cv2
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from construction_progress.core.azure_client import AzureVisionClient
from construction_progress.pipeline.zone_analyzer import ZoneAnalyzer, crop_zone, image_to_bytes
from construction_progress.pipeline.preprocessor import ConstructionPreprocessor
from construction_progress.pipeline.reference_matcher import _image_similarity
from construction_progress.config import DEPLOYMENT_NAME

warnings.filterwarnings("ignore")

# Only show our own logs — suppress httpx and uvicorn access noise
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("construction_progress")
logger.setLevel(logging.INFO)

app = FastAPI(title="Construction Progress Monitor API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_client: Optional[AzureVisionClient] = None

def get_client() -> AzureVisionClient:
    global _client
    if _client is None:
        _client = AzureVisionClient()
    return _client


# ── Models ────────────────────────────────────────────────────────────────────

class PolygonPoint(BaseModel):
    x: float = Field(..., description="X — pixel value or 0-1 fraction")
    y: float = Field(..., description="Y — pixel value or 0-1 fraction")

class ReferenceImage(BaseModel):
    url: str
    progress_pct: float = Field(..., description="Known % this image represents (0-100)")

class AnalyzeRequest(BaseModel):
    current_image_url: str
    reference_images: list[ReferenceImage]
    polygon: list[PolygonPoint]
    zone_name: str = "Zone"
    zone_id: str = "zone_01"
    milestone_description: str = "Standard construction sequence"
    remove_dynamic_objects: bool = False
    image_width: Optional[int] = None
    image_height: Optional[int] = None

class ZoneResult(BaseModel):
    zone_id: str
    zone_name: str
    progress_pct: float
    delta_pct: float
    stage_label: str
    confidence: float
    needs_human_review: bool
    stalled: bool
    reasoning: str
    flags: list[str]

class ReportSummary(BaseModel):
    overall_progress_pct: float
    total_confidence: float
    site_summary: str
    alerts: list[str]

class ReportMetadata(BaseModel):
    run_timestamp: str
    tool: str
    version: str
    method: str

class AnalyzeResponse(BaseModel):
    success: bool
    metadata: ReportMetadata
    summary: ReportSummary
    zones: list[ZoneResult]
    visible_elements: list[str]
    structural_description: str
    processed_image_base64: str
    azure_raw_response: Optional[dict] = Field(default=None, description="Raw Azure AI server response for the winning (median) scoring vote")
    error: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download(url: str) -> bytes:
    with httpx.Client(verify=False, timeout=60, follow_redirects=True) as c:
        r = c.get(url)
        r.raise_for_status()
        return r.content

def _bytes_to_cv2(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def _align_image_to_reference(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Align src image to match ref image using ORB feature matching + homography.
    Handles camera position shifts, slight rotations, zoom changes.

    Returns (aligned_image, success).
    Falls back to original src if alignment fails (not enough features / bad homography).
    """
    try:
        # Resize both to same size for feature matching
        h, w = ref.shape[:2]
        src_resized = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

        gray_ref = cv2.cvtColor(ref,         cv2.COLOR_BGR2GRAY)
        gray_src = cv2.cvtColor(src_resized, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)
        kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
        kp_src, des_src = orb.detectAndCompute(gray_src, None)

        if des_ref is None or des_src is None or len(kp_ref) < 10 or len(kp_src) < 10:
            return src_resized, False

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(des_src, des_ref, k=2)

        # Lowe ratio test — keep only strong matches
        good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

        if len(good) < 10:
            logger.warning("Alignment: only %d good matches — skipping alignment", len(good))
            return src_resized, False

        pts_src = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 5.0)

        if H is None:
            return src_resized, False

        # Reject bad homography — large determinant means extreme distortion
        det = abs(np.linalg.det(H[:2, :2]))
        if det < 0.1 or det > 10.0:
            logger.warning("Alignment: bad homography determinant=%.3f — skipping", det)
            return src_resized, False

        aligned = cv2.warpPerspective(src_resized, H, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        inlier_ratio = mask.ravel().sum() / len(good)
        logger.info("Alignment: %d good matches, inlier_ratio=%.2f, det=%.3f",
                    len(good), inlier_ratio, det)
        return aligned, True

    except Exception as e:
        logger.warning("Alignment failed: %s", e)
        return src, False


def _normalize_polygon(polygon: list[PolygonPoint], img_w: int, img_h: int) -> list[list[float]]:
    """
    Convert polygon points to 0-1 fractions relative to img_w / img_h.

    Handles all three formats callers may send:
      - Pixel coords  (x > 1 or y > 1)  → divide by image dimensions
      - Fraction coords (all 0-1)        → use as-is
      - Mixed                            → treat as pixel (safer — dividing a 0-1
                                           value by 1920 produces a negligible crop,
                                           which is caught by the empty-crop check)

    Always clamps output to [0, 1] so out-of-bounds points don't crash cv2.
    """
    pts = [[p.x, p.y] for p in polygon]
    is_pixel = any(p[0] > 1.0 or p[1] > 1.0 for p in pts)
    if is_pixel:
        result = [[p[0] / img_w, p[1] / img_h] for p in pts]
    else:
        result = [[p[0], p[1]] for p in pts]
    # Clamp to [0,1] — guards against polygons drawn slightly outside image bounds
    return [[round(min(1.0, max(0.0, x)), 6), round(min(1.0, max(0.0, y)), 6)] for x, y in result]

def _draw_zone(
    image: np.ndarray,
    polygon_frac: list[list[float]],
    zone_name: str,
    progress_pct: float,
    confidence: float,
    stage_label: str,
) -> np.ndarray:
    result = image.copy()
    h, w = result.shape[:2]
    pts = np.array([[int(x * w), int(y * h)] for x, y in polygon_frac], dtype=np.int32)

    # Colour: red → yellow → green
    if progress_pct < 50:
        colour = (0, int(progress_pct * 5.1), 220)
    else:
        colour = (0, 255, int((100 - progress_pct) * 5.1))

    overlay = result.copy()
    cv2.fillPoly(overlay, [pts], colour)
    result = cv2.addWeighted(overlay, 0.28, result, 0.72, 0)
    cv2.polylines(result, [pts], True, colour, 3)

    cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, txt in enumerate([zone_name, f"{progress_pct:.1f}%  |  {stage_label}", f"Confidence: {confidence:.0%}"]):
        (tw, th), _ = cv2.getTextSize(txt, font, 0.62, 2)
        ty = cy - 28 + i * 26
        cv2.rectangle(result, (cx - tw//2 - 5, ty - th - 3), (cx + tw//2 + 5, ty + 3), (0, 0, 0), -1)
        cv2.putText(result, txt, (cx - tw//2, ty), font, 0.62, (255, 255, 255), 2)

    # Top banner
    cv2.rectangle(result, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.putText(result, f"Construction Progress  |  {zone_name}: {progress_pct:.1f}%", (10, 24), font, 0.68, (255, 255, 255), 2)

    return result

def _to_base64(image: np.ndarray, quality: int = 88) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _validate_summary_pct(summary: str, final_pct: float, tolerance: float = 5.0) -> bool:
    """
    True only if every percentage mentioned in the AI-written summary is within
    `tolerance` of final_pct. The summary prompt also feeds the model reasoning
    text that references *reference* percentages (e.g. "closest to the 50%
    reference"), which the model can echo as if it were the actual score —
    catch that here rather than trusting free-text generation to self-report
    the number correctly.
    """
    import re
    mentioned = [float(m) for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", summary)]
    return all(abs(pct - final_pct) <= tolerance for pct in mentioned)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    alerts = []

    try:
        # 1 + 4. Download current image AND all reference images in parallel
        logger.info("Downloading current + %d reference images in parallel...", len(req.reference_images))

        async def _fetch(url: str) -> bytes:
            async with httpx.AsyncClient(verify=False, timeout=60, follow_redirects=True) as ac:
                r = await ac.get(url)
                r.raise_for_status()
                return r.content

        all_urls = [req.current_image_url] + [r.url for r in req.reference_images]
        all_data = await asyncio.gather(*[_fetch(u) for u in all_urls], return_exceptions=True)

        # Decode current image
        current_data = all_data[0]
        if isinstance(current_data, Exception):
            raise HTTPException(400, f"Failed to download current image: {current_data}")
        try:
            current_img = _bytes_to_cv2(current_data)
        except Exception as e:
            raise HTTPException(400, f"Failed to decode current image: {e}")

        img_h, img_w = current_img.shape[:2]
        # Use caller-supplied dimensions if provided (more reliable than decoded image size)
        # This is the canonical resolution — all polygon normalization is relative to this
        canonical_w = req.image_width  or img_w
        canonical_h = req.image_height or img_h
        polygon_frac = _normalize_polygon(req.polygon, canonical_w, canonical_h)
        logger.info("Polygon normalized: canonical=%dx%d actual_current=%dx%d polygon_is_pixel=%s",
                    canonical_w, canonical_h, img_w, img_h,
                    any(p.x > 1.0 or p.y > 1.0 for p in req.polygon))

        # 2. Optional dynamic object removal
        clean_img = current_img
        if req.remove_dynamic_objects:
            logger.info("Removing dynamic objects...")
            try:
                client = get_client()
                preprocessor = ConstructionPreprocessor(client)
                tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tf.close()
                cv2.imwrite(tf.name, current_img)
                prep = preprocessor.process(tf.name)
                clean_img = prep.clean_image
                os.unlink(tf.name)
                if prep.detected_objects:
                    alerts.append(f"Removed {len(prep.detected_objects)} dynamic objects")
            except Exception as e:
                alerts.append(f"Dynamic object removal failed — using original image")

        # 3. Crop zone from current image
        current_crop = crop_zone(clean_img, polygon_frac)
        if current_crop.size == 0:
            raise HTTPException(400, "Polygon crop is empty — check coordinates")

        # Decode + align reference images to current image before cropping.
        # Alignment compensates for camera position shifts, slight pans/tilts/zooms
        # that happen over time on site cameras. Without alignment, a shifted camera
        # means the polygon crops a different physical area in old vs new images,
        # causing the AI to compare unrelated zones and produce wrong scores.
        ref_crops: list[tuple[float, np.ndarray]] = []
        align_succeeded = 0
        align_failed    = 0

        for i, ref in enumerate(req.reference_images):
            ref_data = all_data[i + 1]
            if isinstance(ref_data, Exception):
                alerts.append(f"Reference {ref.progress_pct}% failed to load")
                logger.warning("Reference %.0f%% failed: %s", ref.progress_pct, ref_data)
                continue
            try:
                ref_img = _bytes_to_cv2(ref_data)
                ref_h, ref_w = ref_img.shape[:2]

                if ref_w != canonical_w or ref_h != canonical_h:
                    logger.warning(
                        "Reference %.0f%%: resolution mismatch current=%dx%d ref=%dx%d",
                        ref.progress_pct, canonical_w, canonical_h, ref_w, ref_h
                    )

                # Step 1 — Align reference image to current image coordinate space.
                # This warps the reference so its features line up with the current image,
                # compensating for any camera drift since that reference was taken.
                # After alignment the same polygon_frac crops the same physical area
                # in both current and reference images.
                ref_aligned, did_align = _align_image_to_reference(ref_img, clean_img)
                if did_align:
                    align_succeeded += 1
                else:
                    align_failed += 1
                    # Alignment failed — fall back to per-image polygon normalization
                    # which at least handles resolution differences
                    ref_aligned = cv2.resize(ref_img, (canonical_w, canonical_h),
                                             interpolation=cv2.INTER_AREA)

                # Step 2 — Crop using the canonical polygon (already in fraction coords
                # relative to current image, which reference is now aligned to)
                ref_crop = crop_zone(ref_aligned, polygon_frac)

                if ref_crop.size > 0:
                    ref_crops.append((ref.progress_pct, ref_crop))
                    logger.info("Reference %.0f%%: aligned=%s crop=%dx%d",
                                ref.progress_pct, did_align,
                                ref_crop.shape[1], ref_crop.shape[0])
                else:
                    alerts.append(f"Reference {ref.progress_pct}% crop was empty after alignment")
                    logger.warning("Reference %.0f%%: empty crop after alignment", ref.progress_pct)

            except Exception as e:
                alerts.append(f"Reference {ref.progress_pct}% failed to decode")
                logger.warning("Reference %.0f%% decode failed: %s", ref.progress_pct, e)

        if align_succeeded > 0:
            logger.info("Alignment summary: %d succeeded, %d fell back", align_succeeded, align_failed)
        if align_failed > 0 and align_succeeded == 0:
            alerts.append(f"Camera alignment failed for all references — results may be less accurate if camera has shifted")

        if not ref_crops:
            raise HTTPException(400, "No reference images could be loaded")

        # 5. Rank references by structural similarity to the current crop — the
        # caller-supplied progress_pct labels are manually entered and can be
        # wrong or drift over time, so label order is never trusted as ground
        # truth. Similarity rank (0=most alike, N=least alike) is the actual
        # signal driving comparison order; the label is shown only as extra
        # context for the AI, not as the primary ordering.
        ref_ranked = sorted(
            (( _image_similarity(current_crop, crop), pct, crop) for pct, crop in ref_crops),
            key=lambda x: x[0], reverse=True,
        )

        # Flag references whose label ordering contradicts their similarity ordering —
        # e.g. a reference labelled with a low % that is structurally closer to current
        # than references labelled with higher %, which suggests a mislabeled entry.
        by_label = sorted(ref_crops, key=lambda x: x[0])
        sim_by_pct = {pct: sim for sim, pct, _ in ref_ranked}
        for i in range(len(by_label) - 1):
            pct_a, _ = by_label[i]
            pct_b, _ = by_label[i + 1]
            if pct_a == pct_b:
                continue
            if sim_by_pct[pct_a] > sim_by_pct[pct_b] + 0.15:
                alerts.append(
                    f"Reference labels may be inconsistent: {pct_a:.0f}% is structurally more similar "
                    f"to current than {pct_b:.0f}% — check reference data"
                )

        client = get_client()
        all_image_bytes = [image_to_bytes(current_crop)] + [image_to_bytes(crop) for _, _, crop in ref_ranked]

        ref_list_str = "\n".join(
            f"  Image {i+2}: labelled {pct:.0f}% (structural similarity rank {i+1} of {len(ref_ranked)}, most-similar first)"
            for i, (sim, pct, _) in enumerate(ref_ranked)
        )

        ai_prompt = f"""You are a senior construction site progress analyst. Your job is to score Image 1 (current site) by comparing its PHYSICAL STRUCTURES against the reference images.

IMAGE ORDER (sorted by structural similarity to Image 1, most similar first — NOT by label):
- Image 1: CURRENT SITE PHOTO — score this image
- Images 2 to {len(ref_ranked) + 1}: REFERENCE photos, each with a manually-entered label (labels may be wrong or inconsistent — treat them as a hint, not ground truth):
{ref_list_str}

Zone: {req.zone_name}

CRITICAL INSTRUCTIONS:
1. Look ONLY at permanent structural elements in each image: steel members, rebar, formwork panels, concrete, installed components, completed assemblies.
2. DO NOT be influenced by: lighting, time of day, weather, shadows, dust, workers, vehicles, camera angle differences, image brightness.
3. Compare the QUANTITY and COMPLETENESS of structural elements in Image 1 vs each reference.
4. Ask yourself for each reference: "Does Image 1 have MORE structure built than this reference, LESS, or the SAME?"
5. Find the two references (by actual structural completeness, not by their labels) that bracket Image 1.
6. IMPORTANT: reference labels are manually entered and may be wrong — if a label contradicts what you visually observe (e.g. a "10%" reference clearly shows more built structure than a "50%" reference), trust your own structural observation over the label.
7. Judge by WHAT IS BUILT, not overall visual similarity (camera/lighting can make unrelated stages look alike).
8. Your final score should be consistent with the actual amount of structure visible in Image 1, bounded by the labels of whichever references are genuinely less/more built — not by label order alone.

STEP BY STEP:
Step 1 — Describe Image 1: What permanent structures are visible? How complete are they?
Step 2 — For each reference, judged purely by what's built (ignore its label for this step): does Image 1 have more or less built?
Step 3 — Identify the two references — by actual structural completeness — Image 1 falls between.
Step 4 — Score based on how far between those two references' labels Image 1 sits structurally. If labels seem inconsistent with what's built, note this in your reasoning and pick the score that best reflects genuine structural progress.

Return ONLY this JSON (no extra text):
{{
  "progress_pct": <integer 0-100>,
  "closest_reference_pct": <the reference % whose structural completion most closely matches Image 1>,
  "lower_bound_pct": <reference % just below current progress>,
  "upper_bound_pct": <reference % just above current progress>,
  "stage_label": "<one line describing the construction stage visible in Image 1>",
  "confidence": <0.0-1.0>,
  "reasoning": "<4 sentences: (1) permanent structures visible in Image 1, (2) which reference has less built and why, (3) which reference has more built and why, (4) exact % chosen and why>"
}}"""

        logger.info("Calling Azure AI (3 parallel votes)...")

        async def _vote() -> tuple[dict, Optional[dict]]:
            try:
                parsed, raw = await client.ask_json_async(
                    prompt=ai_prompt,
                    image_bytes_list=all_image_bytes,
                    max_tokens=700,
                    temperature=0.0,
                    return_raw=True,
                )
                return parsed, raw
            except Exception as e:
                logger.warning("Vote failed: %s", e)
                return {}, None

        results = await asyncio.gather(_vote(), _vote(), _vote())

        votes: list[tuple[float, dict, Optional[dict]]] = []
        for parsed, raw in results:
            if parsed and "progress_pct" in parsed:
                votes.append((float(parsed["progress_pct"]), parsed, raw))

        azure_raw_response: Optional[dict] = None

        if not votes:
            final_pct        = 0.0
            stage_label      = "Unknown"
            reasoning        = "All AI votes failed"
            final_confidence = 0.1
            method           = "failed"
            alerts.append("AI comparison failed — all 3 votes errored")
        else:
            votes_sorted = sorted(votes, key=lambda v: v[0])
            median_pct, last_result, azure_raw_response = votes_sorted[len(votes_sorted) // 2]
            final_pct        = median_pct
            stage_label      = last_result.get("stage_label", "")
            reasoning        = last_result.get("reasoning", "")
            final_confidence = round(float(last_result.get("confidence", 0.7)), 3)
            method           = f"ai_visual_comparison_votes={len(votes)}"
            logger.info("Votes %s -> median %.1f%%", [v[0] for v in votes], final_pct)

        # 6. Feature extraction for visible elements + structural description
        zone_cfg = {"id": req.zone_id, "name": req.zone_name, "polygon": polygon_frac, "milestone_description": req.milestone_description}
        analyzer = ZoneAnalyzer(client, [zone_cfg])
        feats    = analyzer.extract_features(clean_img, zone_cfg)

        needs_review = final_confidence < 0.6 or feats.occlusion_flag
        if feats.occlusion_flag:
            alerts.append("Zone partially occluded — result may be less accurate")

        # 7. Site summary
        fallback_summary = f"{req.zone_name} is at {final_pct:.1f}% completion — {stage_label}."
        try:
            site_summary = client.ask(
                f"Construction zone '{req.zone_name}' is at {final_pct:.1f}% ({stage_label}), confidence {final_confidence:.0%}. "
                f"Visible: {', '.join(feats.visible_elements) or 'none'}. {reasoning} "
                f"Write a 2-sentence executive summary. Plain text only. "
                f"The completion percentage MUST be stated as exactly {final_pct:.1f}% — do not restate any other percentage from the reasoning above.",
                max_tokens=120,
            )
            if not _validate_summary_pct(site_summary, final_pct):
                logger.warning(
                    "Site summary mentioned a percentage inconsistent with final_pct=%.1f — using fallback. Got: %s",
                    final_pct, site_summary,
                )
                alerts.append("AI summary text was inconsistent with the score — used fallback summary")
                site_summary = fallback_summary
        except Exception:
            site_summary = fallback_summary

        # 8. Annotated image with polygon overlay
        annotated = _draw_zone(clean_img, polygon_frac, req.zone_name, final_pct, final_confidence, stage_label)
        processed_image_b64 = _to_base64(annotated)

        run_ts     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        zone_flags = []
        if needs_review:
            zone_flags.append(f"Confidence {final_confidence:.0%} — human review recommended")
        if feats.occlusion_flag:
            zone_flags.append("Zone partially occluded")

        return AnalyzeResponse(
            success=True,
            metadata=ReportMetadata(run_timestamp=run_ts, tool="construction-progress-monitor", version="1.0.0", method=method),
            summary=ReportSummary(overall_progress_pct=final_pct, total_confidence=final_confidence, site_summary=site_summary, alerts=alerts),
            zones=[ZoneResult(
                zone_id=req.zone_id, zone_name=req.zone_name,
                progress_pct=final_pct, delta_pct=0.0,
                stage_label=stage_label, confidence=final_confidence,
                needs_human_review=needs_review, stalled=False,
                reasoning=reasoning, flags=zone_flags,
            )],
            visible_elements=feats.visible_elements,
            structural_description=feats.structural_description,
            azure_raw_response=azure_raw_response,
            processed_image_base64=processed_image_b64,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error")
        raise HTTPException(500, f"Internal error: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "deployment": DEPLOYMENT_NAME}

@app.get("/")
def root():
    return {"service": "Construction Progress Monitor API", "docs": "/docs", "analyze": "POST /analyze"}

if __name__ == "__main__":
    uvicorn.run("construction_progress.api:app", host="0.0.0.0", port=5001, reload=False, log_level="warning")
