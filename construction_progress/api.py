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

def _normalize_polygon(polygon: list[PolygonPoint], img_w: int, img_h: int) -> list[list[float]]:
    pts = [[p.x, p.y] for p in polygon]
    is_pixel = any(p[0] > 1.0 or p[1] > 1.0 for p in pts)
    if is_pixel:
        return [[round(p[0] / img_w, 6), round(p[1] / img_h, 6)] for p in pts]
    return [[round(p[0], 6), round(p[1], 6)] for p in pts]

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
        polygon_frac = _normalize_polygon(req.polygon, req.image_width or img_w, req.image_height or img_h)

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

        # Decode reference images (already downloaded above)
        ref_crops: list[tuple[float, np.ndarray]] = []
        for i, ref in enumerate(req.reference_images):
            ref_data = all_data[i + 1]
            if isinstance(ref_data, Exception):
                alerts.append(f"Reference {ref.progress_pct}% failed to load")
                logger.warning("Reference %.0f%% failed: %s", ref.progress_pct, ref_data)
                continue
            try:
                ref_img  = _bytes_to_cv2(ref_data)
                ref_crop = crop_zone(ref_img, polygon_frac)
                if ref_crop.size > 0:
                    ref_crops.append((ref.progress_pct, ref_crop))
            except Exception as e:
                alerts.append(f"Reference {ref.progress_pct}% failed to decode")
                logger.warning("Reference %.0f%% decode failed: %s", ref.progress_pct, e)

        if not ref_crops:
            raise HTTPException(400, "No reference images could be loaded")

        # 5. AI visual comparison — 3-vote majority for consistency
        client = get_client()
        ref_crops_sorted = sorted(ref_crops, key=lambda x: x[0])
        all_image_bytes  = [image_to_bytes(current_crop)] + [image_to_bytes(c) for _, c in ref_crops_sorted]

        ref_list_str = "\n".join(
            f"  Image {i+2}: Reference at {pct:.0f}% — look for structural elements present at this stage"
            for i, (pct, _) in enumerate(ref_crops_sorted)
        )

        ai_prompt = f"""You are a construction site progress analyst. You must give a PRECISE and CONSISTENT score.

You are given {len(ref_crops_sorted) + 1} images in order:
- Image 1: CURRENT SITE PHOTO — you must score this image
- Images 2 to {len(ref_crops_sorted) + 1}: REFERENCE photos at KNOWN completion percentages:
{ref_list_str}

Zone: {req.zone_name}

SCORING RULES (follow strictly):
1. Study Image 1 (current) carefully — count visible structural elements: rebar density, formwork panels, concrete pours, installed components.
2. Compare Image 1 against EACH reference image one by one.
3. Find the two references Image 1 falls BETWEEN based on physical structure visible.
4. If Image 1 matches reference X% exactly → score = X.
5. If Image 1 is between reference A% and reference B% → score = A + ((B-A) * how far between them).
6. NEVER guess. Base score ONLY on visible physical construction progress.
7. Ignore workers, vehicles, lighting, weather, shadows.
8. Rebar cage assembled = early stage (10-30%). Formwork = mid stage (40-60%). Concrete poured = late (70-100%).

Return ONLY this JSON:
{{
  "progress_pct": <integer 0-100, must match one of the reference % values or be between two of them>,
  "closest_reference_pct": <the single reference % Image 1 most closely matches>,
  "stage_label": "<one line: what construction stage is visible in Image 1>",
  "confidence": <0.0-1.0>,
  "reasoning": "<3 sentences: (1) what structural elements you see in Image 1, (2) which two references it falls between and why, (3) exact % chosen>"
}}"""

        logger.info("Calling Azure AI (3 parallel votes)...")

        async def _vote() -> dict:
            try:
                return await client.ask_json_async(
                    prompt=ai_prompt,
                    image_bytes_list=all_image_bytes,
                    max_tokens=700,
                    temperature=0.0,
                )
            except Exception as e:
                logger.warning("Vote failed: %s", e)
                return {}

        results = await asyncio.gather(_vote(), _vote(), _vote())

        votes: list[float] = []
        last_result: dict = {}
        for r in results:
            if r and "progress_pct" in r:
                votes.append(float(r["progress_pct"]))
                last_result = r

        if not votes:
            final_pct        = 0.0
            stage_label      = "Unknown"
            reasoning        = "All AI votes failed"
            final_confidence = 0.1
            method           = "failed"
            alerts.append("AI comparison failed — all 3 votes errored")
        else:
            votes_sorted = sorted(votes)
            final_pct        = float(votes_sorted[len(votes_sorted) // 2])
            stage_label      = last_result.get("stage_label", "")
            reasoning        = last_result.get("reasoning", "")
            final_confidence = round(float(last_result.get("confidence", 0.7)), 3)
            method           = f"ai_visual_comparison_votes={len(votes)}"
            logger.info("Votes %s -> median %.1f%%", votes, final_pct)

        # 6. Feature extraction for visible elements + structural description
        zone_cfg = {"id": req.zone_id, "name": req.zone_name, "polygon": polygon_frac, "milestone_description": req.milestone_description}
        analyzer = ZoneAnalyzer(client, [zone_cfg])
        feats    = analyzer.extract_features(clean_img, zone_cfg)

        needs_review = final_confidence < 0.6 or feats.occlusion_flag
        if feats.occlusion_flag:
            alerts.append("Zone partially occluded — result may be less accurate")

        # 7. Site summary
        try:
            site_summary = client.ask(
                f"Construction zone '{req.zone_name}' is at {final_pct:.1f}% ({stage_label}), confidence {final_confidence:.0%}. "
                f"Visible: {', '.join(feats.visible_elements) or 'none'}. {reasoning} "
                f"Write a 2-sentence executive summary. Plain text only.",
                max_tokens=120,
            )
        except Exception:
            site_summary = f"{req.zone_name} is at {final_pct:.1f}% completion — {stage_label}."

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
