"""
Stage 4: Progress Scoring Engine.

Uses Azure Vision to assign a % complete per zone by comparing
current clean image against:
  - Baseline (0% reference)
  - Historical progression
  - Milestone descriptions

Outputs a confidence-gated score per zone.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from construction_progress.core.azure_client import AzureVisionClient
from construction_progress.pipeline.zone_analyzer import ZoneDiff, ZoneFeatures
from construction_progress.pipeline.reference_matcher import ReferenceMatcher
from construction_progress.config import CONFIDENCE_THRESHOLD, DEFAULT_ZONES

logger = logging.getLogger(__name__)


@dataclass
class ZoneScore:
    zone_id: str
    zone_name: str
    progress_pct: float              # 0-100
    confidence: float                # 0-1
    needs_human_review: bool
    stage_label: str                 # e.g. "Rebar placement"
    reasoning: str                   # AI explanation
    stalled: bool = False            # True if no change vs previous score
    previous_progress_pct: float = 0.0
    delta_pct: float = 0.0          # progress since last run
    flags: list[str] = field(default_factory=list)


@dataclass
class SiteProgressReport:
    run_timestamp: str
    image_path: str
    overall_progress_pct: float
    zone_scores: list[ZoneScore]
    site_summary: str
    total_confidence: float
    alerts: list[str] = field(default_factory=list)


# ── Scoring prompts ───────────────────────────────────────────────────────────

PROGRESS_SCORING_PROMPT = """
You are an expert construction site progress analyst.

You are analysing a specific construction zone to estimate its % completion.

Zone name: {zone_name}
Milestone description: {milestone_description}
Detected visible elements: {visible_elements}
AI structural description: {structural_description}
Zone change score vs history: {change_score} (0=no change, 1=complete transformation)
New elements since last check: {new_elements}

Based on the zone crop image AND the above context, estimate:

{{
  "progress_pct": <integer 0-100>,
  "stage_label": "<current stage name, e.g. Rebar placement>",
  "confidence": <0.0-1.0, lower if occluded/unclear>,
  "reasoning": "<2-3 sentences explaining how you arrived at this percentage>",
  "low_confidence_reasons": ["<reason1>", ...],
  "stalled": <true if this zone appears to have no progress compared to history>
}}

Scoring guide:
- 0%:   Empty plot / site cleared only
- 10%:  Excavation started
- 20%:  Excavation complete / PCC blinding done
- 30%:  Rebar placement started
- 50%:  Rebar complete / formwork started
- 65%:  Formwork complete
- 75%:  Concrete poured (wet)
- 85%:  Concrete cured / deshuttering started
- 90%:  Structure complete, finishing started
- 100%: Zone fully complete per design
"""

SITE_SUMMARY_PROMPT = """
You are a construction project manager reviewing site progress.

Zone-by-zone progress summary:
{zone_summaries}

Overall weighted progress: {overall_pct}%

Write a concise 2-3 sentence executive summary of the site status,
highlight any stalled zones, and flag any concerns.
Return plain text (no JSON).
"""


REFERENCE_SCORING_PROMPT = """
You are a construction site expert validating a progress estimate.

Zone: {zone_name}
Milestone sequence: {milestone_description}
Visible elements detected: {visible_elements}
Structural description: {structural_description}

Reference image matching result:
  - Best matching reference: {matched_pct}%
  - Interpolated estimate: {interpolated_pct}%
  - Match confidence: {ref_confidence:.0%}
  - Closest references: {lower_pct}% (sim={lower_sim:.2f}) and {upper_pct}% (sim={upper_sim:.2f})

Based on the zone crop image AND the reference matching above, confirm or adjust the estimate.

{{
  "progress_pct": <integer 0-100, use the reference interpolated value unless you see a clear reason to adjust>,
  "stage_label": "<current stage name>",
  "confidence": <0.0-1.0>,
  "reasoning": "<1-2 sentences explaining the estimate>",
  "reference_used": true,
  "stalled": <true if clearly no change vs reference>
}}
"""


class ProgressScorer:
    """
    Scores each zone's % completion.
    Priority: reference image bank → AI vision scorer → fallback.
    """

    def __init__(self, client: AzureVisionClient, zones_config: list[dict]):
        self.client = client
        self.zones_config = {z["id"]: z for z in zones_config}
        self.ref_matcher = ReferenceMatcher()

    def score_zone(
        self,
        zone_diff: ZoneDiff,
        previous_score: Optional[ZoneScore] = None,
    ) -> ZoneScore:
        zone_id = zone_diff.zone_id
        zone_cfg = self.zones_config.get(zone_id, {})
        milestone_desc = zone_cfg.get("milestone_description", "Standard construction sequence")
        polygon_frac   = zone_cfg.get("polygon")
        current_feats  = zone_diff.current_features
        flags          = []

        logger.info("Scoring zone '%s'...", zone_diff.zone_name)

        # ── Path A: Reference image bank available ────────────────────────────
        ref_match = None
        if current_feats.crop_bytes:
            import numpy as np
            crop_arr = np.frombuffer(current_feats.crop_bytes, np.uint8)
            import cv2
            crop_img = cv2.imdecode(crop_arr, cv2.IMREAD_COLOR)
            if crop_img is not None:
                ref_match = self.ref_matcher.match(zone_id, crop_img, polygon_frac)

        if ref_match is not None:
            # Use reference match + ask AI to confirm/adjust with image context
            prompt = REFERENCE_SCORING_PROMPT.format(
                zone_name=zone_diff.zone_name,
                milestone_description=milestone_desc,
                visible_elements=", ".join(current_feats.visible_elements) or "none detected",
                structural_description=current_feats.structural_description,
                matched_pct=ref_match.matched_pct,
                interpolated_pct=ref_match.interpolated_pct,
                ref_confidence=ref_match.confidence,
                lower_pct=ref_match.lower_pct,
                lower_sim=ref_match.lower_similarity,
                upper_pct=ref_match.upper_pct,
                upper_sim=ref_match.upper_similarity,
            )
            flags.append(f"Reference bank used: best match={ref_match.matched_pct:.0f}%, interpolated={ref_match.interpolated_pct:.1f}%")
        else:
            # ── Path B: No reference images — pure AI scoring ─────────────────
            prompt = PROGRESS_SCORING_PROMPT.format(
                zone_name=zone_diff.zone_name,
                milestone_description=milestone_desc,
                visible_elements=", ".join(current_feats.visible_elements) or "none detected",
                structural_description=current_feats.structural_description,
                change_score=zone_diff.structural_change_score,
                new_elements=", ".join(zone_diff.new_elements) or "none",
            )

        image_bytes_list = [current_feats.crop_bytes] if current_feats.crop_bytes else None

        try:
            if image_bytes_list:
                result = self.client.ask_json(
                    prompt=prompt,
                    image_bytes_list=image_bytes_list,
                    max_tokens=800,
                )
            else:
                result = self.client.ask_json(prompt=prompt, max_tokens=800)
        except Exception as e:
            logger.error("Scoring failed for zone %s: %s", zone_id, e)
            flags.append(f"Scoring error: {e}")
            # Fall back to reference match value if we have it
            fallback_pct = ref_match.interpolated_pct if ref_match else (
                previous_score.progress_pct if previous_score else 0
            )
            result = {
                "progress_pct": fallback_pct,
                "stage_label": "Unknown",
                "confidence": ref_match.confidence if ref_match else 0.1,
                "reasoning": "AI scoring failed; used reference match" if ref_match else "Scoring call failed",
                "stalled": True,
            }

        progress_pct = float(result.get("progress_pct", 0))
        confidence   = float(result.get("confidence", 0.5))
        stalled      = bool(result.get("stalled", False))

        # Boost confidence when reference bank is in use
        if ref_match is not None:
            confidence = min(1.0, max(confidence, ref_match.confidence))

        # Override confidence if occlusion flagged
        if current_feats.occlusion_flag:
            confidence = min(confidence, 0.4)
            flags.append("Zone partially occluded by dynamic objects")

        # Low confidence reasons
        for reason in result.get("low_confidence_reasons", []):
            flags.append(reason)

        needs_review = confidence < CONFIDENCE_THRESHOLD or current_feats.occlusion_flag

        # Regression guard — don't go backwards unless explicitly flagged
        prev_pct = previous_score.progress_pct if previous_score else 0.0
        if progress_pct < prev_pct - 5 and not any("regression" in f.lower() for f in flags):
            logger.warning(
                "Zone %s: score dropped from %.1f to %.1f — clamping to previous.",
                zone_id, prev_pct, progress_pct
            )
            flags.append(f"Score drop detected ({prev_pct:.0f}→{progress_pct:.0f}%) — held at previous")
            progress_pct = prev_pct

        delta = progress_pct - prev_pct

        return ZoneScore(
            zone_id=zone_id,
            zone_name=zone_diff.zone_name,
            progress_pct=round(progress_pct, 1),
            confidence=round(confidence, 3),
            needs_human_review=needs_review,
            stage_label=result.get("stage_label", ""),
            reasoning=result.get("reasoning", ""),
            stalled=stalled,
            previous_progress_pct=prev_pct,
            delta_pct=round(delta, 1),
            flags=flags,
        )

    def compute_overall(self, zone_scores: list[ZoneScore]) -> float:
        """Weighted average — exclude low-confidence zones from overall."""
        valid = [z for z in zone_scores if z.zone_id != "overall_yard"]
        if not valid:
            return 0.0
        weights = [z.confidence for z in valid]
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        weighted_sum = sum(z.progress_pct * z.confidence for z in valid)
        return round(weighted_sum / total_w, 1)

    def generate_site_summary(self, zone_scores: list[ZoneScore], overall_pct: float) -> str:
        zone_summaries = "\n".join(
            f"- {z.zone_name}: {z.progress_pct:.0f}% ({z.stage_label})"
            + (f" [STALLED]" if z.stalled else "")
            + (f" [NEEDS REVIEW]" if z.needs_human_review else "")
            for z in zone_scores
        )
        prompt = SITE_SUMMARY_PROMPT.format(
            zone_summaries=zone_summaries,
            overall_pct=overall_pct,
        )
        try:
            return self.client.ask(prompt, max_tokens=300)
        except Exception as e:
            logger.error("Site summary generation failed: %s", e)
            return f"Site overall progress: {overall_pct:.0f}%. Summary generation failed."

    def build_alerts(self, zone_scores: list[ZoneScore], previous_scores: dict[str, ZoneScore]) -> list[str]:
        alerts = []
        for z in zone_scores:
            if z.stalled and previous_scores.get(z.zone_id):
                alerts.append(f"STALLED: {z.zone_name} has shown no progress.")
            if z.needs_human_review:
                alerts.append(f"REVIEW NEEDED: {z.zone_name} — confidence {z.confidence:.0%}. {'; '.join(z.flags)}")
            if z.delta_pct > 20:
                alerts.append(f"RAPID PROGRESS: {z.zone_name} jumped +{z.delta_pct:.0f}% — verify.")
        return alerts
