"""
Main pipeline orchestrator — wires all stages together.

Flow:
  Input image
    → History Store  (fetch reference + previous features/scores)
    → Preprocessor   (align + remove workers/cranes via Azure)
    → ZoneAnalyzer   (extract features + diff vs history)
    → ProgressScorer (score each zone via Azure)
    → OutputWriter   (annotated image + JSON + dashboard)
    → History Store  (save snapshot)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from construction_progress.config import DEFAULT_ZONES, CONFIDENCE_THRESHOLD
from construction_progress.core.azure_client import AzureVisionClient
from construction_progress.pipeline.preprocessor import ConstructionPreprocessor
from construction_progress.pipeline.zone_analyzer import ZoneAnalyzer, ZoneFeatures
from construction_progress.pipeline.scorer import ProgressScorer, SiteProgressReport, ZoneScore
from construction_progress.storage.history_store import HistoryStore
from construction_progress.output.renderer import OutputWriter

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool
    report: Optional[SiteProgressReport]
    output_paths: dict          # annotated_image, json_report, dashboard
    error: Optional[str] = None


class ConstructionProgressPipeline:
    """
    Single entry point for one analysis run.
    Instantiate once, call run() for each new image.
    """

    def __init__(
        self,
        camera_id: str = "cam_01",
        zones_config: list[dict] = None,
        deployment_name: Optional[str] = None,
    ):
        self.camera_id    = camera_id
        self.zones_config = zones_config or DEFAULT_ZONES

        # Override deployment at runtime if needed
        if deployment_name:
            from construction_progress import config
            config.DEPLOYMENT_NAME = deployment_name

        self.client      = AzureVisionClient()
        self.preprocessor = ConstructionPreprocessor(self.client)
        self.analyzer    = ZoneAnalyzer(self.client, self.zones_config)
        self.scorer      = ProgressScorer(self.client, self.zones_config)
        self.store       = HistoryStore()
        self.writer      = OutputWriter(zones_config=self.zones_config)

    def run(self, image_path: str) -> PipelineResult:
        logger.info("=== Pipeline run started: %s ===", image_path)
        ts = datetime.utcnow().isoformat() + "Z"

        try:
            # ── Stage 1+2: Preprocessing ──────────────────────────────────────
            logger.info("[1/5] Preprocessing...")
            import cv2
            current_raw = cv2.imread(image_path)
            if current_raw is None:
                return PipelineResult(False, None, {}, f"Cannot load image: {image_path}")

            reference_img = self.store.select_best_reference(self.camera_id, current_raw)
            import tempfile, os
            ref_path = None
            if reference_img is not None:
                tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                tf.close()  # close before writing — avoids Windows file lock
                cv2.imwrite(tf.name, reference_img)
                ref_path = tf.name

            prep_result = self.preprocessor.process(image_path, ref_path)
            if ref_path and os.path.exists(ref_path):
                try:
                    os.unlink(ref_path)
                except OSError:
                    pass  # non-fatal on Windows

            if prep_result.warnings:
                for w in prep_result.warnings:
                    logger.warning("Preprocessor: %s", w)

            # ── Stage 3: Zone analysis ────────────────────────────────────────
            logger.info("[2/5] Analysing zones...")
            history_features_raw = self.store.get_latest_zone_features(self.camera_id)

            # Reconstruct ZoneFeatures objects from stored dicts
            history_features: dict[str, ZoneFeatures] = {}
            for zid, fd in history_features_raw.items():
                history_features[zid] = ZoneFeatures(
                    zone_id=fd.get("zone_id", zid),
                    zone_name=fd.get("zone_name", zid),
                    edge_density=fd.get("edge_density", 0),
                    texture_variance=fd.get("texture_variance", 0),
                    brightness=fd.get("brightness", 0),
                    structural_description=fd.get("structural_description", ""),
                    visible_elements=fd.get("visible_elements", []),
                    occlusion_flag=fd.get("occlusion_flag", False),
                    crop_bytes=None,  # not stored, re-cropped fresh
                )

            zone_diffs = self.analyzer.analyse_all_zones(
                prep_result.clean_image,
                history_features,
            )

            # ── Stage 4: Scoring ──────────────────────────────────────────────
            logger.info("[3/5] Scoring progress...")
            prev_scores_raw = self.store.get_latest_zone_scores(self.camera_id)
            prev_scores: dict[str, ZoneScore] = {}
            for zid, sd in prev_scores_raw.items():
                prev_scores[zid] = ZoneScore(
                    zone_id=sd.get("zone_id", zid),
                    zone_name=sd.get("zone_name", zid),
                    progress_pct=sd.get("progress_pct", 0),
                    confidence=sd.get("confidence", 0),
                    needs_human_review=sd.get("needs_human_review", False),
                    stage_label=sd.get("stage_label", ""),
                    reasoning=sd.get("reasoning", ""),
                    stalled=sd.get("stalled", False),
                    previous_progress_pct=sd.get("previous_progress_pct", 0),
                    delta_pct=sd.get("delta_pct", 0),
                    flags=sd.get("flags", []),
                )

            zone_scores: list[ZoneScore] = []
            for diff in zone_diffs:
                score = self.scorer.score_zone(diff, prev_scores.get(diff.zone_id))
                zone_scores.append(score)

            overall_pct    = self.scorer.compute_overall(zone_scores)
            site_summary   = self.scorer.generate_site_summary(zone_scores, overall_pct)
            alerts         = self.scorer.build_alerts(zone_scores, prev_scores)
            total_conf     = sum(z.confidence for z in zone_scores) / max(1, len(zone_scores))

            report = SiteProgressReport(
                run_timestamp=ts,
                image_path=image_path,
                overall_progress_pct=overall_pct,
                zone_scores=zone_scores,
                site_summary=site_summary,
                total_confidence=round(total_conf, 3),
                alerts=alerts,
            )

            # ── Stage 5: Output ───────────────────────────────────────────────
            logger.info("[4/5] Writing outputs...")
            timeline = self.store.get_progress_timeline(self.camera_id)
            output_paths = self.writer.write(
                report,
                prep_result.clean_image,
                prep_result.dynamic_mask,
                image_path,
                timeline,
            )

            # ── Save to history ───────────────────────────────────────────────
            logger.info("[5/5] Saving snapshot...")
            feat_map  = {d.zone_id: d.current_features for d in zone_diffs}
            score_map = {s.zone_id: s for s in zone_scores}

            import json
            from dataclasses import asdict
            report_dict = {
                "run_timestamp": ts,
                "overall_progress_pct": overall_pct,
                "site_summary": site_summary,
                "alerts": alerts,
                "zones": [asdict(s) for s in zone_scores],
            }

            self.store.save_snapshot(
                camera_id=self.camera_id,
                image=prep_result.clean_image,
                zone_features=feat_map,
                zone_scores=score_map,
                overall_progress=overall_pct,
                report=report_dict,
            )

            logger.info("=== Run complete. Overall: %.1f%% ===", overall_pct)
            return PipelineResult(True, report, output_paths)

        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            return PipelineResult(False, None, {}, str(e))

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
