"""
Stage 5: Output generation.

Produces:
  1. Annotated image  — zone overlays with % labels + colour coding
  2. JSON report      — full structured output per run
  3. HTML dashboard   — timeline chart + heatmap (self-contained, no server needed)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from construction_progress.config import OUTPUT_DIR, DEFAULT_ZONES
from construction_progress.pipeline.scorer import ZoneScore, SiteProgressReport

logger = logging.getLogger(__name__)


# ── Colour helpers ────────────────────────────────────────────────────────────

def _progress_colour(pct: float) -> tuple[int, int, int]:
    """BGR colour: red=0%, yellow=50%, green=100%."""
    if pct < 50:
        r, g = 0, int(pct * 5.1)
    else:
        r, g = int((100 - pct) * 5.1), 255
    return (0, g, r)


def _alpha_blend(img: np.ndarray, overlay: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# ── Annotated image ───────────────────────────────────────────────────────────

class ImageAnnotator:
    """Draws zone polygons + progress labels on the clean scene image."""

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    THICKNESS  = 2

    def __init__(self, zones_config: list[dict] = None):
        self.zones = {z["id"]: z for z in (zones_config or DEFAULT_ZONES)}

    def annotate(
        self,
        clean_image: np.ndarray,
        zone_scores: list[ZoneScore],
        dynamic_mask: Optional[np.ndarray] = None,
        overall_pct: float = 0.0,
    ) -> np.ndarray:
        result = clean_image.copy()
        overlay = clean_image.copy()
        h, w = result.shape[:2]

        # Draw dynamic mask in transparent red
        if dynamic_mask is not None and dynamic_mask.max() > 0:
            red_layer = np.zeros_like(result)
            red_layer[dynamic_mask > 0] = (0, 0, 180)
            result = _alpha_blend(result, red_layer, alpha=0.25)

        # Draw each zone
        score_map = {z.zone_id: z for z in zone_scores}
        for zone_id, zone_cfg in self.zones.items():
            score = score_map.get(zone_id)
            if score is None:
                continue

            pts = np.array(
                [[int(x * w), int(y * h)] for x, y in zone_cfg["polygon"]],
                dtype=np.int32,
            )
            colour = _progress_colour(score.progress_pct)

            # Fill polygon
            zone_fill = overlay.copy()
            cv2.fillPoly(zone_fill, [pts], colour)
            result = _alpha_blend(result, zone_fill, alpha=0.25)

            # Border
            cv2.polylines(result, [pts], True, colour, 2)

            # Label background + text
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            label1 = zone_cfg["name"]
            label2 = f"{score.progress_pct:.0f}%"
            review_mark = " *" if score.needs_human_review else ""

            for i, txt in enumerate([label1, label2 + review_mark]):
                (tw, th), _ = cv2.getTextSize(txt, self.FONT, self.FONT_SCALE, self.THICKNESS)
                ty = cy - 10 + i * 22
                cv2.rectangle(result, (cx - tw // 2 - 4, ty - th - 2), (cx + tw // 2 + 4, ty + 2), (0, 0, 0), -1)
                cv2.putText(result, txt, (cx - tw // 2, ty), self.FONT, self.FONT_SCALE, (255, 255, 255), self.THICKNESS)

        # Overall progress banner
        banner = f"Overall Progress: {overall_pct:.0f}%   {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        cv2.rectangle(result, (0, 0), (w, 36), (30, 30, 30), -1)
        cv2.putText(result, banner, (10, 24), self.FONT, 0.7, (255, 255, 255), 2)

        # Legend
        for i, (label, pct) in enumerate([("0%", 0), ("50%", 50), ("100%", 100)]):
            col = _progress_colour(pct)
            lx = w - 160 + i * 50
            cv2.rectangle(result, (lx, h - 30), (lx + 30, h - 10), col, -1)
            cv2.putText(result, label, (lx, h - 35), self.FONT, 0.45, col, 1)

        return result


# ── JSON report ───────────────────────────────────────────────────────────────

def build_json_report(
    report: SiteProgressReport,
    image_path: str,
    annotated_image_path: str,
) -> dict:
    return {
        "metadata": {
            "run_timestamp": report.run_timestamp,
            "image_path": image_path,
            "annotated_image_path": annotated_image_path,
            "tool": "construction-progress-monitor",
            "version": "1.0.0",
        },
        "summary": {
            "overall_progress_pct": report.overall_progress_pct,
            "total_confidence": report.total_confidence,
            "site_summary": report.site_summary,
            "alerts": report.alerts,
        },
        "zones": [
            {
                "zone_id": z.zone_id,
                "zone_name": z.zone_name,
                "progress_pct": z.progress_pct,
                "delta_pct": z.delta_pct,
                "stage_label": z.stage_label,
                "confidence": z.confidence,
                "needs_human_review": z.needs_human_review,
                "stalled": z.stalled,
                "reasoning": z.reasoning,
                "flags": z.flags,
            }
            for z in report.zone_scores
        ],
    }


# ── HTML dashboard ────────────────────────────────────────────────────────────

def build_html_dashboard(
    report: dict,
    timeline: list[dict],
    output_path: str,
):
    """Generate a self-contained HTML dashboard with Chart.js."""
    zone_ids    = [z["zone_id"]   for z in report["zones"]]
    zone_names  = [z["zone_name"] for z in report["zones"]]
    zone_pcts   = [z["progress_pct"] for z in report["zones"]]
    zone_colors = [
        f"rgba({max(0,255-int(p*2.55))},{int(p*2.55)},0,0.8)"
        for p in zone_pcts
    ]

    # Timeline datasets per zone
    tl_labels   = [t["timestamp"][:16].replace("T", " ") for t in timeline]
    tl_datasets = []
    all_zone_ids = list({zid for t in timeline for zid in t.get("zones", {})})
    colours = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
    for i, zid in enumerate(all_zone_ids):
        tl_datasets.append({
            "label": zid,
            "data": [t.get("zones", {}).get(zid, None) for t in timeline],
            "borderColor": colours[i % len(colours)],
            "fill": False,
            "tension": 0.3,
        })

    alerts_html = "".join(
        f'<div class="alert">⚠ {a}</div>' for a in report["summary"]["alerts"]
    ) or "<div class='no-alert'>No alerts</div>"

    zone_rows = "".join(
        f"""<tr class="{'review' if z['needs_human_review'] else ''}">
            <td>{z['zone_name']}</td>
            <td>
              <div class="bar-bg"><div class="bar-fill" style="width:{z['progress_pct']}%;
                background:{zone_colors[i]}"></div></div>
              <span>{z['progress_pct']:.0f}%</span>
            </td>
            <td>{'▲ ' if z['delta_pct']>0 else ''}{z['delta_pct']:+.1f}%</td>
            <td>{z['stage_label']}</td>
            <td>{z['confidence']:.0%}</td>
            <td>{'🔴 STALLED' if z['stalled'] else ('⚠ REVIEW' if z['needs_human_review'] else '✅')}</td>
          </tr>"""
        for i, z in enumerate(report["zones"])
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Construction Progress Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body{{font-family:'Segoe UI',sans-serif;background:#f4f6f9;margin:0;color:#333}}
  header{{background:#1a3c5e;color:#fff;padding:18px 32px;display:flex;align-items:center;gap:16px}}
  header h1{{margin:0;font-size:1.4rem}}
  .overall{{font-size:2.5rem;font-weight:700;color:#2ecc71}}
  .container{{max-width:1200px;margin:24px auto;padding:0 16px}}
  .card{{background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.1);padding:24px;margin-bottom:24px}}
  .card h2{{margin-top:0;font-size:1.1rem;color:#1a3c5e;border-bottom:2px solid #e8ecf0;padding-bottom:8px}}
  table{{width:100%;border-collapse:collapse}}
  th{{background:#f0f4f8;text-align:left;padding:8px 12px;font-size:.85rem}}
  td{{padding:8px 12px;border-bottom:1px solid #eee;font-size:.9rem;vertical-align:middle}}
  tr.review td{{background:#fff8e1}}
  .bar-bg{{background:#e0e0e0;border-radius:4px;height:14px;width:180px;display:inline-block;vertical-align:middle}}
  .bar-fill{{height:14px;border-radius:4px;display:inline-block}}
  .alert{{background:#fff3cd;border-left:4px solid #ffc107;padding:8px 12px;margin:6px 0;border-radius:4px;font-size:.88rem}}
  .no-alert{{color:#27ae60;font-size:.9rem}}
  .summary-text{{color:#555;line-height:1.6;font-size:.95rem}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
  @media(max-width:768px){{.grid2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <div>
    <h1>🏗 Construction Progress Monitor</h1>
    <div style="font-size:.85rem;opacity:.8">Run: {report['metadata']['run_timestamp']}</div>
  </div>
  <div style="margin-left:auto;text-align:right">
    <div style="font-size:.85rem">Overall Progress</div>
    <div class="overall">{report['summary']['overall_progress_pct']:.0f}%</div>
  </div>
</header>
<div class="container">
  <div class="grid2">
    <div class="card">
      <h2>Site Summary</h2>
      <p class="summary-text">{report['summary']['site_summary']}</p>
    </div>
    <div class="card">
      <h2>Alerts</h2>
      {alerts_html}
    </div>
  </div>

  <div class="card">
    <h2>Zone Progress</h2>
    <table>
      <thead><tr><th>Zone</th><th>Progress</th><th>Delta</th><th>Stage</th><th>Confidence</th><th>Status</th></tr></thead>
      <tbody>{zone_rows}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Progress Timeline</h2>
    <canvas id="timelineChart" height="80"></canvas>
  </div>

  <div class="card">
    <h2>Current Zone Breakdown</h2>
    <canvas id="zoneChart" height="60"></canvas>
  </div>
</div>
<script>
const tlLabels = {json.dumps(tl_labels)};
const tlDatasets = {json.dumps(tl_datasets)};
new Chart(document.getElementById('timelineChart'), {{
  type: 'line',
  data: {{ labels: tlLabels, datasets: tlDatasets }},
  options: {{ responsive:true, scales:{{ y:{{ min:0, max:100, title:{{display:true,text:'% Complete'}} }} }} }}
}});
new Chart(document.getElementById('zoneChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(zone_names)},
    datasets: [{{
      label: 'Progress %',
      data: {json.dumps(zone_pcts)},
      backgroundColor: {json.dumps(zone_colors)},
    }}]
  }},
  options: {{ responsive:true, scales:{{ y:{{ min:0, max:100 }} }} }}
}});
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Dashboard written to %s", output_path)


# ── Output writer ─────────────────────────────────────────────────────────────

class OutputWriter:
    """Saves annotated image, JSON report, and HTML dashboard."""

    def __init__(self, output_dir: str = OUTPUT_DIR, zones_config: list[dict] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotator = ImageAnnotator(zones_config)

    def write(
        self,
        report: SiteProgressReport,
        clean_image: np.ndarray,
        dynamic_mask: Optional[np.ndarray],
        original_image_path: str,
        timeline: list[dict],
    ) -> dict:
        ts_slug = report.run_timestamp.replace(":", "").replace("-", "").replace(".", "")[:15]
        run_dir = self.output_dir / ts_slug
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Annotated image
        annotated = self.annotator.annotate(
            clean_image,
            report.zone_scores,
            dynamic_mask,
            report.overall_progress_pct,
        )
        ann_path = str(run_dir / "annotated.jpg")
        cv2.imwrite(ann_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # 2. JSON report
        json_report = build_json_report(report, original_image_path, ann_path)
        json_path = str(run_dir / "report.json")
        with open(json_path, "w") as f:
            json.dump(json_report, f, indent=2, default=str)

        # 3. HTML dashboard
        html_path = str(run_dir / "dashboard.html")
        build_html_dashboard(json_report, timeline, html_path)

        logger.info("Output written to %s", run_dir)
        return {
            "annotated_image": ann_path,
            "json_report": json_path,
            "dashboard": html_path,
        }
