# Construction Progress Monitor

AI-powered construction site progress tracker using **Azure AI Foundry** (GPT-4o vision).

Analyses camera images to track % completion per structural zone by stripping moving objects (workers, cranes) and comparing against historical snapshots.

---

## Setup

```bash
pip install -r requirements.txt
```

### 1. Set your deployment name

Open [construction_progress/config.py](construction_progress/config.py) and set:

```python
DEPLOYMENT_NAME = "your-actual-deployment-name"
```

Find the name in: **Azure Portal → AI Foundry → hci-bbs-proj → Deployments**

### 2. Configure zones

Edit `DEFAULT_ZONES` in [config.py](construction_progress/config.py) to match your site layout.
Each zone needs:
- `id` — unique slug
- `name` — display name
- `polygon` — list of `[x_fraction, y_fraction]` points (0.0–1.0 of image size)
- `milestone_description` — construction sequence for scoring guidance

---

## Usage

### Analyse a single image
```bash
python -m construction_progress.main --image path/to/photo.jpg
```

### Specify camera and deployment
```bash
python -m construction_progress.main --image photo.jpg --camera cam_north --deployment my-gpt4o
```

### Watch a folder for new images (auto-process)
```bash
python -m construction_progress.main --watch C:/photos/cam_01 --camera cam_01 --interval 300
```

### Print current status from history
```bash
python -m construction_progress.main --status --camera cam_01
```

---

## Output (per run)

All outputs are saved to `construction_progress/data/output/<timestamp>/`:

| File | Description |
|------|-------------|
| `annotated.jpg` | Original image with zone overlays, % labels, colour coding |
| `report.json` | Full structured report — zones, scores, alerts, reasoning |
| `dashboard.html` | Self-contained HTML dashboard with timeline charts |

---

## Architecture

```
Camera image
  │
  ▼
[1] Camera Alignment     — ORB feature matching + homography warp
  │
  ▼
[2] Dynamic Object Removal — Azure Vision detects workers/cranes
                             OpenCV inpaints the masked regions
  │
  ▼
[3] Zone Analysis         — Crop each zone, extract CV features,
                             Azure Vision describes structural state,
                             diff against historical zone image
  │
  ▼
[4] Progress Scoring      — Azure Vision assigns % complete per zone
                             with confidence gating + regression guard
  │
  ▼
[5] Output                — Annotated image + JSON report + HTML dashboard
  │
  ▼
[6] History Store         — Snapshot saved for next run's reference
```

---

## Project Structure

```
construction_progress/
├── config.py                   ← API keys, zones, tuning parameters
├── main.py                     ← CLI entry point
├── core/
│   └── azure_client.py         ← Azure AI Foundry API wrapper
├── pipeline/
│   ├── preprocessor.py         ← Alignment + dynamic object removal
│   ├── zone_analyzer.py        ← Feature extraction + zone diff
│   ├── scorer.py               ← Progress % scoring engine
│   └── orchestrator.py         ← Wires all stages together
├── storage/
│   └── history_store.py        ← Snapshot persistence + image pair selection
├── output/
│   └── renderer.py             ← Annotated image + JSON + HTML dashboard
└── data/
    ├── history/                ← Per-camera snapshot history
    └── output/                 ← Run outputs
```

---

## Key design decisions

- **No YOLO / LaMa** — all AI inference goes through your Azure AI Foundry endpoint
- **GPT-4o vision** handles object detection, feature description, and progress scoring
- **OpenCV** handles geometric alignment (homography) and pixel inpainting — no external AI needed for these
- **Regression guard** — scores never drop unexpectedly without a flag
- **Confidence gating** — occluded or low-quality zones are flagged for human review instead of outputting a wrong number
