"""
Central configuration — update DEPLOYMENT_NAME once you confirm it
from your Azure AI Foundry portal (AI Foundry > hci-bbs-proj > Deployments).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Azure AI Foundry ──────────────────────────────────────────────────────────
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
AZURE_API_KEY  = os.environ["AZURE_API_KEY"]

# Set this to the exact deployment name shown in your Azure portal
# e.g. "gpt-4o", "gpt-4o-deployment", "hci-vision-model", etc.
DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-5.2-hci-bbs")

# ── Pipeline tuning ───────────────────────────────────────────────────────────
# Minimum confidence score (0-1) below which a zone is flagged for human review
CONFIDENCE_THRESHOLD = 0.6

# How many history images to pull per zone for temporal comparison
MAX_HISTORY_FRAMES = 10

# Image resize for API calls (width, height) — keeps token cost manageable
API_IMAGE_SIZE = (1024, 1024)

# ── Storage paths ─────────────────────────────────────────────────────────────
import os
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR    = os.path.join(BASE_DIR, "data", "history")
ZONES_DIR      = os.path.join(BASE_DIR, "data", "zones")
OUTPUT_DIR     = os.path.join(BASE_DIR, "data", "output")
REFERENCE_DIR  = os.path.join(BASE_DIR, "data", "reference_images")

# ── Zone definitions (edit per site) ─────────────────────────────────────────
# Each zone: name, polygon (x,y) points as fraction of image width/height (0-1)
# Replace with your actual site zones
DEFAULT_ZONES = [
    {
        "id": "overall_yard",
        "name": "Overall Yard",
        "polygon": [
            [0.6648, 0.2368],
            [0.8414, 0.2333],
            [0.8496, 0.5326],
            [0.6434, 0.5361],
            [0.6359, 0.2431],
            [0.6703, 0.2382]
        ],
        "milestone_description": "Site clearing → Foundation → Structure → Finishing"
    },
    # {
    #     "id": "foundation_a",
    #     "name": "Foundation A",
    #     "polygon": [[0.0, 0.6], [0.35, 0.6], [0.35, 1.0], [0.0, 1.0]],
    #     "milestone_description": "Excavation → PCC → Rebar → Concrete pour → Curing"
    # },
    # {
    #     "id": "column_a",
    #     "name": "Column A",
    #     "polygon": [[0.1, 0.3], [0.25, 0.3], [0.25, 0.65], [0.1, 0.65]],
    #     "milestone_description": "Footing → Starter bar → Formwork → Concrete → Deshutter"
    # },
    # {
    #     "id": "slab_b",
    #     "name": "Slab B",
    #     "polygon": [[0.35, 0.4], [0.75, 0.4], [0.75, 0.75], [0.35, 0.75]],
    #     "milestone_description": "Shoring → Deck plate → Rebar mesh → Pour → Strip"
    # },
]
