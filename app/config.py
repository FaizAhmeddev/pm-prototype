import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "pm.db"
MODELS = ROOT / "models"
SOURCE = os.getenv("PM_SOURCE", "sim")          # "sim" = built-in simulator, anything else = external data via POST /api/ingest
TICK = float(os.getenv("PM_TICK", "2"))         # seconds between simulated readings
STAGES = ["Normal", "Early warning", "Maintenance required", "Critical"]
STAGE_LIMITS = [25, 50, 75]                      # health-score boundaries (illustrative, not OEM limits)

# Edit this list to configure monitored equipment. mode "degrade" ramps a fault from healthy to failed, then resets.
MACHINES = [
    {"tag": "C-101", "name": "Gas compressor", "plant": "Base Plant", "rpm": 1800, "mode": "degrade", "fault": "inner", "rate": 0.006},
    {"tag": "T-101", "name": "Turbo train", "plant": "EnVen", "rpm": 1800, "mode": "healthy"},
    {"tag": "P-201", "name": "Process pump", "plant": "Base Plant", "rpm": 1800, "mode": "healthy"},
    {"tag": "M-301", "name": "Electric motor", "plant": "EnVen", "rpm": 1800, "mode": "degrade", "fault": "imbalance", "rate": 0.003},
]
