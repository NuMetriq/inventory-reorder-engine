from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data paths (all ignored by git)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "m5_raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Modeling defaults
STATE_ID = "WI"
FREQ = "W"
HORIZON = 1

QUANTILES = [0.5]  # add 0.1, 0.9 later
LEAD_TIME_WEEKS = [1, 2, 3, 4]

RANDOM_SEED = 42