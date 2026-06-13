# config.py
# Configuration for eeg-video-analysis-c9orf72

import os

# ── Data paths ─────────────────────────────────────────────────────────────
VIDEO_BASE_3M = r"F:\EEG\Baseline (24hrs)_3monthsold_before_KA"

GROUPS   = ["WT", "KO"]
CHANNELS = {"CA3": 0, "CTX": 1}
FS_DS    = 500    # downsample target Hz
EPOCH_S  = 4.0   # epoch duration seconds

# File size thresholds
ABF_MIN_BYTES = 100 * 1024 * 1024   # 100 MB (~1 hour at 5000 Hz)
WMV_MIN_BYTES = 400 * 1024 * 1024   # 400 MB (~1 hour video)

# ── Portfolio paths ────────────────────────────────────────────────────────
PORT_DIR    = r"C:\Users\belay\eeg-video-analysis-c9orf72"
DATA_DIR    = os.path.join(PORT_DIR, "data")
RESULTS_DIR = os.path.join(PORT_DIR, "results")
FIGURES_DIR = os.path.join(PORT_DIR, "figures")
SRC_9M      = r"C:\Users\belay\eeg-network-vulnerability-c9orf72-9m"

# ── Sleep classification parameters ───────────────────────────────────────
MIN_EPOCHS_PER_STATE = 10

# ── Video analysis parameters ──────────────────────────────────────────────
FRAME_SKIP    = 5      # process every Nth frame
RESIZE_FACTOR = 0.25   # resize frames to this fraction
MOV_THRESHOLD = 15     # pixel difference for movement detection

# ── Statistical parameters ─────────────────────────────────────────────────
ALPHA = 0.05
COLORS = {"WT": "#378ADD", "KO": "#D85A30"}

TP_ORDER  = ["3m", "4m", "6m", "9m", "12m"]
TP_LABELS = {
    "3m":  "Baseline\n3m",
    "4m":  "KA\n4m",
    "6m":  "Post-KA\n6m",
    "9m":  "Long-term\n9m",
    "12m": "Extended\n12m",
}

# ── Session ID convention ──────────────────────────────────────────────────
# animal_id = 5-digit folder prefix (e.g. "16572")
# session_id = animal_id + "1" or "2" (e.g. "165721", "165722")
# Scenario A: use session_id as statistical unit (n=36)
# Scenario B: use animal_id as statistical unit (n=18)
