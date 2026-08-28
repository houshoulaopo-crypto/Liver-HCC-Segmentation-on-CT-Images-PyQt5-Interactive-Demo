"""Project paths. Override MODEL_ROOT via environment variable."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Local model cache (default: ./models next to this repo)
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "models"))

# Hugging Face repo — replace YOUR_USERNAME after you create the HF repo
HF_REPO_ID = os.environ.get("HF_REPO_ID", "houshoulaopo-crypto/liver-hcc-models")

# Model file locations (must match huggingface-models/ layout)
YOLO_WEIGHTS = MODEL_ROOT / "yolo" / "best.pt"
PREDICTIONS_DIR = MODEL_ROOT / "predictions" / "inferTs"
NNUNET_CV_DIR = MODEL_ROOT / "nnunet" / "crossval_results_folds_0_1_2_3_4"
