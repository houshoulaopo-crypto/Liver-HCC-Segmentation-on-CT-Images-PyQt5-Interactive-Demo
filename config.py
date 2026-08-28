"""Project paths — all assets live inside this repo under models/."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "models"))

YOLO_WEIGHTS = MODEL_ROOT / "yolo" / "best.pt"
PREDICTIONS_DIR = MODEL_ROOT / "predictions" / "inferTs"
DEMO_CT_DIR = MODEL_ROOT / "demo" / "sample_ct"
NNUNET_MODEL_DIR = MODEL_ROOT / "nnunet" / "nnUNetTrainer__nnUNetPlans__3d_lowres"
NNUNET_CV_DIR = NNUNET_MODEL_DIR / "crossval_results_folds_0_1_2_3_4"
