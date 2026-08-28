"""Copy model weights and demo data from the original BME project into models/."""
from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "models"
BME_ROOT = PROJECT_ROOT.parent

COPY_RULES: list[tuple[Path, Path]] = [
    (
        BME_ROOT / "ID7200 segmentation result" / "YOLO" / "train4" / "weights" / "best.pt",
        MODEL_ROOT / "yolo" / "best.pt",
    ),
    (
        BME_ROOT / "ID7200 segmentation result" / "YOLO" / "liver_tumor_detect" / "weights" / "best.pt",
        MODEL_ROOT / "yolo" / "best.pt",
    ),
    (
        BME_ROOT / "ID7200 segmentation result" / "liver_tumor_segmentation_testset50" / "inferTs",
        MODEL_ROOT / "predictions" / "inferTs",
    ),
    (
        BME_ROOT
        / "ID7200 segmentation result"
        / "nnUNet-master"
        / "DATASET"
        / "nnUNet_trained_models"
        / "Dataset001_Liver"
        / "nnUNetTrainer__nnUNetPlans__3d_lowres"
        / "crossval_results_folds_0_1_2_3_4",
        MODEL_ROOT / "nnunet" / "nnUNetTrainer__nnUNetPlans__3d_lowres" / "crossval_results_folds_0_1_2_3_4",
    ),
]

NNUNET_SRC = (
    BME_ROOT
    / "ID7200 segmentation result"
    / "nnUNet-master"
    / "DATASET"
    / "nnUNet_trained_models"
    / "Dataset001_Liver"
    / "nnUNetTrainer__nnUNetPlans__3d_lowres"
)
NNUNET_DST = MODEL_ROOT / "nnunet" / "nnUNetTrainer__nnUNetPlans__3d_lowres"

DEMO_CT_SOURCES = [
    BME_ROOT / "Qt" / "final" / "imagesTr",
    BME_ROOT / "ID7200 segmentation result" / "liver_tumor_segmentation_testset50" / "imagesTs",
]


def copy_file(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied: {src.name} -> {dst.relative_to(PROJECT_ROOT)}")
    return True


def copy_tree(src: Path, dst: Path) -> int:
    if not src.is_dir():
        return 0
    count = 0
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            count += 1
    if count:
        print(f"  copied {count} files: {src.name} -> {dst.relative_to(PROJECT_ROOT)}")
    return count


def copy_nnunet_folds() -> int:
    count = 0
    if not NNUNET_SRC.is_dir():
        return 0
    for fold_dir in sorted(NNUNET_SRC.glob("fold_*")):
        for name in ("checkpoint_final.pth", "checkpoint_best.pth"):
            src = fold_dir / name
            if copy_file(src, NNUNET_DST / fold_dir.name / name):
                count += 1
    for meta in ("dataset.json", "plans.json", "readme.txt"):
        copy_file(NNUNET_SRC / meta, NNUNET_DST / meta)
    return count


def copy_demo_ct() -> int:
    dst = MODEL_ROOT / "demo" / "sample_ct"
    for src_dir in DEMO_CT_SOURCES:
        if not src_dir.is_dir():
            continue
        for nii in list(src_dir.glob("*.nii*"))[:3]:
            copy_file(nii, dst / nii.name)
        return len(list(dst.glob("*.nii*")))
    return 0


def main() -> None:
    print("Setting up models/ from local BME project...\n")
    copied = 0

    for src, dst in COPY_RULES:
        if src.is_file():
            if copy_file(src, dst):
                copied += 1
        elif src.is_dir():
            copied += copy_tree(src, dst)

    copied += copy_nnunet_folds()
    demo_count = copy_demo_ct()

    print("\n--- Summary ---")
    print(f"YOLO weights:     {'OK' if (MODEL_ROOT / 'yolo' / 'best.pt').exists() else 'MISSING'}")
    print(f"Predictions:      {len(list((MODEL_ROOT / 'predictions' / 'inferTs').glob('*.nii*')))} files")
    print(f"nnU-Net folds:    {len(list((NNUNET_DST).glob('fold_*/checkpoint*.pth')))} checkpoints")
    print(f"Demo CT samples:  {demo_count} files")

    if not (MODEL_ROOT / "yolo" / "best.pt").exists():
        print(
            "\nModel files not found on this machine.\n"
            "Manually copy into models/:\n"
            "  models/yolo/best.pt\n"
            "  models/predictions/inferTs/*.nii.gz\n"
            "  models/nnunet/.../fold_*/checkpoint_final.pth"
        )


if __name__ == "__main__":
    main()
