"""Download model weights and predictions from Hugging Face."""
import argparse
from pathlib import Path

from config import HF_REPO_ID, MODEL_ROOT, PROJECT_ROOT


def download(repo_id: str, local_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print("Done.")
    print(f"YOLO weights expected at: {local_dir / 'yolo' / 'best.pt'}")
    print(f"Predictions expected at: {local_dir / 'predictions' / 'inferTs'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download liver HCC segmentation models from Hugging Face")
    parser.add_argument(
        "--repo",
        default=HF_REPO_ID,
        help=f"Hugging Face repo id (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--output",
        default=str(MODEL_ROOT),
        help=f"Local output directory (default: {MODEL_ROOT})",
    )
    args = parser.parse_args()

    if args.repo == "YOUR_USERNAME/liver-hcc-models":
        raise SystemExit(
            "Set your Hugging Face repo id first:\n"
            "  1. Edit config.py -> HF_REPO_ID\n"
            "  2. Or run: python download_models.py --repo your-username/liver-hcc-models"
        )

    download(args.repo, Path(args.output))


if __name__ == "__main__":
    main()
