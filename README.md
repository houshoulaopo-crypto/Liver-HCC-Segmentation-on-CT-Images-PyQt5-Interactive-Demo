# Liver HCC Segmentation on CT Images — PyQt5 Interactive Demo

**Competition entry (ID: 7200)** — nnU-Net v2 + YOLOv8 fusion pipeline with a **PyQt5 desktop demo** for slice viewing, mask overlay, and VTK 3D reconstruction.

Model weights are hosted separately on **Hugging Face** (too large for GitHub).

> **Important:** Run all `git` commands inside **`E:\BME\github-upload`**, not the parent `E:\BME` folder.
> This folder is already initialized as a Git repository.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models from Hugging Face (after you upload them)
python download_models.py --repo YOUR_USERNAME/liver-hcc-models

# 3. Launch GUI
python app/main.py
```

## Project Layout

```
github-upload/                 ← upload THIS folder to GitHub
├── app/
│   ├── main.py                ← GUI entry point
│   ├── liver_tumor_segment_app.py
│   └── liver_tumor_segment_app.ui
├── scripts/
│   ├── check_nii.py           ← nnU-Net + YOLO fusion
│   ├── cal_dice.py            ← evaluation
│   └── yolo/                  ← YOLO training & inference scripts
├── preprocessing/             ← data preprocessing utilities
├── nnunet_metadata/           ← training logs & cross-val metrics
├── config.py                  ← paths configuration
├── download_models.py         ← download from Hugging Face
├── requirements.txt
└── README.md

huggingface-models/            ← upload THIS folder to Hugging Face
├── yolo/best.pt
├── predictions/inferTs/
└── nnunet/crossval_results_folds_0_1_2_3_4/
```

## Hugging Face Setup

1. Go to [huggingface.co/new](https://huggingface.co/new) and create a **Model** repo (e.g. `liver-hcc-models`)
2. Copy your model files into `E:\BME\huggingface-models\` following the folder layout
3. Upload to Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli login
cd E:\BME\huggingface-models
huggingface-cli upload YOUR_USERNAME/liver-hcc-models . .
```

4. Edit `config.py` and set `HF_REPO_ID = "YOUR_USERNAME/liver-hcc-models"`

## GitHub Upload

**Use this folder:** `E:\BME\github-upload` (Git repo is already initialized here)

### Option A — Command line

```bash
cd E:\BME\github-upload
git remote add origin https://github.com/YOUR_USERNAME/liver-hcc-qt-demo.git
git branch -M main
git push -u origin main
```

### Option B — GitHub Desktop

1. Open GitHub Desktop → **File → Add local repository**
2. Choose folder: `E:\BME\github-upload` (not `E:\BME`)
3. Click **Publish repository**

### Option C — VS Code / Cursor

1. **File → Open Folder** → select `E:\BME\github-upload`
2. Source Control panel → **Publish to GitHub**

If you see *"not a git repository"*, you opened the wrong folder. Go up one level and pick `github-upload`.

### First-time setup (already done)

```bash
cd E:\BME\github-upload
git init
git add .
git commit -m "Initial commit: Liver HCC Segmentation PyQt5 interactive demo"
```

## Results

| Metric | Value |
|--------|-------|
| Dice (DSC) | 0.815 |
| HD95 | 26.637 |

## License

Academic competition project. Third-party libraries retain their original licenses.
