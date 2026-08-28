# Liver HCC Segmentation on CT Images — PyQt5 Interactive Demo

nnU-Net v2 + YOLOv8 fusion pipeline with a **PyQt5 desktop demo** for slice viewing, mask overlay, and VTK 3D reconstruction.

Model weights are hosted separately on **Hugging Face** (too large for GitHub).


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


## Results

| Metric | Value |
|--------|-------|
| Dice (DSC) | 0.815 |
| HD95 | 26.637 |

## License

Academic competition project. Third-party libraries retain their original licenses.
