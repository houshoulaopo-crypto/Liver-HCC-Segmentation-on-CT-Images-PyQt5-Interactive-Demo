# Liver HCC Segmentation on CT Images — PyQt5 Interactive Demo

9th National Undergraduate Biomedical Engineering Innovation Design Competition, 2024.

All-in-one runnable demo: nnU-Net v2 + YOLOv8 fusion with **PyQt5 GUI** for slice viewing, mask overlay, and VTK 3D reconstruction.

**Repo:** https://github.com/houshoulaopo-crypto/Liver-HCC-Segmentation-on-CT-Images-PyQt5-Interactive-Demo

## Demo Video

https://github.com/houshoulaopo-crypto/Liver-HCC-Segmentation-on-CT-Images-PyQt5-Interactive-Demo/raw/main/docs/demo.mp4

<video src="docs/demo.mp4" controls width="100%"></video>

## Quick Start

```bash
git clone https://github.com/houshoulaopo-crypto/Liver-HCC-Segmentation-on-CT-Images-PyQt5-Interactive-Demo.git
cd Liver-HCC-Segmentation-on-CT-Images-PyQt5-Interactive-Demo

pip install -r requirements.txt
python run.py             # launch GUI
```

Windows: double-click `run.bat`

## Project Layout

```
├── docs/
│   └── demo.mp4               # GUI demo screencast
├── run.py / run.bat           # launch demo
├── setup_assets.py            # auto-copy models from local BME backup (optional)
├── config.py                  # paths (everything under models/)
├── app/
│   ├── main.py                # PyQt5 GUI entry
│   ├── liver_tumor_segment_app.py
│   └── liver_tumor_segment_app.ui
├── models/                    # weights, predictions & demo CT (in repo)
│   ├── yolo/best.pt
│   ├── predictions/inferTs/
│   ├── nnunet/.../crossval_results_folds_0_1_2_3_4/
│   └── demo/sample_ct/
├── scripts/                   # training, fusion, evaluation
├── preprocessing/
└── nnunet_metadata/           # cross-val metrics & logs
```

> **Note:** nnU-Net are **not** included. The demo runs using pre-computed masks in `models/predictions/inferTs/`.

## GUI Usage

| Button | Function |
|--------|----------|
| Open File | Load `.nii` / `.nii.gz` CT scan (try `models/demo/sample_ct/`) |
| Start Segmentation | Load pre-computed mask from `models/predictions/inferTs/` |
| YOLO Prediction | Run slice-wise detection with `models/yolo/best.pt` |
| Show Mask / Segmentation | Toggle overlay views |
| 3D Segmentation | VTK surface rendering |

## Results

| Metric | Value |
|--------|-------|
| Dice (DSC) | 0.815 |
| HD95 | 26.637 |

## License

Academic competition project. Third-party libraries retain their original licenses.
