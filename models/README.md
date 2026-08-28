# Model & Demo Assets

All files needed to run the Qt demo live in this folder and are committed to GitHub.

## Required layout

```
models/
├── yolo/
│   └── best.pt                          # YOLOv8 weights (~6 MB)
├── predictions/
│   └── inferTs/
│       ├── 151.nii.gz                   # pre-computed segmentation masks
│       └── ...
├── nnunet/
│   └── nnUNetTrainer__nnUNetPlans__3d_lowres/
│       ├── fold_0/checkpoint_final.pth  # nnU-Net weights (optional, for re-training/inference)
│       ├── fold_1/checkpoint_final.pth
│       └── ...
└── demo/
    └── sample_ct/
        └── 151_0000.nii.gz              # sample CT for quick demo (optional)
```

## Auto-copy from original project

If you still have the old `E:\BME` folder with model backups:

```bash
python setup_assets.py
```

## Manual copy (original paths)

| File | Original location |
|------|-------------------|
| `yolo/best.pt` | `ID7200 segmentation result/YOLO/train4/weights/best.pt` |
| `predictions/inferTs/` | `ID7200 segmentation result/liver_tumor_segmentation_testset50/inferTs/` |
| `nnunet/.../fold_*/checkpoint_final.pth` | `ID7200 segmentation result/nnUNet-master/DATASET/nnUNet_trained_models/Dataset001_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres/` |

## GUI usage

1. **Open File** → load a `.nii` / `.nii.gz` CT volume (or use `demo/sample_ct/`)
2. **Start Segmentation** → loads mask from `predictions/inferTs/` matching case ID prefix
3. **YOLO Prediction** → runs `yolo/best.pt` slice-by-slice

Mask matching example: CT `151_0000.nii.gz` → mask `151.nii.gz`
