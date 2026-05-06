# FSAM-YOLO: Few-Shot Attention Meta-YOLO for Reef Detection

*Source code and dataset for the paper "Shallow-Water Reef Detection Driven by Few-Shot Learning with Airborne LiDAR Bathymetry" (Su et al.). Provided for academic testing, verification, and reproduction.*

Implementation of the FSAM-YOLO model with Cross-Scale Attention Module (CSAM) and Reptile-style MAML meta-learning for shallow-water reef detection using ALB data feature normalized images.

## Quick Start

### 1. Setup Environment

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Preparation

#### Using the Provided ReefFeat-Img Dataset

The dataset is included in the project at `datasets/ReefFeat-Img dataset/`. The config file is `ultralytics/cfg/datasets/reef.yaml`.

#### Using a Custom Dataset

You can also build your own dataset by modifying `ultralytics/cfg/datasets/reef.yaml` to point to your data path and class names.

### 3. Run Training

```bash
python fsam_yolo_train_.py
```

Training results will be saved to `runs/fsam_yolo/train/`.

## Project Structure

```
FSAM-YOLO/
├── fsam_yolo_train_.py                    # MAML + YOLO training script
├── sample_augmentation.py                 # Composite data augmentation for reef samples
├── requirements.txt                       # Python dependencies
├── pyproject.toml                         # Package configuration
├── ultralytics/                           # YOLOv10 source with CSAM
│   ├── nn/modules/conv.py                 # CSAM attention module
│   ├── nn/tasks.py                        # CSAM parser + YOLOv10DetectionModel
│   └── cfg/models/v10/yolov10s-csam.yaml  # CSAM model config
├── datasets/
│   └── ReefFeat-Img dataset/              # Dataset constructed for this paper
│       ├── DA/                            # Augmented training samples
│       │   ├── image/
│       │   └── labels/
│       └── Primitive/                     # Original un-augmented samples
│           ├── image/
│           └── labels/
└── weights/                               # Pretrained weights (download manually)
    └── yolov10s.pt
```

## License

This project is released under the AGPL-3.0 license, inherited from Ultralytics YOLOv10.
