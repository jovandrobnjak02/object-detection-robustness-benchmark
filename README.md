# Object Detection Robustness Benchmark

Project comparing a custom CNN (ResNet-50 + FPN) against YOLO26m on the BDD100K dataset. Models are trained on clear daytime conditions and evaluated on adverse weather and lighting conditions to measure robustness.

## Models

**Custom CNN**: ResNet-50 backbone (pretrained on ImageNet) + FPN neck + shared detection head.
- Multi-scale detection across 3 FPN levels (P3, P4, P5)
- 2 anchor boxes per cell, 10 classes
- Backbone frozen during warmup (3 epochs), then progressively unfrozen layer-by-layer
- AdamW optimizer, LR warmup + cosine annealing, mixed precision

**YOLO26m** - Ultralytics YOLO26 medium, pretrained on COCO, fine-tuned on BDD100K.
- Anchor-free, multi-scale detection

## Dataset

BDD100K - 100K diverse driving images with weather and time-of-day annotations.

| Split | Condition | Images |
|---|---|---|
| Train | Clear + daytime | 12,477 |
| Val | Clear + daytime | 1,764 |
| Test | Rainy | 1,306 |
| Test | Snowy | 1,551 |
| Test | Night | 8,036 |
| Test | Overcast | 2,568 |

10 detection classes: car, person, traffic sign, traffic light, truck, bus, bike, rider, motor, train.

See [data/README.md](data/README.md) for setup instructions.

## Project Structure

```
├── data/                       Dataset (see data/README.md for setup)
├── datasets/
│   └── bdd100k_dataset.py      PyTorch Dataset with letterboxing + mosaic augmentation
├── models/custom_cnn/
│   ├── model.py                Custom CNN architecture
│   └── loss.py                 Detection loss, decode_predictions, NMS
├── training/
│   ├── train_custom_cnn.py     Custom CNN training script
│   └── train_yolo26.py         YOLO26 fine-tuning script
├── evaluation/
│   ├── evaluate.py             Evaluate both models → JSON + CSV
│   └── plot_results.py         Generate thesis plots from evaluation results
├── scripts/
│   ├── prepare_bdd100k.py      Filter + convert BDD100K to YOLO format
│   └── cache_to_shm.py         Pre-cache images to /dev/shm for faster training
├── notebooks/
│   └── eda.ipynb               Dataset exploration
└── results/
    ├── logs/                   Training loss CSVs
    ├── metrics/                Evaluation outputs (JSON, CSV)
    └── plots/                  Generated figures
```

## Quick Start

### 1. Setup data

```bash
# Download bdd100k_images_100k.zip and bdd100k_labels.zip from http://bdd-data.berkeley.edu
cd data
unzip bdd100k_images_100k.zip && mv 100k images
unzip bdd100k_labels.zip && mv 100k labels
cd .. && python scripts/prepare_bdd100k.py
```

### 2. Cache images (optional, speeds up training)

```bash
python scripts/cache_to_shm.py        # full dataset (~850MB in /dev/shm)
python scripts/cache_to_shm.py 0.5    # 50% of training data
```

### 3. Train

```bash
# Custom CNN
python training/train_custom_cnn.py

# YOLO26
python training/train_yolo26.py --model yolo26m
```

### 4. Evaluate

See [evaluation/README.md](evaluation/README.md) for details.

```bash
python evaluation/evaluate.py
python evaluation/plot_results.py
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, ultralytics, OpenCV, NumPy, matplotlib.
