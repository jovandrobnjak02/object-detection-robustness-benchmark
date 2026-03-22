# Data

## Source

BDD100K dataset: http://bdd-data.berkeley.edu/download.html

Download `bdd100k_images_100k.zip` and `bdd100k_labels.zip`, place them in this directory, then run:

```bash
cd data
unzip bdd100k_images_100k.zip && mv 100k images
unzip bdd100k_labels.zip && mv 100k labels
cd .. && python scripts/prepare_bdd100k.py
```

This filters and converts the raw data into the structure below. After the script finishes, delete the raw data to free space:

```bash
rm -rf data/images data/labels data/*.zip
```

Optionally, pre-cache training images to `/dev/shm` for faster data loading (lost on reboot):

```bash
python scripts/cache_to_shm.py        # full dataset
python scripts/cache_to_shm.py 0.5    # 50% of training images
```

## Structure

```
data/
├── clear_day/
│   ├── train/          12,477 images (clear weather + daytime, from original train split)
│   │   ├── images/     1280×720 JPG, original resolution
│   │   └── labels/     YOLO format .txt, one per image
│   └── val/            1,764 images (clear weather + daytime, from original val split)
│       ├── images/
│       └── labels/
├── rainy/              1,306 images (rainy weather, from original test split)
│   ├── images/
│   └── labels/
├── snowy/              1,551 images (snowy weather, from original test split)
│   ├── images/
│   └── labels/
├── night/              8,036 images (nighttime, from original test split)
│   ├── images/
│   └── labels/
├── overcast/           2,568 images (overcast weather, from original test split)
│   ├── images/
│   └── labels/
└── bdd100k.yaml        Ultralytics dataset config for YOLO26 training
```

## Split Design

Models are trained on `clear_day/train` and validated on `clear_day/val`. Robustness is evaluated on the four adverse condition splits (`rainy`, `snowy`, `night`, `overcast`), which come from the original BDD100K test split and were never seen during training.

## Labels

YOLO format: `class_id cx cy w h` (normalized 0-1), one `.txt` per image.

10 detection classes (lane markings and drivable area annotations excluded):

| ID | Class         | Train instances |
|----|---------------|----------------|
| 0  | car           | 714,121        |
| 1  | person        | 91,435         |
| 2  | traffic sign  | 239,961        |
| 3  | traffic light | 186,301        |
| 4  | truck         | 30,012         |
| 5  | bus           | 11,688         |
| 6  | bike          | 7,227          |
| 7  | rider         | 4,522          |
| 8  | motor         | 3,002          |
| 9  | train         | 136            |

## Citation

```bibtex
@InProceedings{bdd100k,
    author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
    title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```