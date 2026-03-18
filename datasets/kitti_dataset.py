import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 3
DATA_DIR = Path(__file__).parent.parent / "data" / "kitti"


class KITTIDataset(Dataset):
    def __init__(self, split: str, img_size: int = 448, augment: bool = False):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got '{split}'"
        self.img_size = img_size
        self.augment = augment
        self.images_dir = DATA_DIR / split / "images"
        self.labels_dir = DATA_DIR / split / "labels"

        self.samples = sorted(
            p.stem for p in self.images_dir.glob("*.png")
            if (self.labels_dir / (p.stem + ".txt")).exists()
        )

        self._label_cache: dict[str, torch.Tensor] = {}
        for stem in self.samples:
            boxes = []
            for line in (self.labels_dir / f"{stem}.txt").read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([float(x) for x in parts])
            self._label_cache[stem] = (
                torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem = self.samples[idx]

        img = cv2.imread(str(self.images_dir / f"{stem}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        labels = self._label_cache[stem].clone()

        if self.augment:
            img, labels = self._apply_augmentations(img, labels)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, labels

    def _apply_augmentations(self, img: np.ndarray, labels: torch.Tensor):
        # Horizontal flip (50% chance)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            if labels.shape[0] > 0:
                labels[:, 1] = 1.0 - labels[:, 1]  # flip cx

        # Color jitter — random brightness, contrast, saturation
        if random.random() < 0.5:
            img = img.astype(np.float32)
            # Brightness: shift by ±30
            img += random.uniform(-30, 30)
            # Contrast: scale by 0.7–1.3
            img = img * random.uniform(0.7, 1.3)
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Random HSV saturation shift
        if random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.6, 1.4)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img, labels


def collate_fn(batch):
    images, label_list = zip(*batch)
    images = torch.stack(images)

    max_boxes = max(l.shape[0] for l in label_list)
    padded = torch.full((len(label_list), max_boxes, 5), -1.0)
    for i, l in enumerate(label_list):
        padded[i, : l.shape[0]] = l

    return images, padded


def get_dataloader(
    split: str,
    img_size: int = 448,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool | None = None,
) -> DataLoader:
    dataset = KITTIDataset(split=split, img_size=img_size, augment=(split == "train"))
    should_shuffle = shuffle if shuffle is not None else (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
