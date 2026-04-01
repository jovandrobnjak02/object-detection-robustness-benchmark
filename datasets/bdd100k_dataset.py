import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

NUM_CLASSES = 10
DATA_DIR = Path(__file__).parent.parent / "data"
SHM_ROOT  = Path("/dev/shm/bdd100k_cache")


class BDD100KDataset(Dataset):
    def __init__(self, split: str, img_size: int = 448, augment: bool = False):
        self.img_size = img_size
        self.augment = augment
        self.images_dir = DATA_DIR / split / "images"
        self.labels_dir = DATA_DIR / split / "labels"

        # Use /dev/shm cache if available (pre-letterboxed, much faster reads)
        shm_dir = SHM_ROOT / split / "images"
        self._cache_dir = shm_dir if shm_dir.exists() else None

        if self._cache_dir:
            # Only use images that are actually in the cache
            cached_stems = set(p.stem for p in self._cache_dir.glob("*.jpg"))
            self.samples = sorted(
                s for s in (p.stem for p in self.images_dir.glob("*.jpg"))
                if s in cached_stems and (self.labels_dir / (s + ".txt")).exists()
            )
            print(f"  [{split}] using /dev/shm cache ({len(self.samples)} images)")
        else:
            self.samples = sorted(
                p.stem for p in self.images_dir.glob("*.jpg")
                if (self.labels_dir / (p.stem + ".txt")).exists()
            )

        # Pre-compute letterbox params (constant for all 1280x720 BDD100K images)
        self._orig_w, self._orig_h = 1280, 720
        self._scale = img_size / max(self._orig_w, self._orig_h)
        self._new_w = int(self._orig_w * self._scale)
        self._new_h = int(self._orig_h * self._scale)
        self._pad_left = (img_size - self._new_w) // 2
        self._pad_top = (img_size - self._new_h) // 2

        # Cache labels in memory, pre-remapped to letterboxed coords
        self._label_cache: dict[str, torch.Tensor] = {}
        for stem in self.samples:
            boxes = []
            for line in (self.labels_dir / f"{stem}.txt").read_text().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append([float(x) for x in parts])
            labels = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
            self._label_cache[stem] = self._remap_labels(
                labels, self._orig_w, self._orig_h,
                self._scale, self._pad_left, self._pad_top, img_size
            )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _letterbox(img: np.ndarray, target: int):
        h, w = img.shape[:2]
        scale = target / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

        pad_top = (target - new_h) // 2
        pad_bot = target - new_h - pad_top
        pad_left = (target - new_w) // 2
        pad_right = target - new_w - pad_left
        img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, scale, pad_left, pad_top

    @staticmethod
    def _remap_labels(labels: torch.Tensor, orig_w: int, orig_h: int,
                      scale: float, pad_left: int, pad_top: int, target: int):
        if labels.shape[0] == 0:
            return labels
        # Convert from normalized original coords to pixel coords
        cx_px = labels[:, 1] * orig_w * scale + pad_left
        cy_px = labels[:, 2] * orig_h * scale + pad_top
        w_px = labels[:, 3] * orig_w * scale
        h_px = labels[:, 4] * orig_h * scale
        # Back to normalized in letterboxed image
        labels[:, 1] = cx_px / target
        labels[:, 2] = cy_px / target
        labels[:, 3] = w_px / target
        labels[:, 4] = h_px / target
        return labels

    def _load_image(self, stem: str, size: int | None = None):
        target = size or self.img_size

        if self._cache_dir and target == self.img_size:
            cached = self._cache_dir / f"{stem}.jpg"
            if cached.exists():
                img = cv2.imread(str(cached))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Slow path: read original and letterbox on the fly
        img = cv2.imread(str(self.images_dir / f"{stem}.jpg"))
        if img is None:
            raise FileNotFoundError(f"Failed to load: {self.images_dir / stem}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _, _, _ = self._letterbox(img, target)
        return img

    def __getitem__(self, idx: int):
        if self.augment and random.random() < 0.5:
            img, labels = self._apply_mosaic(idx)
        else:
            stem = self.samples[idx]
            img = self._load_image(stem)
            labels = self._label_cache[stem].clone()  # ALWAYS clone

        if self.augment:
            img, labels = self._apply_augmentations(img, labels)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, labels

    def _apply_mosaic(self, idx: int):
        half = self.img_size // 2
        indices = [idx] + random.choices(range(len(self)), k=3)

        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        all_labels = []

        for i, src_idx in enumerate(indices):
            stem = self.samples[src_idx]

            # Load FULL size, then resize to quadrant
            img = self._load_image(stem)
            img = cv2.resize(img, (half, half))

            row, col = divmod(i, 2)
            y_off = row * half
            x_off = col * half

            canvas[y_off:y_off + half, x_off:x_off + half] = img

            labels = self._label_cache[stem].clone()

            if labels.shape[0] > 0:
                # Scale labels from full image → quadrant
                labels[:, 1] = labels[:, 1] * 0.5 + col * 0.5
                labels[:, 2] = labels[:, 2] * 0.5 + row * 0.5
                labels[:, 3] *= 0.5
                labels[:, 4] *= 0.5

                # Clip boxes
                x1 = (labels[:, 1] - labels[:, 3] / 2).clamp(0, 1)
                y1 = (labels[:, 2] - labels[:, 4] / 2).clamp(0, 1)
                x2 = (labels[:, 1] + labels[:, 3] / 2).clamp(0, 1)
                y2 = (labels[:, 2] + labels[:, 4] / 2).clamp(0, 1)

                labels[:, 1] = (x1 + x2) / 2
                labels[:, 2] = (y1 + y2) / 2
                labels[:, 3] = x2 - x1
                labels[:, 4] = y2 - y1

                # Keep small objects (less aggressive)
                valid = (labels[:, 3] > 0.002) & (labels[:, 4] > 0.002)
                labels = labels[valid]

                if labels.shape[0] > 0:
                    all_labels.append(labels)

        if all_labels:
            all_labels = torch.cat(all_labels, dim=0)
        else:
            all_labels = torch.zeros((0, 5))

        return canvas, all_labels

    def _apply_augmentations(self, img: np.ndarray, labels: torch.Tensor):
        # Horizontal flip (50% chance)
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            if labels.shape[0] > 0:
                labels[:, 1] = 1.0 - labels[:, 1]  # flip cx

        # Color jitter
        if random.random() < 0.5:
            img = img.astype(np.float32)
            img += random.uniform(-30, 30)
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
    if max_boxes == 0:
        max_boxes = 1
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
    dataset = BDD100KDataset(split=split, img_size=img_size, augment=("train" in split))
    should_shuffle = shuffle if shuffle is not None else ("train" in split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )