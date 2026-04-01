import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMG_SIZE  = 640
SHM_ROOT  = Path("/dev/shm/bdd100k_cache")
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SPLITS    = ["clear_day/train", "clear_day/val"]
TRAIN_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0


def letterbox(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))
    pad_top   = (target - new_h) // 2
    pad_bot   = target - new_h - pad_top
    pad_left  = (target - new_w) // 2
    pad_right = target - new_w - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def cache_split(split: str):
    src = DATA_ROOT / split / "images"
    dst = SHM_ROOT / split / "images"
    dst.mkdir(parents=True, exist_ok=True)

    imgs = sorted(src.glob("*.jpg"))

    # Subsample training set if TRAIN_FRAC < 1.0
    if "train" in split and TRAIN_FRAC < 1.0:
        k = int(len(imgs) * TRAIN_FRAC)
        rng = random.Random(42)
        imgs = rng.sample(imgs, k)
        imgs = sorted(imgs)

    already = set(p.name for p in dst.glob("*.jpg"))
    todo = [p for p in imgs if p.name not in already]

    if not todo:
        print(f"  {split}: already cached ({len(imgs)} images)")
        return

    print(f"  {split}: caching {len(todo)}/{len(imgs)} images ...", flush=True)
    t0 = time.time()
    for i, img_path in enumerate(todo):
        img = cv2.imread(str(img_path))
        img = letterbox(img, IMG_SIZE)
        cv2.imwrite(str(dst / img_path.name), img)
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(todo)}", flush=True)
    print(f"    done in {time.time() - t0:.0f}s")


def main():
    print(f"Caching to {SHM_ROOT} ...")
    for split in SPLITS:
        cache_split(split)

    used = sum(f.stat().st_size for f in SHM_ROOT.rglob("*.jpg")) / 1e6
    print(f"\nTotal cache size: {used:.0f} MB")
    print("Run training now.")


if __name__ == "__main__":
    main()
