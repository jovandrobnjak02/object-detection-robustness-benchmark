"""
Requires:
  scripts/split_dataset.sh must be run first on raw KITTI data.

Usage:
  python scripts/kitti_to_yolo.py
"""

import struct
from pathlib import Path


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        f.read(16)
        w = struct.unpack(">I", f.read(4))[0]
        h = struct.unpack(">I", f.read(4))[0]
    return w, h


CLASS_MAP = {
    "Car": 0,
    "Van": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}

DATA_DIR = Path(__file__).parent.parent / "data" / "kitti"
SPLITS = ["train", "val"]


def convert_label(label_path: Path, image_path: Path) -> None:
    img_w, img_h = png_size(image_path)

    yolo_lines = []
    with open(label_path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 15:
                continue
            obj_type = fields[0]
            if obj_type not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[obj_type]
            left, top, right, bottom = map(float, fields[4:8])

            cx = min(max((left + right) / 2 / img_w, 0.0), 1.0)
            cy = min(max((top + bottom) / 2 / img_h, 0.0), 1.0)
            w = min((right - left) / img_w, 1.0)
            h = min((bottom - top) / img_h, 1.0)

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")


def main():
    for split in SPLITS:
        labels_dir = DATA_DIR / split / "labels"
        images_dir = DATA_DIR / split / "images"

        label_files = sorted(labels_dir.glob("*.txt"))
        converted, skipped = 0, 0

        for label_path in label_files:
            image_path = images_dir / (label_path.stem + ".png")
            if not image_path.exists():
                print(f"  Missing image for {label_path.name}, skipping")
                skipped += 1
                continue
            convert_label(label_path, image_path)
            converted += 1

        print(f"{split}: converted {converted} labels, skipped {skipped}")


if __name__ == "__main__":
    main()
