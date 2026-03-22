import json
import shutil
from pathlib import Path

CATEGORIES = {
    "car": 0,
    "person": 1,
    "traffic sign": 2,
    "traffic light": 3,
    "truck": 4,
    "bus": 5,
    "bike": 6,
    "rider": 7,
    "motor": 8,
    "train": 9,
}

IMG_W, IMG_H = 1280, 720


def convert_label(json_path):
    with open(json_path) as f:
        data = json.load(f)

    attrs = data.get("attributes", {})
    lines = []

    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            cat = obj.get("category")
            if cat not in CATEGORIES:
                continue

            box = obj.get("box2d")
            if box is None:
                continue

            x1 = max(0, min(float(box["x1"]), IMG_W))
            y1 = max(0, min(float(box["y1"]), IMG_H))
            x2 = max(0, min(float(box["x2"]), IMG_W))
            y2 = max(0, min(float(box["y2"]), IMG_H))

            w = (x2 - x1) / IMG_W
            h = (y2 - y1) / IMG_H
            if w <= 0 or h <= 0:
                continue

            cx = (x1 + x2) / 2 / IMG_W
            cy = (y1 + y2) / 2 / IMG_H

            lines.append(f"{CATEGORIES[cat]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines, attrs


def process_split(src_img_dir, src_lbl_dir, dst_dir, filter_fn):
    src_img_dir = Path(src_img_dir)
    src_lbl_dir = Path(src_lbl_dir)
    dst_dir = Path(dst_dir)

    dst_img = dst_dir / "images"
    dst_lbl = dst_dir / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for json_file in sorted(src_lbl_dir.glob("*.json")):
        stem = json_file.stem
        img_file = src_img_dir / f"{stem}.jpg"
        if not img_file.exists():
            continue

        lines, attrs = convert_label(json_file)

        if not filter_fn(attrs):
            continue

        # Copy image
        dst_img_path = dst_img / f"{stem}.jpg"
        if not dst_img_path.exists():
            shutil.copy2(img_file, dst_img_path)

        # Write YOLO label
        dst_lbl_path = dst_lbl / f"{stem}.txt"
        with open(dst_lbl_path, "w") as f:
            f.write("\n".join(lines))

        count += 1

    return count


def main():
    base = Path("data")
    src_images = base / "images"
    src_labels = base / "labels"

    clear_day = lambda a: a.get("weather") == "clear" and a.get("timeofday") == "daytime"

    print("Processing clear/daytime train...")
    n = process_split(
        src_images / "train", src_labels / "train",
        base / "clear_day" / "train", clear_day,
    )
    print(f"  {n} images")

    print("Processing clear/daytime val...")
    n = process_split(
        src_images / "val", src_labels / "val",
        base / "clear_day" / "val", clear_day,
    )
    print(f"  {n} images")

    conditions = {
        "rainy":    lambda a: a.get("weather") == "rainy",
        "snowy":    lambda a: a.get("weather") == "snowy",
        "night":    lambda a: a.get("timeofday") == "night",
        "overcast": lambda a: a.get("weather") == "overcast",
    }

    for name, filter_fn in conditions.items():
        print(f"Processing {name} (from test)...")
        n = process_split(
            src_images / "test", src_labels / "test",
            base / name, filter_fn,
        )
        print(f"  {n} images")

    print("\nDone. You can now delete data/images/ and data/labels/ to free space.")


if __name__ == "__main__":
    main()
