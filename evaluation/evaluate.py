import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.bdd100k_dataset import BDD100KDataset
from models.custom_cnn import CustomCNN, decode_predictions, nms
from evaluation.metrics import (
    box_cxcywh_to_xyxy,
    compute_map,
    compute_confusion_matrix,
)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 448
CONF_THRESH  = 0.25
NMS_THRESH   = 0.45
IOU_THRESH   = 0.5   # mAP@50

CUSTOM_CNN_CKPT = Path("checkpoints/custom_cnn/bdd100k_custom_cnn_best.pt")
YOLO26_CKPT     = Path("checkpoints/yolo26/bdd100k/weights/best.pt")

SPLITS = ["clear_day/val"]

CLASS_NAMES = [
    "car", "person", "traffic sign", "traffic light",
    "truck", "bus", "bike", "rider", "motor", "train",
]
NUM_CLASSES = len(CLASS_NAMES)

RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_gt(dataset: BDD100KDataset) -> list[dict]:
    gts = []
    for img_id, stem in enumerate(dataset.samples):
        labels = dataset._label_cache[stem]
        if labels.shape[0] == 0:
            gts.append({"img_id": img_id, "boxes": []})
            continue
        boxes_xyxy = box_cxcywh_to_xyxy(labels[:, 1:].numpy())
        cls = labels[:, 0:1].numpy()
        boxes = np.concatenate([boxes_xyxy, cls], axis=1)
        gts.append({"img_id": img_id, "boxes": boxes})
    return gts


def load_custom_cnn() -> torch.nn.Module:
    model = CustomCNN(C=NUM_CLASSES)
    if not CUSTOM_CNN_CKPT.exists():
        raise FileNotFoundError(f"Custom CNN checkpoint not found: {CUSTOM_CNN_CKPT}")
    ckpt = torch.load(CUSTOM_CNN_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    return model


def run_custom_cnn(model: torch.nn.Module, dataset: BDD100KDataset) -> tuple[list[dict], float, float]:
    all_preds = []
    total_time = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for img_id, stem in enumerate(dataset.samples):
            img = dataset._load_image(stem)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(DEVICE)

            t0 = time.perf_counter()
            preds = model(tensor)
            detections = decode_predictions(preds, conf_thresh=CONF_THRESH)
            dets = nms(detections[0], iou_thresh=NMS_THRESH)
            torch.cuda.synchronize()
            total_time += time.perf_counter() - t0

            boxes = []
            if dets.shape[0] > 0:
                dets_np = dets.cpu().numpy()
                cxcywh = dets_np[:, :4]
                xyxy   = box_cxcywh_to_xyxy(cxcywh)
                conf   = dets_np[:, 4]
                cls_id = dets_np[:, 6].astype(int)
                for k in range(len(dets_np)):
                    boxes.append([*xyxy[k], conf[k], cls_id[k]])

            all_preds.append({"img_id": img_id, "boxes": boxes})

    fps = len(dataset.samples) / total_time
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return all_preds, fps, gpu_mem_mb


def load_yolo26():
    from ultralytics import YOLO
    if not YOLO26_CKPT.exists():
        raise FileNotFoundError(f"YOLO26 checkpoint not found: {YOLO26_CKPT}")
    return YOLO(str(YOLO26_CKPT))


def run_yolo26(model, dataset: BDD100KDataset) -> tuple[list[dict], float, float]:
    all_preds = []
    total_time = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for img_id, stem in enumerate(dataset.samples):
        img = dataset._load_image(stem)

        t0 = time.perf_counter()
        results = model.predict(img, imgsz=IMG_SIZE, conf=CONF_THRESH,
                                iou=NMS_THRESH, verbose=False)
        torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        boxes = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxyn   = r.boxes.xyxyn.cpu().numpy()
            confs   = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for k in range(len(xyxyn)):
                boxes.append([*xyxyn[k], confs[k], cls_ids[k]])

        all_preds.append({"img_id": img_id, "boxes": boxes})

    fps = len(dataset.samples) / total_time
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return all_preds, fps, gpu_mem_mb


def evaluate_split(split: str, custom_cnn_model, yolo26_model) -> dict:
    print(f"\n  Split: {split}")
    dataset = BDD100KDataset(split=split, img_size=IMG_SIZE, augment=False)
    print(f"    {len(dataset)} images")

    all_gts = load_gt(dataset)
    results = {"split": split}

    # Custom CNN
    print(f"    Running Custom CNN ...", end=" ", flush=True)
    cnn_preds, cnn_fps, cnn_gpu_mb = run_custom_cnn(custom_cnn_model, dataset)
    cnn_map, cnn_aps, cnn_precs, cnn_recs = compute_map(cnn_preds, all_gts, NUM_CLASSES, IOU_THRESH)
    cnn_cm = compute_confusion_matrix(cnn_preds, all_gts, NUM_CLASSES, IOU_THRESH)

    results["custom_cnn_mAP50"]      = round(cnn_map * 100, 2)
    results["custom_cnn_fps"]        = round(cnn_fps, 1)
    results["custom_cnn_gpu_mem_mb"] = round(cnn_gpu_mb, 1)
    results["custom_cnn_ap_per_class"] = {
        CLASS_NAMES[i]: round(cnn_aps[i] * 100, 2) if not np.isnan(cnn_aps[i]) else None
        for i in range(NUM_CLASSES)
    }
    results["custom_cnn_precision_per_class"] = {
        CLASS_NAMES[i]: round(cnn_precs[i] * 100, 2) if not np.isnan(cnn_precs[i]) else None
        for i in range(NUM_CLASSES)
    }
    results["custom_cnn_recall_per_class"] = {
        CLASS_NAMES[i]: round(cnn_recs[i] * 100, 2) if not np.isnan(cnn_recs[i]) else None
        for i in range(NUM_CLASSES)
    }
    results["custom_cnn_confusion_matrix"] = cnn_cm.tolist()
    print(f"mAP50={cnn_map*100:.1f}%  FPS={cnn_fps:.1f}  GPU={cnn_gpu_mb:.0f}MB")

    # YOLO26
    if yolo26_model is not None:
        print(f"    Running YOLO26 ...", end=" ", flush=True)
        yolo_preds, yolo_fps, yolo_gpu_mb = run_yolo26(yolo26_model, dataset)
        yolo_map, yolo_aps, yolo_precs, yolo_recs = compute_map(yolo_preds, all_gts, NUM_CLASSES, IOU_THRESH)
        yolo_cm = compute_confusion_matrix(yolo_preds, all_gts, NUM_CLASSES, IOU_THRESH)

        results["yolo26_mAP50"]      = round(yolo_map * 100, 2)
        results["yolo26_fps"]        = round(yolo_fps, 1)
        results["yolo26_gpu_mem_mb"] = round(yolo_gpu_mb, 1)
        results["yolo26_ap_per_class"] = {
            CLASS_NAMES[i]: round(yolo_aps[i] * 100, 2) if not np.isnan(yolo_aps[i]) else None
            for i in range(NUM_CLASSES)
        }
        results["yolo26_precision_per_class"] = {
            CLASS_NAMES[i]: round(yolo_precs[i] * 100, 2) if not np.isnan(yolo_precs[i]) else None
            for i in range(NUM_CLASSES)
        }
        results["yolo26_recall_per_class"] = {
            CLASS_NAMES[i]: round(yolo_recs[i] * 100, 2) if not np.isnan(yolo_recs[i]) else None
            for i in range(NUM_CLASSES)
        }
        results["yolo26_confusion_matrix"] = yolo_cm.tolist()
        print(f"mAP50={yolo_map*100:.1f}%  FPS={yolo_fps:.1f}  GPU={yolo_gpu_mb:.0f}MB")

    return results


def main():
    print("Loading models...")
    custom_cnn = load_custom_cnn()
    print(f"  Custom CNN loaded from {CUSTOM_CNN_CKPT}")

    yolo26 = None
    if YOLO26_CKPT.exists():
        yolo26 = load_yolo26()
        print(f"  YOLO26 loaded from {YOLO26_CKPT}")
    else:
        print(f"  YOLO26 checkpoint not found ({YOLO26_CKPT}), skipping")

    print("\nEvaluating...")
    all_results = []
    for split in SPLITS:
        try:
            r = evaluate_split(split, custom_cnn, yolo26)
            all_results.append(r)
        except Exception as e:
            print(f"  [SKIP] {split}: {e}")

    out_json = RESULTS_DIR / "bdd100k_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_json}")

    print("\n" + "=" * 76)
    print(f"{'Split':<20} {'CNN mAP50':>10} {'YOL mAP50':>10} {'CNN FPS':>8} {'YOL FPS':>8} {'CNN MB':>7} {'YOL MB':>7}")
    print("-" * 76)
    for r in all_results:
        print(
            f"{r['split']:<20}"
            f" {str(r.get('custom_cnn_mAP50', '-')):>10}"
            f" {str(r.get('yolo26_mAP50', 'N/A')):>10}"
            f" {str(r.get('custom_cnn_fps', '-')):>8}"
            f" {str(r.get('yolo26_fps', 'N/A')):>8}"
            f" {str(r.get('custom_cnn_gpu_mem_mb', '-')):>7}"
            f" {str(r.get('yolo26_gpu_mem_mb', 'N/A')):>7}"
        )
    print("=" * 76)

    out_csv = RESULTS_DIR / "bdd100k_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split",
            "custom_cnn_mAP50", "custom_cnn_fps", "custom_cnn_gpu_mem_mb",
            "yolo26_mAP50",     "yolo26_fps",     "yolo26_gpu_mem_mb",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "split":                  r["split"],
                "custom_cnn_mAP50":       r.get("custom_cnn_mAP50", ""),
                "custom_cnn_fps":         r.get("custom_cnn_fps", ""),
                "custom_cnn_gpu_mem_mb":  r.get("custom_cnn_gpu_mem_mb", ""),
                "yolo26_mAP50":           r.get("yolo26_mAP50", ""),
                "yolo26_fps":             r.get("yolo26_fps", ""),
                "yolo26_gpu_mem_mb":      r.get("yolo26_gpu_mem_mb", ""),
            })
    print(f"Summary CSV saved to {out_csv}")


if __name__ == "__main__":
    main()
