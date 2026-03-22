import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.bdd100k_dataset import BDD100KDataset
from models.custom_cnn import CustomCNN, decode_predictions, nms

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 448
CONF_THRESH  = 0.25
NMS_THRESH   = 0.45
IOU_THRESH   = 0.5   # mAP@50

CUSTOM_CNN_CKPT = Path("checkpoints/custom_cnn/bdd100k_custom_cnn_best.pt")
YOLO26_CKPT     = Path("checkpoints/yolo26/bdd100k/weights/best.pt")

SPLITS = ["clear_day/val", "rainy", "snowy", "night", "overcast"]

CLASS_NAMES = [
    "car", "person", "traffic sign", "traffic light",
    "truck", "bus", "bike", "rider", "motor", "train",
]
NUM_CLASSES = len(CLASS_NAMES)

RESULTS_DIR = Path("results/metrics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] → [x1, y1, x2, y2]"""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def compute_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
    area_box   = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-6
    return inter / union


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map(
    all_preds: list[dict],   # [{"boxes": [N,6 xyxy+conf+cls], "img_id": int}]
    all_gts:   list[dict],   # [{"boxes": [M,5 xyxy+cls],       "img_id": int}]
    num_classes: int,
    iou_thresh: float = 0.5,
) -> tuple[float, list[float], list[float], list[float]]:
    gt_by_img_cls: dict[tuple, list] = {}
    for gt in all_gts:
        img_id = gt["img_id"]
        for box in gt["boxes"]:
            cls = int(box[4])
            key = (img_id, cls)
            gt_by_img_cls.setdefault(key, []).append(box[:4])

    aps = []
    precisions_out = []
    recalls_out = []

    for cls in range(num_classes):
        preds_cls = []
        for pred in all_preds:
            img_id = pred["img_id"]
            for box in pred["boxes"]:
                if int(box[5]) == cls:
                    preds_cls.append((box[4], img_id, box[:4]))

        preds_cls.sort(key=lambda x: -x[0])

        total_gt = sum(
            len(v) for (iid, c), v in gt_by_img_cls.items() if c == cls
        )
        if total_gt == 0:
            aps.append(float("nan"))
            precisions_out.append(float("nan"))
            recalls_out.append(float("nan"))
            continue

        tp = np.zeros(len(preds_cls))
        fp = np.zeros(len(preds_cls))
        matched: dict[tuple, set] = {}

        for i, (conf, img_id, pred_box) in enumerate(preds_cls):
            key = (img_id, cls)
            gt_boxes = gt_by_img_cls.get(key, [])

            if not gt_boxes:
                fp[i] = 1
                continue

            gt_arr = np.array(gt_boxes)
            ious = compute_iou_np(np.array(pred_box), gt_arr)
            best_iou_idx = int(np.argmax(ious))
            best_iou = ious[best_iou_idx]

            already_matched = matched.setdefault(key, set())
            if best_iou >= iou_thresh and best_iou_idx not in already_matched:
                tp[i] = 1
                already_matched.add(best_iou_idx)
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls    = tp_cum / (total_gt + 1e-6)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        aps.append(compute_ap(recalls, precisions))

        # Best-F1 operating point
        if len(precisions) > 0:
            f1 = 2 * precisions * recalls / (precisions + recalls + 1e-6)
            best_idx = int(np.argmax(f1))
            precisions_out.append(float(precisions[best_idx]))
            recalls_out.append(float(recalls[best_idx]))
        else:
            precisions_out.append(0.0)
            recalls_out.append(0.0)

    valid_aps = [a for a in aps if not np.isnan(a)]
    map_val = float(np.mean(valid_aps)) if valid_aps else 0.0
    return map_val, aps, precisions_out, recalls_out


def compute_confusion_matrix(
    all_preds: list[dict],
    all_gts: list[dict],
    num_classes: int,
    iou_thresh: float = 0.5,
) -> np.ndarray:
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    bg = num_classes

    gt_by_img: dict[int, list] = {}
    for gt in all_gts:
        gt_by_img[gt["img_id"]] = list(gt["boxes"])

    for pred in all_preds:
        img_id = pred["img_id"]
        gt_boxes = list(gt_by_img.get(img_id, []))
        matched_gt: set[int] = set()

        det_boxes = sorted(pred["boxes"], key=lambda b: -b[4])

        for b in det_boxes:
            pred_cls = int(b[5])
            pred_box = np.array(b[:4])

            if not gt_boxes:
                matrix[pred_cls][bg] += 1
                continue

            gt_arr = np.array([g[:4] for g in gt_boxes])
            ious = compute_iou_np(pred_box, gt_arr)
            best_idx = int(np.argmax(ious))
            best_iou = ious[best_idx]

            if best_iou >= iou_thresh and best_idx not in matched_gt:
                gt_cls = int(gt_boxes[best_idx][4])
                matrix[pred_cls][gt_cls] += 1
                matched_gt.add(best_idx)
            else:
                matrix[pred_cls][bg] += 1

        # Unmatched GTs are false negatives
        for i, g in enumerate(gt_boxes):
            if i not in matched_gt:
                gt_cls = int(g[4])
                matrix[bg][gt_cls] += 1

    return matrix



def load_gt(dataset: BDD100KDataset) -> list[dict]:
    gts = []
    for img_id, stem in enumerate(dataset.samples):
        labels = dataset._label_cache[stem]  # [N, 5] cls cx cy w h
        if labels.shape[0] == 0:
            gts.append({"img_id": img_id, "boxes": []})
            continue
        boxes_xyxy = box_cxcywh_to_xyxy(labels[:, 1:].numpy())
        cls = labels[:, 0:1].numpy()
        boxes = np.concatenate([boxes_xyxy, cls], axis=1)  # [N, 5] xyxy+cls
        gts.append({"img_id": img_id, "boxes": boxes})
    return gts


def load_custom_cnn() -> torch.nn.Module:
    model = CustomCNN(use_batchnorm=True, use_residual=True, C=NUM_CLASSES)
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
            img = dataset._load_image(stem)  # HWC uint8
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
                # [cx, cy, w, h, conf, cls_conf, cls_id]
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
        img = dataset._load_image(stem)  # HWC RGB uint8, already letterboxed

        t0 = time.perf_counter()
        results = model.predict(img, imgsz=IMG_SIZE, conf=CONF_THRESH,
                                iou=NMS_THRESH, verbose=False)
        torch.cuda.synchronize()
        total_time += time.perf_counter() - t0

        boxes = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxyn   = r.boxes.xyxyn.cpu().numpy()   # normalized xyxy
            confs   = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for k in range(len(xyxyn)):
                boxes.append([*xyxyn[k], confs[k], cls_ids[k]])

        all_preds.append({"img_id": img_id, "boxes": boxes})

    fps = len(dataset.samples) / total_time
    gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
    return all_preds, fps, gpu_mem_mb


def evaluate_split(
    split: str,
    custom_cnn_model,
    yolo26_model,
) -> dict:
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

    results["custom_cnn_mAP50"]     = round(cnn_map * 100, 2)
    results["custom_cnn_fps"]       = round(cnn_fps, 1)
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

        results["yolo26_mAP50"]     = round(yolo_map * 100, 2)
        results["yolo26_fps"]       = round(yolo_fps, 1)
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

    # Save JSON
    out_json = RESULTS_DIR / "bdd100k_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_json}")

    # Print summary table
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

    # Save CSV summary
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
