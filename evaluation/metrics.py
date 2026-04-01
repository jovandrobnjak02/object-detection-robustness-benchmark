import numpy as np


def box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
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
    all_preds: list[dict],
    all_gts:   list[dict],
    num_classes: int,
    iou_thresh: float = 0.5,
) -> tuple[float, list[float], list[float], list[float]]:
    # Pre-convert GT boxes to numpy arrays once, keyed by (img_id, cls)
    gt_by_img_cls: dict[tuple, np.ndarray] = {}
    _tmp: dict[tuple, list] = {}
    for gt in all_gts:
        img_id = gt["img_id"]
        for box in gt["boxes"]:
            cls = int(box[4])
            _tmp.setdefault((img_id, cls), []).append(box[:4])
    gt_by_img_cls = {k: np.array(v) for k, v in _tmp.items()}

    aps, precisions_out, recalls_out = [], [], []

    for cls in range(num_classes):
        preds_cls = []
        for pred in all_preds:
            img_id = pred["img_id"]
            for box in pred["boxes"]:
                if int(box[5]) == cls:
                    preds_cls.append((box[4], img_id, box[:4]))

        preds_cls.sort(key=lambda x: -x[0])

        total_gt = sum(len(v) for (iid, c), v in gt_by_img_cls.items() if c == cls)
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
            gt_boxes = gt_by_img_cls.get(key, None)
            if gt_boxes is None:
                fp[i] = 1
                continue

            ious = compute_iou_np(np.array(pred_box), gt_boxes)
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
    all_gts:   list[dict],
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

        for b in sorted(pred["boxes"], key=lambda b: -b[4]):
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

        for i, g in enumerate(gt_boxes):
            if i not in matched_gt:
                matrix[bg][int(g[4])] += 1

    return matrix
