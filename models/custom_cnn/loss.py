import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def iou(box_pred: torch.Tensor, box_gt: torch.Tensor) -> torch.Tensor:
    px1 = box_pred[..., 0] - box_pred[..., 2] / 2
    py1 = box_pred[..., 1] - box_pred[..., 3] / 2
    px2 = box_pred[..., 0] + box_pred[..., 2] / 2
    py2 = box_pred[..., 1] + box_pred[..., 3] / 2

    gx1 = box_gt[..., 0] - box_gt[..., 2] / 2
    gy1 = box_gt[..., 1] - box_gt[..., 3] / 2
    gx2 = box_gt[..., 0] + box_gt[..., 2] / 2
    gy2 = box_gt[..., 1] + box_gt[..., 3] / 2

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = (
        (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        + (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
        - inter
    )
    return inter / (union + 1e-6)


def ciou(box_pred: torch.Tensor, box_gt: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    # Convert to corners
    px1 = box_pred[..., 0] - box_pred[..., 2] / 2
    py1 = box_pred[..., 1] - box_pred[..., 3] / 2
    px2 = box_pred[..., 0] + box_pred[..., 2] / 2
    py2 = box_pred[..., 1] + box_pred[..., 3] / 2

    gx1 = box_gt[..., 0] - box_gt[..., 2] / 2
    gy1 = box_gt[..., 1] - box_gt[..., 3] / 2
    gx2 = box_gt[..., 0] + box_gt[..., 2] / 2
    gy2 = box_gt[..., 1] + box_gt[..., 3] / 2

    # IoU
    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    area_g = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = area_p + area_g - inter
    iou_val = inter / (union + eps)

    # Center distance penalty
    rho2 = (box_pred[..., 0] - box_gt[..., 0]) ** 2 + \
           (box_pred[..., 1] - box_gt[..., 1]) ** 2
    # Diagonal of smallest enclosing box
    enc_x1 = torch.min(px1, gx1)
    enc_y1 = torch.min(py1, gy1)
    enc_x2 = torch.max(px2, gx2)
    enc_y2 = torch.max(py2, gy2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    # Aspect ratio penalty
    v = (4 / (math.pi ** 2)) * (
        torch.atan(box_gt[..., 2] / (box_gt[..., 3] + eps))
        - torch.atan(box_pred[..., 2] / (box_pred[..., 3] + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou_val + v + eps)

    return 1 - iou_val + rho2 / c2 + alpha * v


def decode_predictions(
    predictions: torch.Tensor,
    S: int = 14,
    B: int = 2,
    C: int = 10,
    conf_thresh: float = 0.25,
) -> list[torch.Tensor]:
    batch_size = predictions.shape[0]
    cell_size = 1.0 / S
    results = []

    for b in range(batch_size):
        detections = []
        pred = predictions[b]  # (S, S, B*5+C)

        for row in range(S):
            for col in range(S):
                class_scores = pred[row, col, B * 5:].sigmoid()  # logits → probs
                best_cls_conf, best_cls = class_scores.max(0)

                for j in range(B):
                    base = j * 5
                    conf = pred[row, col, base + 4]

                    if conf < conf_thresh:
                        continue

                    cx = (pred[row, col, base + 0] + col) * cell_size
                    cy = (pred[row, col, base + 1] + row) * cell_size
                    w  = pred[row, col, base + 2]
                    h  = pred[row, col, base + 3]

                    detections.append(torch.stack([
                        cx, cy, w.abs(), h.abs(), conf, best_cls_conf, best_cls.float()
                    ]))

        if detections:
            results.append(torch.stack(detections))
        else:
            results.append(torch.zeros((0, 7), device=predictions.device))

    return results


def nms(detections: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    if detections.shape[0] == 0:
        return detections

    keep = []
    order = detections[:, 4].argsort(descending=True)
    detections = detections[order]

    while detections.shape[0] > 0:
        keep.append(detections[0])
        if detections.shape[0] == 1:
            break

        ious = iou(
            detections[0, :4].unsqueeze(0).expand(detections.shape[0] - 1, -1),
            detections[1:, :4],
        )
        same_class = detections[1:, 6] == detections[0, 6]
        suppress = (ious > iou_thresh) & same_class
        detections = detections[1:][~suppress]

    return torch.stack(keep)


class CustomCNNLoss(nn.Module):
    def __init__(
        self,
        S: int = 14,
        B: int = 2,
        C: int = 10,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.1,
    ):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def _build_targets(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):

        batch_size = predictions.shape[0]
        device = predictions.device
        cell_size = 1.0 / self.S

        gt_box          = torch.zeros(batch_size, self.S, self.S, 4, device=device)
        gt_cls          = torch.zeros(batch_size, self.S, self.S, self.C, device=device)
        obj_mask        = torch.zeros(batch_size, self.S, self.S, dtype=torch.bool, device=device)
        best_iou_gt     = torch.zeros(batch_size, self.S, self.S, device=device)
        responsible_idx = torch.zeros(batch_size, self.S, self.S, dtype=torch.long, device=device)

        for b in range(batch_size):
            gt = targets[b]
            gt = gt[gt[:, 0] >= 0]
            if gt.numel() == 0:
                continue

            cls_ids = gt[:, 0].long()
            cx, cy, w, h = gt[:, 1], gt[:, 2], gt[:, 3], gt[:, 4]

            cols = (cx / cell_size).long().clamp(max=self.S - 1)
            rows = (cy / cell_size).long().clamp(max=self.S - 1)
            cx_rel = cx / cell_size - cols.float()
            cy_rel = cy / cell_size - rows.float()

            for idx in range(gt.shape[0]):
                r, c = rows[idx].item(), cols[idx].item()
                cls_id = cls_ids[idx].item()

                pred_raw = predictions[b, r, c, :self.B * 5].reshape(self.B, 5)
                pred_cx = (pred_raw[:, 0] + c) * cell_size
                pred_cy = (pred_raw[:, 1] + r) * cell_size 
                pred_w  = pred_raw[:, 2]
                pred_h  = pred_raw[:, 3]
                decoded = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)

                gt_abs = torch.stack([cx[idx], cy[idx], w[idx], h[idx]])
                ious = iou(decoded, gt_abs.unsqueeze(0).expand(self.B, -1))
                best_j = int(ious.argmax().item())

                if not obj_mask[b, r, c] or ious[best_j] > best_iou_gt[b, r, c]:
                    obj_mask[b, r, c]        = True
                    gt_box[b, r, c]          = torch.stack([cx_rel[idx], cy_rel[idx], w[idx], h[idx]])
                    best_iou_gt[b, r, c]     = ious[best_j]
                    responsible_idx[b, r, c] = best_j

                gt_cls[b, r, c, cls_id] = 1.0

        return gt_box, gt_cls, obj_mask, best_iou_gt, responsible_idx

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = predictions.shape[0]

        gt_box, gt_cls, obj_mask, best_iou_gt, responsible_idx = self._build_targets(
            predictions, targets
        )

        pred_boxes_all = predictions[..., : self.B * 5].reshape(
            batch_size, self.S, self.S, self.B, 5
        )

        pred_cls = predictions[..., self.B * 5 :]

        resp_idx = responsible_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 5)
        pred_resp = pred_boxes_all.gather(3, resp_idx).squeeze(3)
        pred_box_resp  = pred_resp[..., :4]
        pred_conf_resp = pred_resp[..., 4]
        pred_conf_all  = pred_boxes_all[..., 4]

        obj  = obj_mask.float()
        obj4 = obj.unsqueeze(-1)

        # --- CIoU localization loss (replaces MSE xy + sqrt wh) ---
        cell_size = 1.0 / self.S
        dev = predictions.device
        # Decode predicted boxes to absolute coords for CIoU
        cols = torch.arange(self.S, device=dev).float().view(1, 1, self.S).expand(batch_size, self.S, self.S)
        rows = torch.arange(self.S, device=dev).float().view(1, self.S, 1).expand(batch_size, self.S, self.S)
        pred_abs = torch.stack([
            (pred_box_resp[..., 0] + cols) * cell_size,
            (pred_box_resp[..., 1] + rows) * cell_size,
            pred_box_resp[..., 2].abs(),
            pred_box_resp[..., 3].abs(),
        ], dim=-1)
        gt_abs = torch.stack([
            (gt_box[..., 0] + cols) * cell_size,
            (gt_box[..., 1] + rows) * cell_size,
            gt_box[..., 2],
            gt_box[..., 3],
        ], dim=-1)
        loss_coord = (obj * ciou(pred_abs, gt_abs)).sum()

        # --- Confidence loss (MSE) ---
        loss_obj = (obj * (pred_conf_resp - best_iou_gt) ** 2).sum()

        resp_onehot = torch.zeros_like(pred_conf_all)
        resp_onehot.scatter_(3, responsible_idx.unsqueeze(-1), 1.0)
        noobj_mask = (~obj_mask).unsqueeze(-1) | (resp_onehot == 0)
        loss_noobj = (noobj_mask.float() * pred_conf_all ** 2).sum()

        # --- Classification loss (BCE with logits — numerically stable) ---
        loss_cls = (obj4 * F.binary_cross_entropy_with_logits(
            pred_cls, gt_cls, reduction='none'
        )).sum()

        total = (
            self.lambda_coord * loss_coord
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_cls
        ) / batch_size

        return total
