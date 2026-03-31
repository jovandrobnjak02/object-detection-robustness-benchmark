import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms


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
    area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    area_g = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = area_p + area_g - inter
    iou_val = inter / (union + eps)

    rho2 = (box_pred[..., 0] - box_gt[..., 0]) ** 2 + \
           (box_pred[..., 1] - box_gt[..., 1]) ** 2
    enc_x1 = torch.min(px1, gx1)
    enc_y1 = torch.min(py1, gy1)
    enc_x2 = torch.max(px2, gx2)
    enc_y2 = torch.max(py2, gy2)
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + eps

    v = (4 / (math.pi ** 2)) * (
        torch.atan(box_gt[..., 2] / (box_gt[..., 3] + eps))
        - torch.atan(box_pred[..., 2] / (box_pred[..., 3] + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou_val + v + eps)

    return 1 - iou_val + rho2 / c2 + alpha * v


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = targets * p + (1 - targets) * (1 - p)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    return alpha_t * (1 - p_t) ** gamma * bce


def _decode_single(
    pred: torch.Tensor,
    S: int,
    B: int,
    C: int,
    conf_thresh: float,
) -> list[torch.Tensor]:
    N = pred.shape[0]
    device = pred.device
    cell_size = 1.0 / S

    rows = torch.arange(S, device=device).float().view(1, S, 1).expand(N, S, S)
    cols = torch.arange(S, device=device).float().view(1, 1, S).expand(N, S, S)

    # Class scores — batched over N
    best_cls_conf, best_cls = pred[:, :, :, B * 5:].sigmoid().max(dim=3)  # (N, S, S)

    all_dets = []
    for j in range(B):
        base = j * 5
        cx   = (pred[:, :, :, base + 0] + cols) * cell_size  # (N, S, S)
        cy   = (pred[:, :, :, base + 1] + rows) * cell_size
        w    = pred[:, :, :, base + 2].abs()
        h    = pred[:, :, :, base + 3].abs()
        conf = pred[:, :, :, base + 4]

        dets = torch.stack([
            cx.flatten(1), cy.flatten(1), w.flatten(1), h.flatten(1),
            conf.flatten(1), best_cls_conf.flatten(1), best_cls.float().flatten(1),
        ], dim=2)  # (N, S*S, 7)
        all_dets.append(dets)

    all_dets = torch.cat(all_dets, dim=1)  # (N, S*S*B, 7)

    results = []
    for b in range(N):
        mask = all_dets[b, :, 4] >= conf_thresh
        results.append(all_dets[b][mask] if mask.any() else torch.zeros((0, 7), device=device))
    return results


def decode_predictions(
    predictions,
    S: int = 14,
    B: int = 2,
    C: int = 10,
    conf_thresh: float = 0.25,
) -> list[torch.Tensor]:
    if isinstance(predictions, list):
        device = predictions[0].device
        scale_results = [
            _decode_single(p, p.shape[1], B, C, conf_thresh) for p in predictions
        ]
        batch_size = len(scale_results[0])
        merged = []
        for b in range(batch_size):
            dets = [s[b] for s in scale_results if s[b].shape[0] > 0]
            if dets:
                merged.append(torch.cat(dets, dim=0))
            else:
                merged.append(torch.zeros((0, 7), device=device))
        return merged

    return _decode_single(predictions, S, B, C, conf_thresh)


def nms(detections: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    if detections.shape[0] == 0:
        return detections

    cx, cy, w, h = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
    boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    scores  = detections[:, 4]
    cls_ids = detections[:, 6].long()

    keep = batched_nms(boxes_xyxy, scores, cls_ids, iou_thresh)
    return detections[keep]


class CustomCNNLoss(nn.Module):
    def __init__(
        self,
        B: int = 2,
        C: int = 10,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.1,
    ):
        super().__init__()
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def _build_targets(self, predictions: torch.Tensor, targets: torch.Tensor, S: int):
        batch_size = predictions.shape[0]
        device = predictions.device
        cell_size = 1.0 / S

        gt_box          = torch.zeros(batch_size, S, S, 4, device=device)
        gt_cls          = torch.zeros(batch_size, S, S, self.C, device=device)
        obj_mask        = torch.zeros(batch_size, S, S, dtype=torch.bool, device=device)
        best_iou_gt     = torch.zeros(batch_size, S, S, device=device)
        responsible_idx = torch.zeros(batch_size, S, S, dtype=torch.long, device=device)

        for b in range(batch_size):
            gt = targets[b]
            gt = gt[gt[:, 0] >= 0]
            if gt.numel() == 0:
                continue

            cls_ids = gt[:, 0].long()
            cx, cy, w, h = gt[:, 1], gt[:, 2], gt[:, 3], gt[:, 4]

            cols = (cx / cell_size).long().clamp(max=S - 1)
            rows = (cy / cell_size).long().clamp(max=S - 1)
            cx_rel = cx / cell_size - cols.float()
            cy_rel = cy / cell_size - rows.float()

            for idx in range(gt.shape[0]):
                r, c = rows[idx].item(), cols[idx].item()
                cls_id = cls_ids[idx].item()

                pred_raw = predictions[b, r, c, :self.B * 5].reshape(self.B, 5)
                pred_cx = (pred_raw[:, 0] + c) * cell_size
                pred_cy = (pred_raw[:, 1] + r) * cell_size
                decoded = torch.stack([pred_cx, pred_cy, pred_raw[:, 2], pred_raw[:, 3]], dim=-1)

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

    def _forward_single(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        S = predictions.shape[1]
        batch_size = predictions.shape[0]

        gt_box, gt_cls, obj_mask, best_iou_gt, responsible_idx = self._build_targets(
            predictions, targets, S
        )

        pred_boxes_all = predictions[..., :self.B * 5].reshape(batch_size, S, S, self.B, 5)
        pred_cls = predictions[..., self.B * 5:]

        resp_idx = responsible_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 5)
        pred_resp = pred_boxes_all.gather(3, resp_idx).squeeze(3)
        pred_box_resp  = pred_resp[..., :4]
        pred_conf_resp = pred_resp[..., 4]
        pred_conf_all  = pred_boxes_all[..., 4]

        obj  = obj_mask.float()
        obj4 = obj.unsqueeze(-1)

        # CIoU localization loss
        cell_size = 1.0 / S
        dev = predictions.device
        cols = torch.arange(S, device=dev).float().view(1, 1, S).expand(batch_size, S, S)
        rows = torch.arange(S, device=dev).float().view(1, S, 1).expand(batch_size, S, S)

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

        # Confidence loss
        loss_obj = (obj * (pred_conf_resp - best_iou_gt) ** 2).sum()

        resp_onehot = torch.zeros_like(pred_conf_all)
        resp_onehot.scatter_(3, responsible_idx.unsqueeze(-1), 1.0)
        noobj_mask = (~obj_mask).unsqueeze(-1) | (resp_onehot == 0)
        loss_noobj = (noobj_mask.float() * pred_conf_all ** 2).sum()

        # Focal loss for classification (only at obj cells)
        loss_cls = (obj4 * focal_loss(pred_cls, gt_cls)).sum()

        total = (
            self.lambda_coord * loss_coord
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_cls
        ) / batch_size

        return total

    def forward(self, predictions, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(predictions, list):
            predictions = [p.float() for p in predictions]
            return sum(self._forward_single(p, targets) for p in predictions) / len(predictions)
        return self._forward_single(predictions.float(), targets)
