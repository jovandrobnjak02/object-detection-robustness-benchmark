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
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy(p, targets, reduction="none")
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

    return _decode_single(predictions, predictions.shape[1], B, C, conf_thresh)


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

        # Fully vectorized across entire batch — no Python loop over batch items
        valid = targets[:, :, 0] >= 0  # (batch, max_boxes)
        if valid.any():
            b_idx, n_idx = valid.nonzero(as_tuple=True)  # (total_gts,)
            gt_valid = targets[b_idx, n_idx]              # (total_gts, 5)

            cls_ids = gt_valid[:, 0].long()
            cx, cy, w, h = gt_valid[:, 1], gt_valid[:, 2], gt_valid[:, 3], gt_valid[:, 4]

            col_f = cx / cell_size
            row_f = cy / cell_size
            cols  = col_f.long().clamp(0, S - 1)
            rows  = row_f.long().clamp(0, S - 1)
            cx_rel = col_f - cols.float()
            cy_rel = row_f - rows.float()

            # Get predictions at all GT cell locations: (total_gts, B, 5)
            pred_at = predictions[b_idx, rows, cols, :self.B * 5].reshape(-1, self.B, 5)

            p_cx = (pred_at[:, :, 0] + cols.float().unsqueeze(1)) * cell_size
            p_cy = (pred_at[:, :, 1] + rows.float().unsqueeze(1)) * cell_size

            ious = iou(
                torch.stack([p_cx, p_cy, pred_at[:, :, 2], pred_at[:, :, 3]], dim=-1),
                torch.stack([cx, cy, w, h], dim=1).unsqueeze(1).expand(-1, self.B, -1),
            )  # (total_gts, B)

            best_js       = ious.argmax(dim=1)
            best_iou_vals = ious.max(dim=1).values

            # 3D flat index: batch_item * S*S + row * S + col
            flat_3d = b_idx * (S * S) + rows * S + cols

            obj_mask.view(-1).scatter_(0, flat_3d,
                torch.ones(len(flat_3d), dtype=torch.bool, device=device))
            best_iou_gt.view(-1).scatter_(0, flat_3d, best_iou_vals)
            responsible_idx.view(-1).scatter_(0, flat_3d, best_js)

            gt_box.view(-1, 4).scatter_(
                0, flat_3d.unsqueeze(1).expand(-1, 4),
                torch.stack([cx_rel, cy_rel, w, h], dim=1),
            )
            gt_cls.view(-1, self.C).scatter_(
                0, flat_3d.unsqueeze(1).expand(-1, self.C),
                F.one_hot(cls_ids, self.C).float(),
            )

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
