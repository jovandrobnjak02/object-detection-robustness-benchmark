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
    eps = 1e-6

    pw = box_pred[..., 2].clamp(min=eps)
    ph = box_pred[..., 3].clamp(min=eps)
    gw = box_gt[..., 2].clamp(min=eps)
    gh = box_gt[..., 3].clamp(min=eps)

    px1 = box_pred[..., 0] - pw / 2
    py1 = box_pred[..., 1] - ph / 2
    px2 = box_pred[..., 0] + pw / 2
    py2 = box_pred[..., 1] + ph / 2

    gx1 = box_gt[..., 0] - gw / 2
    gy1 = box_gt[..., 1] - gh / 2
    gx2 = box_gt[..., 0] + gw / 2
    gy2 = box_gt[..., 1] + gh / 2

    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area_p = (px2 - px1) * (py2 - py1)
    area_g = (gx2 - gx1) * (gy2 - gy1)
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
        torch.atan(gw / (gh + eps))
        - torch.atan(pw / (ph + eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou_val + v + eps)

    result = 1 - iou_val + rho2 / c2 + alpha * v
    return torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)



def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = targets * torch.sigmoid(logits) + (1 - targets) * torch.sigmoid(-logits)
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

    best_cls_conf, best_cls = pred[:, :, :, B * 5:].sigmoid().max(dim=3)

    all_dets = []
    for j in range(B):
        base = j * 5
        # apply sigmoid to center offsets (keep them within cell) and exp for sizes
        cx   = (torch.sigmoid(pred[:, :, :, base + 0]) + cols) * cell_size
        cy   = (torch.sigmoid(pred[:, :, :, base + 1]) + rows) * cell_size
        # clip logits before exponentiating to avoid overflow / extreme sizes
        w = torch.exp(torch.clamp(pred[:, :, :, base + 2], -10.0, 10.0)) * cell_size
        h = torch.exp(torch.clamp(pred[:, :, :, base + 3], -10.0, 10.0)) * cell_size
        conf = torch.sigmoid(pred[:, :, :, base + 4])

        # clamp to valid normalized image coords / sizes to avoid numerical explosion
        cx = cx.clamp(0.0, 1.0)
        cy = cy.clamp(0.0, 1.0)
        w = w.clamp(min=1e-6, max=1.0)
        h = h.clamp(min=1e-6, max=1.0)

        dets = torch.stack([
            cx.flatten(1), cy.flatten(1), w.flatten(1), h.flatten(1),
            conf.flatten(1), best_cls_conf.flatten(1), best_cls.float().flatten(1),
        ], dim=2)
        all_dets.append(dets)

    all_dets = torch.cat(all_dets, dim=1)  # (N, S*S*B, 7)

    # Vectorized filtering: mask all batch at once
    batch_idx = torch.arange(N, device=device).view(N, 1).expand(N, all_dets.shape[1])
    conf_mask = all_dets[..., 4] >= conf_thresh
    
    results = []
    for b in range(N):
        if conf_mask[b].any():
            results.append(all_dets[b, conf_mask[b]])
        else:
            results.append(torch.zeros((0, 7), device=device))
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
        coord_mask      = torch.zeros(batch_size, S, S, dtype=torch.bool, device=device)
        best_iou_gt     = torch.zeros(batch_size, S, S, device=device)
        responsible_idx = torch.zeros(batch_size, S, S, dtype=torch.long, device=device)

        valid = targets[:, :, 0] >= 0
        if not valid.any():
            return gt_box, gt_cls, obj_mask, coord_mask, best_iou_gt, responsible_idx

        b_idx, n_idx = valid.nonzero(as_tuple=True)
        gt_valid = targets[b_idx, n_idx]

        cls_ids = gt_valid[:, 0].long()
        cx, cy, w, h = gt_valid[:, 1], gt_valid[:, 2], gt_valid[:, 3], gt_valid[:, 4]

        col_f  = cx / cell_size
        row_f  = cy / cell_size
        cols_c = col_f.long().clamp(0, S - 1)
        rows_c = row_f.long().clamp(0, S - 1)
        cx_rel = col_f - cols_c.float()
        cy_rel = row_f - rows_c.float()

        # --- Center cell assignments: coords + obj + cls ---
        pred_at = predictions[b_idx, rows_c, cols_c, :self.B * 5].reshape(-1, self.B, 5)
        p_cx = (torch.sigmoid(pred_at[:, :, 0]) + cols_c.float().unsqueeze(1)) * cell_size
        p_cy = (torch.sigmoid(pred_at[:, :, 1]) + rows_c.float().unsqueeze(1)) * cell_size
        # clip regression logits before exp to keep sizes reasonable
        p_w = torch.exp(torch.clamp(pred_at[:, :, 2], -10.0, 10.0)) * cell_size
        p_h = torch.exp(torch.clamp(pred_at[:, :, 3], -10.0, 10.0)) * cell_size

        # clamp predicted sizes/centers to valid range
        p_cx = p_cx.clamp(0.0, 1.0)
        p_cy = p_cy.clamp(0.0, 1.0)
        p_w = p_w.clamp(min=1e-6, max=1.0)
        p_h = p_h.clamp(min=1e-6, max=1.0)

        ious = iou(
            torch.stack([p_cx, p_cy, p_w, p_h], dim=-1),
            torch.stack([cx, cy, w, h], dim=1).unsqueeze(1).expand(-1, self.B, -1),
        )
        best_js       = ious.argmax(dim=1)
        best_iou_vals = ious.detach().max(dim=1).values

        flat_center = b_idx * (S * S) + rows_c * S + cols_c
        ones_c = torch.ones(len(flat_center), dtype=torch.bool, device=device)

        obj_mask.view(-1).scatter_(0, flat_center, ones_c)
        coord_mask.view(-1).scatter_(0, flat_center, ones_c)
        best_iou_gt.view(-1).scatter_(0, flat_center, best_iou_vals)
        responsible_idx.view(-1).scatter_(0, flat_center, best_js)

        gt_box.view(-1, 4).scatter_(
            0, flat_center.unsqueeze(1).expand(-1, 4),
            torch.stack([cx_rel, cy_rel, w, h], dim=1),
        )
        gt_cls.view(-1, self.C).scatter_(
            0, flat_center.unsqueeze(1).expand(-1, self.C),
            F.one_hot(cls_ids, self.C).float(),
        )

        # --- Neighbor assignments: obj + cls only (no coords) ---
        frac_x = col_f - cols_c.float()
        frac_y = row_f - rows_c.float()
        dx = torch.where(frac_x >= 0.5, torch.ones_like(cols_c), -torch.ones_like(cols_c))
        dy = torch.where(frac_y >= 0.5, torch.ones_like(rows_c), -torch.ones_like(rows_c))

        for nb_col, nb_row in [(cols_c + dx, rows_c), (cols_c, rows_c + dy)]:
            valid_nb = (nb_col >= 0) & (nb_col < S) & (nb_row >= 0) & (nb_row < S)
            if not valid_nb.any():
                continue
            b_nb   = b_idx[valid_nb]
            col_nb = nb_col[valid_nb]
            row_nb = nb_row[valid_nb]
            cls_nb = cls_ids[valid_nb]
            js_nb  = best_js[valid_nb]

            flat_nb = b_nb * (S * S) + row_nb * S + col_nb
            obj_mask.view(-1).scatter_(0, flat_nb,
                torch.ones(len(flat_nb), dtype=torch.bool, device=device))
            responsible_idx.view(-1).scatter_(0, flat_nb, js_nb)
            gt_cls.view(-1, self.C).scatter_(
                0, flat_nb.unsqueeze(1).expand(-1, self.C),
                F.one_hot(cls_nb, self.C).float(),
            )

        return gt_box, gt_cls, obj_mask, coord_mask, best_iou_gt, responsible_idx

    def _forward_single(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        S = predictions.shape[1]
        batch_size = predictions.shape[0]

        gt_box, gt_cls, obj_mask, coord_mask, best_iou_gt, responsible_idx = self._build_targets(
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
        loss_coord = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype, requires_grad=True)
        
        if coord_mask.any():
            dev = predictions.device
            cols = torch.arange(S, device=dev).float().view(1, 1, S).expand(batch_size, S, S)
            rows = torch.arange(S, device=dev).float().view(1, S, 1).expand(batch_size, S, S)

            p = pred_box_resp[coord_mask]   # (N_center, 4)
            g = gt_box[coord_mask]          # (N_center, 4)
            c = cols[coord_mask]            # (N_center,)
            r = rows[coord_mask]            # (N_center,)

            pred_abs = torch.stack([
                    ((torch.sigmoid(p[:, 0]) + c) * cell_size).clamp(0.0, 1.0),
                    ((torch.sigmoid(p[:, 1]) + r) * cell_size).clamp(0.0, 1.0),
                    (torch.exp(torch.clamp(p[:, 2], -10.0, 10.0)) * cell_size).clamp(min=1e-6, max=1.0),
                    (torch.exp(torch.clamp(p[:, 3], -10.0, 10.0)) * cell_size).clamp(min=1e-6, max=1.0),
                ], dim=-1)
            gt_abs = torch.stack([
                    (g[:, 0] + c) * cell_size,
                    (g[:, 1] + r) * cell_size,
                    g[:, 2],
                    g[:, 3],
                ], dim=-1)
            loss_coord = ciou(pred_abs, gt_abs).sum()

        # Confidence loss (use predicted probabilities)
        pred_conf_resp_sig = torch.sigmoid(pred_conf_resp)
        pred_conf_all_sig = torch.sigmoid(pred_conf_all)
        loss_obj = F.binary_cross_entropy_with_logits(
            pred_conf_resp, obj, reduction="none"
        ).sum()

        resp_onehot = torch.zeros_like(pred_conf_all)
        resp_onehot.scatter_(3, responsible_idx.unsqueeze(-1), 1.0)
        noobj_mask = (~obj_mask).unsqueeze(-1) | (resp_onehot == 0)
        loss_noobj = (noobj_mask.float() * pred_conf_all_sig ** 2).sum()

        loss_cls = (obj4 * focal_loss(pred_cls, gt_cls)).sum()

        total = (
            self.lambda_coord * loss_coord
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_cls
        ) / batch_size

        return total

    def forward(self, predictions, targets: torch.Tensor) -> torch.Tensor:
        predictions = [p.float() for p in predictions]
        losses = []
        
        for i, p in enumerate(predictions):
            level_targets = targets.clone()
            # Calculate areas in pixel-space for easier thresholding
            # Assuming targets[..., 3:5] are normalized (0-1)
            areas = (level_targets[..., 3] * 640) * (level_targets[..., 4] * 640)
            
            if i == 0:   # P3: Small (< 32x32 pixels)
                mask = (areas < 1024)
            elif i == 1: # P4: Medium (32x32 to 96x96)
                mask = (areas >= 1024) & (areas < 9216)
            else:        # P5: Large (> 96x96)
                mask = (areas >= 9216)
            
            level_targets[~mask] = -1
            l = self._forward_single(p, level_targets)
            if torch.isfinite(l):
                losses.append(l)
                
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=targets.device)
