import torch
import torch.nn as nn


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
        (px2 - px1) * (py2 - py1)
        + (gx2 - gx1) * (gy2 - gy1)
        - inter
    )
    return inter / (union + 1e-6)


class CustomCNNLoss(nn.Module):
    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 3,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
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

            for obj in gt:
                cls_id = int(obj[0].item())
                cx, cy, w, h = obj[1].item(), obj[2].item(), obj[3].item(), obj[4].item()

                col = min(int(cx / cell_size), self.S - 1)
                row = min(int(cy / cell_size), self.S - 1)

                cx_rel = cx / cell_size - col
                cy_rel = cy / cell_size - row

                gt_abs = torch.tensor([cx, cy, w, h], device=device)

                pred_boxes = predictions[b, row, col, :self.B * 5].reshape(self.B, 5)[:, :4]
                ious = iou(pred_boxes, gt_abs.unsqueeze(0).expand(self.B, -1))
                best_j = int(ious.argmax().item())

                if not obj_mask[b, row, col] or ious[best_j] > best_iou_gt[b, row, col]:
                    obj_mask[b, row, col]        = True
                    gt_box[b, row, col]          = torch.tensor([cx_rel, cy_rel, w, h], device=device)
                    best_iou_gt[b, row, col]     = ious[best_j]
                    responsible_idx[b, row, col] = best_j

                gt_cls[b, row, col, cls_id] = 1.0

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

        pred_conf_all = pred_boxes_all[..., 4]

        obj  = obj_mask.float()
        obj4 = obj.unsqueeze(-1)

        loss_xy = (obj * (
            (pred_box_resp[..., 0] - gt_box[..., 0]) ** 2
            + (pred_box_resp[..., 1] - gt_box[..., 1]) ** 2
        )).sum()

        loss_wh = (obj * (
            (pred_box_resp[..., 2].clamp(min=0).sqrt() - gt_box[..., 2].sqrt()) ** 2
            + (pred_box_resp[..., 3].clamp(min=0).sqrt() - gt_box[..., 3].sqrt()) ** 2
        )).sum()

        loss_coord = loss_xy + loss_wh

        loss_obj = (obj * (pred_conf_resp - best_iou_gt) ** 2).sum()

        resp_onehot = torch.zeros_like(pred_conf_all)
        resp_onehot.scatter_(3, responsible_idx.unsqueeze(-1), 1.0)
        noobj_mask = (~obj_mask).unsqueeze(-1) | (resp_onehot == 0)
        loss_noobj = (noobj_mask.float() * pred_conf_all ** 2).sum()

        loss_cls = (obj4 * (pred_cls - gt_cls) ** 2).sum()

        total = (
            self.lambda_coord * loss_coord
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_cls
        ) / batch_size

        return total
