import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torchvision.transforms.functional as TF
from datasets.bdd100k_dataset import BDD100KDataset, get_dataloader
from models.custom_cnn import CustomCNN, CustomCNNLoss, decode_predictions, nms
from evaluation.metrics import compute_map, box_cxcywh_to_xyxy

EPOCHS         = 200
WARMUP_EPOCHS  = 3
BATCH_SIZE     = 32
NUM_WORKERS    = 4
IMG_SIZE       = 640
LR             = 1e-3
NUM_CLASSES    = 10
MAP_EVAL_EVERY = 5     # compute val mAP every N epochs
CONF_THRESH    = 0.001  # low threshold for mAP evaluation (captures all detections)
NMS_THRESH     = 0.45
ACCUM_STEPS    = 2

CHECKPOINTS = Path(__file__).parent.parent / "checkpoints" / "custom_cnn"
LOGS        = Path(__file__).parent.parent / "results" / "logs"


def run_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps=ACCUM_STEPS):
    model.train()
    total_loss = 0.0
    n_finite = 0

    optimizer.zero_grad(set_to_none=True)
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda"):
            preds = model(images)
            loss  = criterion(preds, labels)

        # scale loss down for gradient accumulation
        loss_to_back = loss / float(accum_steps)
        scaler.scale(loss_to_back).backward()

        # perform optimizer step every accum_steps
        if (i + 1) % accum_steps == 0:
            # unscale before clipping
            try:
                scaler.unscale_(optimizer)
            except Exception:
                pass
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if torch.isfinite(loss):
            total_loss += loss.item()
            n_finite += 1

    # final step if there are leftover gradients
    if len(loader) % accum_steps != 0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isfinite(total_norm):
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_finite, 1)


@torch.no_grad()
def run_val(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda"):
            preds = model(images)
            total_loss += criterion(preds, labels).item()
    return total_loss / len(loader)


@torch.no_grad()
def compute_val_map(model, val_dataset: BDD100KDataset, device: torch.device) -> float:
    model.eval()
    all_preds = []
    stems = val_dataset.samples

    # Batched inference
    batch_size = 32
    for start in range(0, len(stems), batch_size):
        batch_stems = stems[start:start + batch_size]
        imgs = []
        for stem in batch_stems:
            img = val_dataset._load_image(stem)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            imgs.append(img_tensor)
        batch = torch.stack(imgs).to(device)

        preds = model(batch)
        detections = decode_predictions(preds, conf_thresh=CONF_THRESH)

        for i, dets_raw in enumerate(detections):
            img_id = start + i
            dets = nms(dets_raw, iou_thresh=NMS_THRESH)
            boxes = []
            if dets.shape[0] > 0:
                d = dets.cpu().numpy()
                xyxy = box_cxcywh_to_xyxy(d[:, :4])
                scores = d[:, 4] * d[:, 5]
                boxes = np.column_stack([xyxy, scores, d[:, 6]]).tolist()
            all_preds.append({"img_id": img_id, "boxes": boxes})

    # Build GT from label cache
    all_gts = []
    for img_id, stem in enumerate(stems):
        labels = val_dataset._label_cache[stem]
        if labels.shape[0] > 0:
            xyxy = box_cxcywh_to_xyxy(labels[:, 1:].numpy())
            cls  = labels[:, 0:1].numpy()
            all_gts.append({"img_id": img_id, "boxes": np.concatenate([xyxy, cls], axis=1)})
        else:
            all_gts.append({"img_id": img_id, "boxes": []})

    map_val, _, _, _ = compute_map(all_preds, all_gts, NUM_CLASSES)
    return map_val


def train():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    tag = "bdd100k_custom_cnn"
    print(f"Run: {tag} (ResNet-50 + FPN, C={NUM_CLASSES})", flush=True)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{tag}.csv"

    train_loader = get_dataloader(
        "clear_day/train", img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    val_loader = get_dataloader(
        "clear_day/val", img_size=IMG_SIZE, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, shuffle=False
    )
    val_dataset: BDD100KDataset = val_loader.dataset

    raw_model = CustomCNN(C=NUM_CLASSES).to(device)
    criterion = CustomCNNLoss(C=NUM_CLASSES)
    
    # Separate param groups: backbone vs head for different learning rates
    backbone_params = list(raw_model.stem.parameters()) + \
                      list(raw_model.layer1.parameters()) + \
                      list(raw_model.layer2.parameters()) + \
                      list(raw_model.layer3.parameters()) + \
                      list(raw_model.layer4.parameters())
    head_params = list(raw_model.fpn.parameters()) + \
                  list(raw_model.cls_head.parameters()) + \
                  list(raw_model.reg_head.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LR * 0.1},  # Backbone: 10x lower LR when unfrozen
        {'params': head_params, 'lr': LR},             # Head: normal LR
    ], weight_decay=1e-4)
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
    )
    scaler = GradScaler("cuda")

    start_epoch = 1
    best_val    = float("inf")
    best_map    = -1.0
    resume_path = CHECKPOINTS / f"{tag}_latest.pt"

    # Freeze backbone during warmup (only train FPN + head)
    raw_model.freeze_backbone()
    print("Backbone frozen for warmup.")

    try:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["val_loss"]
        best_map    = ckpt.get("best_map", -1.0)
        if start_epoch > WARMUP_EPOCHS:
            raw_model.unfreeze_layer4_only()
            print(f"Resumed from epoch {ckpt['epoch']} (layer4 unfrozen), val_loss {ckpt['val_loss']:.4f}")
        else:
            print(f"Resumed from epoch {ckpt['epoch']} (backbone frozen), val_loss {ckpt['val_loss']:.4f}")
    except FileNotFoundError:
        pass
    except (RuntimeError, KeyError) as e:
        print(f"Cannot resume (architecture changed): {e}")
        print("Starting fresh.")

    if start_epoch == 1 or not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "map50", "lr", "epoch_time_s"])

    # Use default compile mode (more stable with grad accumulation than reduce-overhead)
    model = torch.compile(raw_model)

    for epoch in range(start_epoch, EPOCHS + 1):
        # Unfreeze backbone progressively after warmup
        if epoch == WARMUP_EPOCHS + 1:
            raw_model.unfreeze_layer4_only()
            print("Layer4 unfrozen.") 
        elif epoch == WARMUP_EPOCHS + 5:
            raw_model.unfreeze_progressive()
            print("Layer3 unfrozen.")
        elif epoch == WARMUP_EPOCHS + 10:
            raw_model.unfreeze_progressive()
            print("Layer2 unfrozen.")
        elif epoch == WARMUP_EPOCHS + 15:
            raw_model.unfreeze_progressive()
            print("Layer1 unfrozen.")
        elif epoch == WARMUP_EPOCHS + 20:
            raw_model.unfreeze_progressive()
            print("Stem unfrozen (full backbone active).")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, scaler, device, accum_steps=ACCUM_STEPS)
        val_loss   = run_val(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        # mAP evaluation every MAP_EVAL_EVERY epochs
        map50 = ""
        if epoch % MAP_EVAL_EVERY == 0 or epoch == EPOCHS:
            map50_val = compute_val_map(raw_model, val_dataset, device)
            map50 = f"{map50_val * 100:.2f}"

        current_lr = optimizer.param_groups[0]["lr"]
        gpu_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        map_str = f" | mAP50={map50}%" if map50 else ""
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.4f} | val {val_loss:.4f}{map_str} | "
            f"lr {current_lr:.2e} | {elapsed:.0f}s | GPU {gpu_mb:.0f}MB",
            flush=True,
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                map50, f"{current_lr:.2e}", f"{elapsed:.1f}",
            ])

        ckpt = {
            "epoch":           epoch,
            "model_state":     raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state":    scaler.state_dict(),
            "val_loss":        val_loss,
            "best_map":        best_map,
        }
        torch.save(ckpt, CHECKPOINTS / f"{tag}_latest.pt")

        if val_loss < best_val:
            best_val = val_loss

        if map50 and float(map50) > best_map:
            best_map = float(map50)
            torch.save(ckpt, CHECKPOINTS / f"{tag}_best.pt")
            print(f"  -> best mAP50: {best_map:.2f}%")

    print(f"Training complete. Best mAP50: {best_map:.2f}%")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    train()
