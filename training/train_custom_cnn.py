import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

from datasets.bdd100k_dataset import get_dataloader
from models.custom_cnn import CustomCNN, CustomCNNLoss

EPOCHS         = 200
WARMUP_EPOCHS  = 5
BATCH_SIZE     = 128
NUM_WORKERS    = 8
IMG_SIZE       = 448
LR             = 1e-4
NUM_CLASSES    = 10

CHECKPOINTS    = Path(__file__).parent.parent / "checkpoints" / "custom_cnn"
LOGS           = Path(__file__).parent.parent / "results" / "logs"


def run_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            preds = model(images)
            loss  = criterion(preds.float(), labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def run_val(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast("cuda"):
            preds = model(images)
            total_loss += criterion(preds.float(), labels).item()
    return total_loss / len(loader)


def train():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tag = "bdd100k_custom_cnn"
    print(f"Run: {tag} (full variant, C={NUM_CLASSES})")

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{tag}.csv"

    train_loader = get_dataloader("clear_day/train", img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader   = get_dataloader("clear_day/val",   img_size=IMG_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    # Full variant: BN + residual, 10 classes
    raw_model = CustomCNN(C=NUM_CLASSES, use_batchnorm=True, use_residual=True).to(device)
    criterion = CustomCNNLoss(C=NUM_CLASSES)
    optimizer = optim.AdamW(raw_model.parameters(), lr=LR, weight_decay=1e-4)
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])
    scaler    = GradScaler("cuda")

    start_epoch = 1
    best_val    = float("inf")
    resume_path = CHECKPOINTS / f"{tag}_latest.pt"

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
        print(f"Resumed from epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")
    except FileNotFoundError:
        pass
    except (RuntimeError, KeyError) as e:
        print(f"Cannot resume (architecture changed): {e}")
        print("Starting fresh.")

    if not log_path.exists() or log_path.stat().st_size == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_time_s", "gpu_peak_mem_mb"])

    model = torch.compile(raw_model)

    for epoch in range(start_epoch, EPOCHS + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss   = run_val(model, val_loader, criterion, device)
        scheduler.step()
        elapsed    = time.time() - t0
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"lr {current_lr:.2e} | {elapsed:.0f}s | GPU {gpu_mem_mb:.0f}MB"
        )

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.2e}", f"{elapsed:.1f}", f"{gpu_mem_mb:.0f}"])

        ckpt = {
            "epoch": epoch,
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(ckpt, CHECKPOINTS / f"{tag}_latest.pt")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, CHECKPOINTS / f"{tag}_best.pt")
            print(f"  -> best val loss: {best_val:.4f}")

    print(f"Training complete. Best val loss: {best_val:.4f}")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    train()