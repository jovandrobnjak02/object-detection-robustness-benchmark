import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_JSON = Path("results/metrics/bdd100k_results.json")
LOGS_DIR     = Path("results/logs")
PLOTS_DIR    = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "car", "person", "traffic sign", "traffic light",
    "truck", "bus", "bike", "rider", "motor", "train",
]

SPLIT_LABELS = {
    "clear_day/val": "Clear Day",
    "rainy":         "Rainy",
    "snowy":         "Snowy",
    "night":         "Night",
    "overcast":      "Overcast",
}

CNN_COLOR  = "#4C72B0"
YOLO_COLOR = "#DD8452"

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def load_results() -> list[dict]:
    if not RESULTS_JSON.exists():
        sys.exit(f"Results not found at {RESULTS_JSON}. Run evaluate.py first.")
    with open(RESULTS_JSON) as f:
        return json.load(f)


def has_yolo(results: list[dict]) -> bool:
    return any("yolo26_mAP50" in r for r in results)


def plot_map_comparison(results: list[dict]):
    splits      = [SPLIT_LABELS.get(r["split"], r["split"]) for r in results]
    cnn_maps    = [r.get("custom_cnn_mAP50", 0) for r in results]
    yolo_maps   = [r.get("yolo26_mAP50",    None) for r in results]
    has_y       = any(v is not None for v in yolo_maps)

    x = np.arange(len(splits))
    width = 0.35 if has_y else 0.5

    fig, ax = plt.subplots(figsize=(9, 5))

    if has_y:
        ax.bar(x - width/2, cnn_maps,  width, label="Custom CNN", color=CNN_COLOR,  zorder=3)
        ax.bar(x + width/2, [v or 0 for v in yolo_maps], width, label="YOLO26n", color=YOLO_COLOR, zorder=3)
    else:
        ax.bar(x, cnn_maps, width, label="Custom CNN", color=CNN_COLOR, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mAP@50 (%)")
    ax.set_title("Object Detection Performance by Condition")
    ax.set_ylim(0, max(max(cnn_maps), max(v or 0 for v in yolo_maps) if has_y else 0) * 1.2 + 5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend()

    for rect in ax.patches:
        h = rect.get_height()
        if h > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = PLOTS_DIR / "map_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_degradation(results: list[dict]):
    baseline = next((r for r in results if r["split"] == "clear_day/val"), None)
    if baseline is None:
        print("  [skip] no clear_day/val baseline for degradation plot")
        return

    adverse = [r for r in results if r["split"] != "clear_day/val"]
    if not adverse:
        print("  [skip] no adverse splits for degradation plot")
        return

    base_cnn  = baseline.get("custom_cnn_mAP50", 0)
    base_yolo = baseline.get("yolo26_mAP50", None)

    labels     = [SPLIT_LABELS.get(r["split"], r["split"]) for r in adverse]
    cnn_drop   = [base_cnn  - r.get("custom_cnn_mAP50", 0) for r in adverse]
    yolo_drop  = [base_yolo - r.get("yolo26_mAP50",    0) for r in adverse] if base_yolo else None

    x     = np.arange(len(labels))
    width = 0.35 if yolo_drop else 0.5

    fig, ax = plt.subplots(figsize=(8, 5))

    if yolo_drop:
        ax.bar(x - width/2, cnn_drop,  width, label="Custom CNN", color=CNN_COLOR,  zorder=3)
        ax.bar(x + width/2, yolo_drop, width, label="YOLO26n",    color=YOLO_COLOR, zorder=3)
    else:
        ax.bar(x, cnn_drop, width, label="Custom CNN", color=CNN_COLOR, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mAP@50 drop vs clear day (%)")
    ax.set_title("Robustness: mAP Degradation Under Adverse Conditions")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend()

    for rect in ax.patches:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.2,
                f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = PLOTS_DIR / "robustness_degradation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")



def plot_per_class_ap(results: list[dict]):
    baseline = next((r for r in results if r["split"] == "clear_day/val"), None)
    if baseline is None:
        print("  [skip] no clear_day/val for per-class AP plot")
        return

    cnn_aps  = [baseline["custom_cnn_ap_per_class"].get(c) or 0 for c in CLASS_NAMES]
    yolo_aps = None
    if "yolo26_ap_per_class" in baseline:
        yolo_aps = [baseline["yolo26_ap_per_class"].get(c) or 0 for c in CLASS_NAMES]

    y      = np.arange(len(CLASS_NAMES))
    height = 0.35 if yolo_aps else 0.5

    fig, ax = plt.subplots(figsize=(8, 7))

    if yolo_aps:
        ax.barh(y + height/2, cnn_aps,  height, label="Custom CNN", color=CNN_COLOR,  zorder=3)
        ax.barh(y - height/2, yolo_aps, height, label="YOLO26n",    color=YOLO_COLOR, zorder=3)
    else:
        ax.barh(y, cnn_aps, height, label="Custom CNN", color=CNN_COLOR, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("AP@50 (%)")
    ax.set_title("Per-Class AP — Clear Day Validation")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend()

    fig.tight_layout()
    path = PLOTS_DIR / "per_class_ap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_class_precision_recall(results: list[dict]):
    baseline = next((r for r in results if r["split"] == "clear_day/val"), None)
    if baseline is None:
        print("  [skip] no clear_day/val for precision/recall plot")
        return
    if "custom_cnn_precision_per_class" not in baseline:
        print("  [skip] precision/recall not in results (re-run evaluate.py)")
        return

    cnn_p  = [baseline["custom_cnn_precision_per_class"].get(c) or 0 for c in CLASS_NAMES]
    cnn_r  = [baseline["custom_cnn_recall_per_class"].get(c) or 0 for c in CLASS_NAMES]
    yolo_p = yolo_r = None
    if "yolo26_precision_per_class" in baseline:
        yolo_p = [baseline["yolo26_precision_per_class"].get(c) or 0 for c in CLASS_NAMES]
        yolo_r = [baseline["yolo26_recall_per_class"].get(c) or 0 for c in CLASS_NAMES]

    y = np.arange(len(CLASS_NAMES))

    models = [("Custom CNN", cnn_p, cnn_r, CNN_COLOR)]
    if yolo_p:
        models.append(("YOLO26n", yolo_p, yolo_r, YOLO_COLOR))

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 7), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, (name, prec, rec, color) in zip(axes, models):
        height = 0.35
        ax.barh(y + height/2, prec, height, label="Precision", color=color, zorder=3)
        ax.barh(y - height/2, rec,  height, label="Recall",    color=color, alpha=0.5, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("%")
        ax.set_title(f"{name} — Precision & Recall")
        ax.set_xlim(0, 105)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.legend()

    fig.suptitle("Per-Class Precision & Recall (best F1) — Clear Day Val", y=1.01)
    fig.tight_layout()
    path = PLOTS_DIR / "per_class_precision_recall.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_fps(results: list[dict]):
    baseline = next((r for r in results if r["split"] == "clear_day/val"), None)
    if baseline is None:
        return

    models = ["Custom CNN"]
    fps    = [baseline.get("custom_cnn_fps", 0)]
    colors = [CNN_COLOR]

    if "yolo26_fps" in baseline:
        models.append("YOLO26n")
        fps.append(baseline["yolo26_fps"])
        colors.append(YOLO_COLOR)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(models, fps, color=colors, width=0.4, zorder=3)
    ax.set_ylabel("Frames per second")
    ax.set_title("Inference Speed (GPU)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)

    for bar, val in zip(bars, fps):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    path = PLOTS_DIR / "fps_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_map_trend(results: list[dict]):
    splits    = [SPLIT_LABELS.get(r["split"], r["split"]) for r in results]
    cnn_maps  = [r.get("custom_cnn_mAP50", 0) for r in results]
    yolo_maps = [r.get("yolo26_mAP50", None) for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(splits))

    ax.plot(x, cnn_maps, "o-", color=CNN_COLOR,  label="Custom CNN", linewidth=2, markersize=7)
    if any(v is not None for v in yolo_maps):
        ax.plot(x, [v or 0 for v in yolo_maps], "s-", color=YOLO_COLOR,
                label="YOLO26n", linewidth=2, markersize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("mAP@50 (%)")
    ax.set_title("Detection Performance Across Conditions")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    for i, v in enumerate(cnn_maps):
        ax.annotate(f"{v:.1f}", (x[i], v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=CNN_COLOR)
    if any(v is not None for v in yolo_maps):
        for i, v in enumerate(yolo_maps):
            if v is not None:
                ax.annotate(f"{v:.1f}", (x[i], v), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=9, color=YOLO_COLOR)

    fig.tight_layout()
    path = PLOTS_DIR / "map_trend.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_loss_curves():
    log_path = LOGS_DIR / "bdd100k_custom_cnn.csv"
    if not log_path.exists():
        print(f"  [skip] training log not found at {log_path}")
        return

    epochs, train_losses, val_losses = [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    if not epochs:
        print("  [skip] training log is empty")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, color=CNN_COLOR,  label="Train loss", linewidth=1.5)
    ax.plot(epochs, val_losses,   color=YOLO_COLOR, label="Val loss",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Custom CNN Training Loss Curves")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    path = PLOTS_DIR / "loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")

def _plot_single_confusion_matrix(ax, matrix: np.ndarray, title: str):
    """Plot a normalized (by GT column) confusion matrix on the given axes."""
    n = len(CLASS_NAMES)
    cm = np.array(matrix, dtype=float)[:n, :n]  # exclude background row/col

    col_sums = cm.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    cm_norm = cm / col_sums  # column-normalize: shows recall per GT class

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Predicted")
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.2f}" if val > 0.01 else "",
                    ha="center", va="center", fontsize=6, color=color)

    return im


def plot_confusion_matrices(results: list[dict]):
    baseline = next((r for r in results if r["split"] == "clear_day/val"), None)
    if baseline is None:
        print("  [skip] no clear_day/val for confusion matrix plot")
        return
    if "custom_cnn_confusion_matrix" not in baseline:
        print("  [skip] confusion matrix not in results (re-run evaluate.py)")
        return

    has_y = "yolo26_confusion_matrix" in baseline
    ncols = 2 if has_y else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    im = _plot_single_confusion_matrix(axes[0], baseline["custom_cnn_confusion_matrix"], "Custom CNN")
    if has_y:
        _plot_single_confusion_matrix(axes[1], baseline["yolo26_confusion_matrix"], "YOLO26n")

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="Recall (col-normalized)")
    fig.suptitle("Confusion Matrix — Clear Day Validation (col-normalized)", y=1.01)
    fig.tight_layout()
    path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

def main():
    results = load_results()
    print(f"Loaded results for {len(results)} splits")
    print("Generating plots...")

    plot_map_comparison(results)
    plot_degradation(results)
    plot_per_class_ap(results)
    plot_per_class_precision_recall(results)
    plot_fps(results)
    plot_map_trend(results)
    plot_loss_curves()
    plot_confusion_matrices(results)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
