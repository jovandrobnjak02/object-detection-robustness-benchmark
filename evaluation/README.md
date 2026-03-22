# Evaluation

Run in this order:

## 1. Train models

Custom CNN:
```bash
python scripts/cache_to_shm.py
python training/train_custom_cnn.py
```

YOLO26:
```bash
python training/train_yolo26.py
```

## 2. Run evaluation

Evaluates both models on all splits (clear day, rainy, snowy, night, overcast).
YOLO26 is skipped automatically if its checkpoint is missing.

```bash
python evaluation/evaluate.py
```

Outputs:
- `results/metrics/bdd100k_results.json` — full results per split including per-class AP, precision, recall, confusion matrices, FPS, and GPU memory
- `results/metrics/bdd100k_summary.csv` — condensed table

## 3. Generate plots

Reads from the JSON output and the training log CSV. Saves all figures to `results/plots/`.

```bash
python evaluation/plot_results.py
```

Plots generated:
| File | Description |
|---|---|
| `map_comparison.png` | mAP@50 bar chart across all conditions |
| `map_trend.png` | mAP@50 line chart across conditions |
| `robustness_degradation.png` | mAP drop vs clear-day baseline |
| `per_class_ap.png` | AP@50 per class (clear day val) |
| `per_class_precision_recall.png` | Precision & recall per class at best F1 |
| `fps_comparison.png` | Inference speed (FPS) |
| `loss_curves.png` | Training & validation loss over epochs |
| `confusion_matrix.png` | Column-normalised confusion matrix (clear day val) |
