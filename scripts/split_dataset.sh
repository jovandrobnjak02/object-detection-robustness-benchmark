#!/bin/bash
#
# Split: 5985 train / 1496 val (7481 total)
#
# Original images downloaded from:
# https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
#
# The unsplit images were deleted to save space
# Usage: bash scripts/split_dataset.sh

set -e

DATA_DIR="$(dirname "$0")/../data/kitti"
IMAGES_DIR="$DATA_DIR/images"
LABELS_DIR="$DATA_DIR/labels"

TRAIN_IMAGES="$DATA_DIR/train/images"
TRAIN_LABELS="$DATA_DIR/train/labels"
VAL_IMAGES="$DATA_DIR/val/images"
VAL_LABELS="$DATA_DIR/val/labels"

TRAIN_COUNT=5985
VAL_COUNT=1496

mkdir -p "$TRAIN_IMAGES" "$TRAIN_LABELS" "$VAL_IMAGES" "$VAL_LABELS"

mapfile -t FILES < <(
    for img in "$IMAGES_DIR"/*.png; do
        base=$(basename "$img" .png)
        if [[ -f "$LABELS_DIR/$base.txt" ]]; then
            echo "$base"
        fi
    done | shuf
)

TOTAL=${#FILES[@]}
echo "Found $TOTAL matched image/label pairs"

if (( TOTAL != TRAIN_COUNT + VAL_COUNT )); then
    echo "Warning: expected $((TRAIN_COUNT + VAL_COUNT)) pairs, got $TOTAL"
fi


echo "Copying $TRAIN_COUNT train samples..."
for i in "${!FILES[@]}"; do
    if (( i < TRAIN_COUNT )); then
        base="${FILES[$i]}"
        cp "$IMAGES_DIR/$base.png" "$TRAIN_IMAGES/$base.png"
        cp "$LABELS_DIR/$base.txt" "$TRAIN_LABELS/$base.txt"
    fi
done

echo "Copying $VAL_COUNT val samples..."
for i in "${!FILES[@]}"; do
    if (( i >= TRAIN_COUNT )); then
        base="${FILES[$i]}"
        cp "$IMAGES_DIR/$base.png" "$VAL_IMAGES/$base.png"
        cp "$LABELS_DIR/$base.txt" "$VAL_LABELS/$base.txt"
    fi
done

echo "Done."
echo "  train/images: $(ls "$TRAIN_IMAGES" | wc -l)"
echo "  train/labels: $(ls "$TRAIN_LABELS" | wc -l)"
echo "  val/images:   $(ls "$VAL_IMAGES" | wc -l)"
echo "  val/labels:   $(ls "$VAL_LABELS" | wc -l)"