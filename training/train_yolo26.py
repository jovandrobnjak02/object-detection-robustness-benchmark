import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
DATA_YAML    = PROJECT_ROOT / "data" / "kitti" / "kitti.yaml"
RESULTS_DIR  = PROJECT_ROOT / "results"


def train(model_name: str, epochs: int, batch: int, imgsz: int):
    model = YOLO(f"{model_name}.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(PROJECT_ROOT / "checkpoints" / "yolo26"),
        name=model_name,
        exist_ok=True,
        device=0,
        workers=4,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26 on KITTI")
    parser.add_argument("--model", type=str, default="yolo26s", help="Model variant (yolo26n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch, args.imgsz)
