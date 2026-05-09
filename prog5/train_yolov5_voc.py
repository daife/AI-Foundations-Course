from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "ultralytics" / "ultralytics" / "cfg" / "datasets" / "yolo.yaml"
DEFAULT_MODEL = ROOT.parent / "ultralytics" / "ultralytics" / "cfg" / "models" / "v5" / "yolov5.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv5 on Pascal VOC 2007.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-period", type=int, default=1)
    return parser.parse_args()


def select_device(device: str) -> str | int | list[int]:
    if device != "auto":
        if "," in device:
            return [int(item) for item in device.split(",") if item.strip()]
        return int(device) if device.isdigit() else device

    if not torch.cuda.is_available():
        return "cpu"

    gpu_count = torch.cuda.device_count()
    return list(range(gpu_count)) if gpu_count > 1 else 0


def main() -> None:
    args = parse_args()
    train_device = select_device(args.device)

    print(f"Using data config: {args.data}")
    print(f"Using model config: {args.model}")
    print(f"Using device: {train_device}")

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model config not found: {args.model}. "
            "Clone https://github.com/ultralytics/ultralytics.git in the project root first."
        )
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {args.data}. "
            "Create ultralytics/ultralytics/cfg/datasets/yolo.yaml first."
        )

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        degrees=1,
        shear=0.5,
        perspective=0,
        flipud=0.5,
        mixup=0.5,
        pretrained=False,
        scale=0.5,
        multi_scale=True,
        save_period=args.save_period,
        device=train_device,
    )


if __name__ == "__main__":
    main()
