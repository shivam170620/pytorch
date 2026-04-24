#!/usr/bin/env python3
"""Train YOLOv8n on the Roboflow license-plate dataset (same flow as number_plate_detection.ipynb).

Requires: pip install ultralytics roboflow
Set ROBOFLOW_API_KEY in the environment (or a .env file next to this script via python-dotenv).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _require_api_key() -> str:
    if load_dotenv:
        load_dotenv(Path(__file__).resolve().parent / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        print(
            "Missing ROBOFLOW_API_KEY. Export it or add it to object_detection/.env",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)

    import torch
    from roboflow import Roboflow
    from ultralytics import YOLO

    if torch.cuda.is_available():
        print("Running on GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    api_key = _require_api_key()
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project(
        "license-plate-recognition-rxg4e"
    )
    version = project.version(13)
    dataset = version.download("yolov8")

    model = YOLO("yolov8n.pt")
    data_yaml = Path(dataset.location) / "data.yaml"
    model.train(
        data=str(data_yaml),
        epochs=5,
        imgsz=640,
        batch=16,
        workers=2,
        device=0,
        name="yolov8n-plate-detection",
    )

    out_dir = root / "saved_models"
    out_dir.mkdir(exist_ok=True)
    run_weights = root / "runs" / "detect" / "yolov8n-plate-detection" / "weights"
    shutil.copy(run_weights / "best.pt", out_dir / "best.pt")
    shutil.copy(run_weights / "last.pt", out_dir / "last.pt")
    print("Best and Last model weights have been saved to the 'saved_models' directory.")


if __name__ == "__main__":
    main()
