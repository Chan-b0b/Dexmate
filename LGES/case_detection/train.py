"""Fine-tune YOLO-OBB on the case dataset (thin wrapper over ultralytics).

    pip install ultralytics
    python train.py                      # uses dataset/data.yaml
    python train.py --epochs 200 --model yolov8s-obb.pt

Trains from a small pretrained OBB checkpoint. Result: runs/obb/case/weights/best.pt
-> set that as cfg.OBB_MODEL_PATH, then run_test.py uses the learned detector.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["case", "bin"], default="case",
                    help="which detector to train (box type from cfg.TRAIN_OBB)")
    ap.add_argument("--data", default=None, help="data.yaml (default dataset_<target>/data.yaml)")
    ap.add_argument("--model", default=None, help="pretrained checkpoint (default per target)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    # Box type comes from cfg.TRAIN_OBB so labels (SAM2) and task always match.
    if cfg.TRAIN_OBB[args.target]:
        model, project = args.model or "yolov8n-obb.pt", "obb"
    else:
        model, project = args.model or "yolov8n.pt", "detect"
    cfg_path = "OBB_MODEL_PATH" if args.target == "case" else "BIN_MODEL_PATH"
    data = args.data or f"dataset_{args.target}/data.yaml"

    from ultralytics import YOLO

    YOLO(model).train(
        data=str(HERE / data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(HERE / "runs" / project),
        name=args.target,
        exist_ok=True,
        # one orientation-bearing object on a fairly fixed backdrop; heavy
        # rotation/flip aug teaches angle (case) / position robustness (bin).
        degrees=180.0, fliplr=0.5, flipud=0.5,
    )
    print(f"\nDone. Set cfg.{cfg_path} = 'runs/{project}/{args.target}/weights/best.pt'")


if __name__ == "__main__":
    main()
