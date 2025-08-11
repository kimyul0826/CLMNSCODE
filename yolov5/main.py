"""
Unified entrypoint for YOLOv5 tasks.

Usage examples:
  - Train:      python main.py --mode train      --data data/coco128.yaml --weights yolov5s.pt --imgsz 640
  - Val:        python main.py --mode val        --data data/coco128.yaml --weights yolov5s.pt --imgsz 640
  - Detect:     python main.py --mode detect     --source data/images --weights yolov5s.pt
  - Train+Test: python main.py --mode train_val  --data data/coco128.yaml --weights yolov5s.pt --imgsz 640
  - Tune HPO:   python main.py --mode tune       --data data/coco128.yaml --weights yolov5s.pt --imgsz 640

This script forwards all arguments (other than --mode) to the respective
YOLOv5 module (`train.py`, `val.py`, `detect.py`) by reusing their own
`parse_opt()` and `main()` implementations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH for local imports
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path (for consistency with YOLOv5 scripts)


def parse_main_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse only the top-level --mode argument and return remaining args.

    Returns:
        (args, rest): Parsed args with `mode`, and the remaining CLI args to forward.
    """
    parser = argparse.ArgumentParser(description="YOLOv5 unified entrypoint")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "val", "detect", "train_val", "tune"],
        help="Select subcommand to run: train | val | detect | train_val | tune",
    )
    args, rest = parser.parse_known_args()
    return args, rest


def main() -> None:
    args, rest = parse_main_args()

    # Set sys.argv so submodules' parse_opt() read the forwarded args correctly
    sys.argv = [f"{args.mode}.py"] + rest

    if args.mode == "train":
        from train import main as train_main, parse_opt as parse_train_opt

        opt = parse_train_opt()
        train_main(opt)

    elif args.mode == "val":
        from val import main as val_main, parse_opt as parse_val_opt

        opt = parse_val_opt()
        val_main(opt)

    elif args.mode == "detect":
        from detect import main as detect_main, parse_opt as parse_detect_opt

        opt = parse_detect_opt()
        detect_main(opt)

    elif args.mode == "train_val":
        # 1) Run training with forwarded args
        from train import main as train_main, parse_opt as parse_train_opt
        from val import main as val_main, parse_opt as parse_val_opt

        # Ensure submodule parsers see the correct argv
        sys.argv = ["train.py"] + rest
        train_opt = parse_train_opt()
        train_main(train_opt)

        # 2) Locate best.pt produced by training
        #    YOLOv5 saves to {save_dir}/weights/best.pt
        save_dir = Path(str(getattr(train_opt, "save_dir", "")))
        best_weights = save_dir / "weights" / "best.pt"
        if not best_weights.is_file():
            # Fallback: try to infer from default pattern runs/train/{name}/{name_YYMMDD_HHMMSS}/weights/best.pt
            # If not found, raise a helpful error.
            raise FileNotFoundError(f"best.pt not found at expected path: {best_weights}")

        # 3) Run validation in test mode using the same dataset and image size
        sys.argv = ["val.py"]  # start from defaults, then override programmatically
        val_opt = parse_val_opt()
        val_opt.task = "test"
        val_opt.weights = [str(best_weights)]
        # Reuse key options from training
        if hasattr(train_opt, "data"):
            val_opt.data = train_opt.data
        if hasattr(train_opt, "imgsz"):
            val_opt.imgsz = train_opt.imgsz
        if getattr(train_opt, "device", ""):
            val_opt.device = train_opt.device
        # Enable common save options by default
        val_opt.save_txt = True
        val_opt.save_json = True
        val_opt.save_conf = True
        val_opt.save_hybrid = True
        val_opt.verbose = True
        # Run validation
        val_main(val_opt)

    elif args.mode == "tune":
        # Forward to train with evolution enabled and sensible defaults
        from train import main as train_main, parse_opt as parse_train_opt
        
        # Start from forwarded args and inject defaults for tuning
        mod_rest = list(rest)
        # Ensure --evolve present (no value -> defaults to 300 generations per train.py)
        if not any(a.startswith("--evolve") for a in mod_rest):
            mod_rest += ["--evolve"]
        # Default epochs to 300 if not explicitly provided
        if "--epochs" not in mod_rest:
            mod_rest += ["--epochs", "300"]
        # Prefer automatic batch size search unless user provided one
        if "--batch-size" not in mod_rest:
            mod_rest += ["--batch-size", "-1"]

        # Parse and run
        sys.argv = ["train.py"] + mod_rest
        train_opt = parse_train_opt()
        train_main(train_opt)

    else:  # pragma: no cover - safeguarded by argparse choices
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()

