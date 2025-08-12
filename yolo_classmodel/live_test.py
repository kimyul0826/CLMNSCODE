import os
import sys
import time
import math
import json
import yaml
import shutil
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import torch
import numpy as np

# Add project submodules to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # .../CLMNSCODE
YOLOV5_DIR = PROJECT_ROOT / "yolov5"
CLASSMODEL_DIR = PROJECT_ROOT / "classmodel"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))
if str(CLASSMODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSMODEL_DIR))

# YOLOv5 imports
from models.common import DetectMultiBackend  # type: ignore
from utils.dataloaders import LoadStreams  # type: ignore
from utils.general import (  # type: ignore
    check_img_size,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
    cv2,
)
from utils.torch_utils import select_device, smart_inference_mode  # type: ignore
from ultralytics.utils.plotting import Annotator, colors, save_one_box  # type: ignore

# ClassModel imports
from utils.transforms import build_transform_for_split  # type: ignore


KST = time.tzname[0]  # Not used for formatting, kept for completeness


def get_kst_timestamp() -> str:
    """Return KST timestamp as YYYYMMDD_HHMMSS without external deps."""
    # Convert current UTC time to KST (+9h) for formatting
    kst_epoch = time.time() + 9 * 3600
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime(kst_epoch))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class StabilityConfig:
    required_seconds: float
    tolerance_seconds: float


class StabilityTracker:
    """Tracks whether a boolean condition stays True for a required duration allowing brief gaps."""

    def __init__(self, required_seconds: float, tolerance_seconds: float) -> None:
        self.required_seconds = required_seconds
        self.tolerance_seconds = tolerance_seconds
        self.window_start_time: Optional[float] = None
        self.false_gap_accum: float = 0.0
        self.last_time: Optional[float] = None

    def update(self, condition_true: bool, now_time: float) -> bool:
        if self.last_time is None:
            self.last_time = now_time
            self.window_start_time = now_time if condition_true else None
            self.false_gap_accum = 0.0
            return False

        dt = max(0.0, now_time - self.last_time)
        self.last_time = now_time

        if condition_true:
            if self.window_start_time is None:
                # Start a new window at the first true observation
                self.window_start_time = now_time
            # When condition is true, reset the false gap accumulator
            self.false_gap_accum = 0.0
        else:
            # Accumulate the gap when condition is false
            self.false_gap_accum += dt
            if self.false_gap_accum > self.tolerance_seconds:
                # Reset the window after exceeding tolerance
                self.window_start_time = None
                self.false_gap_accum = 0.0

        if self.window_start_time is None:
            return False

        elapsed = now_time - self.window_start_time
        return elapsed >= self.required_seconds


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_output_dir(base_runs_dir: Path, parts: str, name: Optional[str]) -> Path:
    if name and str(name).strip():
        folder_name = name.strip()
    else:
        folder_name = f"live_{parts}_{get_kst_timestamp()}"
    return base_runs_dir / folder_name


def yolo_label_line(cls_id: int, xyxy: np.ndarray, img_shape: Tuple[int, int], save_conf: bool, conf: Optional[float]) -> str:
    h, w = img_shape[:2]
    gn = torch.tensor([w, h, w, h], dtype=torch.float32)
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
    if save_conf and conf is not None:
        vals = (cls_id, *xywh, conf)
    else:
        vals = (cls_id, *xywh)
    return ("%g " * len(vals)).rstrip() % vals


def load_classification_model(model_path: Path, base_config_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a classification model checkpoint and return model + config dict."""
    from models.resnet import ResNet18, ResNet50  # type: ignore
    from models.efficientnet import EfficientNet  # type: ignore
    from models.mobilenet import MobileNet  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(model_path), map_location=device)

    # Load base config for model architecture defaults
    cfg = load_yaml(base_config_path)
    model_name = cfg.get("model", {}).get("name", "resnet18")
    pretrained = cfg.get("model", {}).get("pretrained", True)
    classes = checkpoint.get("classes", ["good", "bad"])  # default order
    num_classes = len(classes)

    if model_name == "resnet18":
        model = ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet50":
        model = ResNet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "efficientnet":
        model = EfficientNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "mobilenet":
        model = MobileNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model in config: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore
    model.to(device)
    model.eval()
    # Attach classes to cfg for reference
    cfg.setdefault("dataset", {}).setdefault("classes", {0: "good", 1: "bad"})
    return model, cfg


def build_transform_from_config(base_cfg: Dict[str, Any], transform_type: str) -> Any:
    """Create a torchvision transform for inference split based on provided transform type."""
    # Clone and override for test split
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via json
    ds = cfg.setdefault("dataset", {})
    # Prefer new location dataset.transforms.test
    transforms_cfg = ds.setdefault("transforms", {})
    transforms_cfg["test"] = transform_type
    # Ensure resize/normalize exist
    ds.setdefault("resize", ds.get("resize", [224, 224]))
    aug = ds.setdefault("augmentation", {})
    aug.setdefault("normalize", {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    })
    return build_transform_for_split(cfg, "test")


def classify_images(model: torch.nn.Module, transform, image_paths: List[Path]) -> Tuple[List[int], List[List[float]]]:
    """Run model on a list of image paths. Return predicted classes and probability vectors."""
    device = next(model.parameters()).device
    preds: List[int] = []
    probs: List[List[float]] = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            # Create a gray placeholder if read fails
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image  # local import
        pil = Image.fromarray(img_rgb)
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
            pred = int(np.argmax(prob))
        preds.append(pred)
        probs.append(prob)
    return preds, probs


def draw_text(img, text: str, org: Tuple[int, int], color=(0, 255, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def inside(xyxy_inner: np.ndarray, xyxy_outer: np.ndarray) -> bool:
    x1, y1, x2, y2 = xyxy_inner
    ox1, oy1, ox2, oy2 = xyxy_outer
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return (ox1 <= cx <= ox2) and (oy1 <= cy <= oy2)


@smart_inference_mode()
def run_live_pipeline(config_path: Optional[str] = None):
    # 1) Load config
    config_file = Path(config_path) if config_path else CURRENT_DIR / "live_config.yaml"
    cfg = load_yaml(config_file)

    source = str(cfg.get("source", "0"))  # webcam index as string for LoadStreams
    parts = str(cfg.get("parts", "bolt")).strip()
    name_opt = cfg.get("name", None)

    yolo_cfg: Dict[str, Any] = cfg.get("yolo", {})
    yolo_weights = str(yolo_cfg.get("model", str(YOLOV5_DIR / "yolov5s.pt")))
    imgsz = int(yolo_cfg.get("imgsz", 640))
    conf_thres = float(yolo_cfg.get("conf_thres", 0.25))
    device_str = str(yolo_cfg.get("device", ""))
    save_txt = bool(yolo_cfg.get("save_txt", True))
    save_crop = bool(yolo_cfg.get("save_crop", True))
    save_format = int(yolo_cfg.get("save_format", 0))  # 0: YOLO, 1: PascalVOC
    save_conf = bool(yolo_cfg.get("save_conf", False))

    # 2) Prepare output directories
    base_runs_dir = CURRENT_DIR / "runs" / "live"
    out_dir = build_output_dir(base_runs_dir, parts, name_opt)
    labels_dir = out_dir / "labels"
    crops_dir = out_dir / "crops"
    annotated_dir = out_dir / "annotated"
    ensure_dir(labels_dir)
    ensure_dir(crops_dir)
    ensure_dir(annotated_dir)

    # 3) Initialize YOLO model and dataloader (webcam only as per spec)
    device = select_device(device_str)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((imgsz, imgsz), s=stride)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    bs = len(dataset)
    if bs < 1:
        print("No webcam streams available.")
        return

    # Determine FPS (fallback to 30)
    fps = 30.0
    try:
        if hasattr(dataset, "fps") and dataset.fps:
            fps = float(dataset.fps)
        else:
            # Try to read from internal video capture
            if hasattr(dataset, "caps") and dataset.caps:
                vcap = dataset.caps[0]
                fps_read = vcap.get(cv2.CAP_PROP_FPS)
                if fps_read and fps_read > 0:
                    fps = float(fps_read)
    except Exception:
        fps = 30.0

    # Stability trackers
    if parts == "bolt":
        required_seconds = 5.0
    else:  # door
        required_seconds = 3.0
    tolerance_seconds = 0.5
    stability_tracker = StabilityTracker(required_seconds, tolerance_seconds)

    frame_indices = [0 for _ in range(bs)]

    # Prepare class model configs
    resnet_cfg: Dict[str, Any] = cfg.get("resnet", {})
    base_classmodel_config_path = Path(resnet_cfg.get("config", CLASSMODEL_DIR / "config.yaml"))

    # Pre-load classification models (bolt single, door per-region)
    bolt_model: Optional[torch.nn.Module] = None
    bolt_base_cfg: Optional[Dict[str, Any]] = None
    door_models: Dict[str, torch.nn.Module] = {}
    door_cfgs: Dict[str, Dict[str, Any]] = {}

    if parts == "bolt":
        bolt_model_path = Path(resnet_cfg.get("model", ""))
        if str(bolt_model_path).strip():
            bolt_model, bolt_base_cfg = load_classification_model(bolt_model_path, base_classmodel_config_path)
        else:
            print("[WARN] Bolt classification model path is empty. Will skip classification when needed.")
    else:  # door
        for region_key in ["high", "mid", "low"]:
            mkey = f"{region_key}_model"
            if mkey in resnet_cfg and str(resnet_cfg[mkey]).strip():
                model_path = Path(resnet_cfg[mkey])
                model_obj, base_cfg_obj = load_classification_model(model_path, base_classmodel_config_path)
                door_models[region_key] = model_obj
                door_cfgs[region_key] = base_cfg_obj
            else:
                print(f"[WARN] Door classification model path for '{region_key}' is empty. Will skip classification.")

    # Build transforms per part/region from config
    if parts == "bolt":
        bolt_transform_type = str(resnet_cfg.get("transform", "standard"))
        bolt_transform = None
        if bolt_base_cfg is not None:
            bolt_transform = build_transform_from_config(bolt_base_cfg, bolt_transform_type)
    else:
        transforms_per_region: Dict[str, Any] = {}
        for region_key in ["high", "mid", "low"]:
            tkey = f"{region_key}_transform"
            region_t = str(resnet_cfg.get(tkey, "standard"))
            base_cfg_obj = door_cfgs.get(region_key)
            if base_cfg_obj is not None:
                transforms_per_region[region_key] = build_transform_from_config(base_cfg_obj, region_t)

    print(f"📁 Live output directory: {out_dir}")
    print(f"🎥 Source: {source} | parts={parts} | fps={fps:.1f} | stability={required_seconds}s tol={tolerance_seconds}s")

    # Per-image crop indexing
    crop_index_map: Dict[str, int] = {}

    # For door RF optional
    rf_model = None
    rf_model_path = resnet_cfg.get("random_forest_model")
    if rf_model_path and str(rf_model_path).strip():
        try:
            import joblib  # type: ignore
            rf_model = joblib.load(rf_model_path)
        except Exception:
            print("[WARN] Failed to load RandomForest model. RF results will be skipped.")

    # 4) Inference loop
    for (paths, im, im0s, vid_cap, s) in dataset:
        now = time.time()

        # Preprocess
        im_tensor = torch.from_numpy(im).to(model.device)
        im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()  # uint8 to fp16/32
        im_tensor /= 255.0
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]

        # Inference
        model.warmup(imgsz=(1 if pt or model.triton else len(im_tensor), 3, *imgsz))
        pred = model(im_tensor)
        pred = non_max_suppression(pred, conf_thres, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

        # For each image in batch (streams)
        for i, det in enumerate(pred):
            # Acquire per-stream data
            p = Path(paths[i])
            im0 = im0s[i].copy()
            frame_idx = frame_indices[i]
            frame_indices[i] += 1

            # Prepare annotator
            annotator = Annotator(im0, line_width=3, example=str(names))

            # Rescale to original size
            if len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()

            # Collect detections by class
            class_to_boxes: Dict[int, List[np.ndarray]] = {}
            class_to_confs: Dict[int, List[float]] = {}
            for *xyxy_t, conf_t, cls_t in reversed(det):
                cls_id = int(cls_t)
                xyxy_np = torch.tensor(xyxy_t).view(1, 4).cpu().numpy()[0]
                conf_val = float(conf_t)
                class_to_boxes.setdefault(cls_id, []).append(xyxy_np)
                class_to_confs.setdefault(cls_id, []).append(conf_val)
                # Draw detection boxes
                label = names[cls_id]
                annotator.box_label(xyxy_t, label, color=colors(cls_id, True))

            # Write YOLO labels for this frame if requested
            if save_txt and len(det):
                label_path = labels_dir / f"frame_{frame_idx:06d}.txt"
                with open(label_path, "w") as lf:
                    for *xyxy_t, conf_t, cls_t in reversed(det):
                        cls_id = int(cls_t)
                        xyxy_np = torch.tensor(xyxy_t).view(1, 4).cpu().numpy()[0]
                        line = yolo_label_line(cls_id, xyxy_np, im0.shape, save_conf, float(conf_t) if save_conf else None)
                        lf.write(line + "\n")

            # Stability/presence condition check
            is_stable = False
            if parts == "bolt":
                # Require classes 1..6 all present and maintained for 5s (0.5s tolerance)
                have_all_frames = all((c in class_to_boxes and len(class_to_boxes[c]) > 0) for c in [1, 2, 3, 4, 5, 6])
                is_stable = stability_tracker.update(have_all_frames, now)
            else:
                # Door: exactly one each for classes 0,1,2
                counts = [len(class_to_boxes.get(c, [])) for c in [0, 1, 2]]
                exactly_one_each = (counts == [1, 1, 1])
                is_stable = stability_tracker.update(exactly_one_each, now)

            # Cropping and classification when stable
            crops_saved_paths: List[Path] = []
            crops_saved_regions: List[Tuple[str, np.ndarray]] = []  # (tag, xyxy)
            if is_stable:
                if parts == "bolt":
                    frame_boxes = []
                    for c in [1, 2, 3, 4, 5, 6]:
                        for b in class_to_boxes.get(c, []):
                            frame_boxes.append(b)
                    bolt_boxes = class_to_boxes.get(0, [])
                    # Select bolts whose centers are inside any frame box
                    selected_bolts = []
                    for b in bolt_boxes:
                        if any(inside(b, f) for f in frame_boxes):
                            selected_bolts.append(b)
                    # Save crops
                    crop_count = 0
                    for b in selected_bolts:
                        x1, y1, x2, y2 = map(int, b)
                        crop_img = im0[y1:y2, x1:x2]
                        crop_name = f"frame_{frame_idx:06d}_{crop_count}.jpg"  # index from 0
                        crop_path = crops_dir / crop_name
                        cv2.imwrite(str(crop_path), crop_img)
                        crops_saved_paths.append(crop_path)
                        crops_saved_regions.append((f"bolt_{crop_count}", b))
                        crop_count += 1

                    # Bolt classification and voting
                    hard_vote_result: Optional[int] = None
                    soft_vote_result: Optional[int] = None
                    soft_vote_probs: Optional[List[float]] = None
                    per_crop_preds: List[int] = []
                    per_crop_probs: List[List[float]] = []

                    if len(crops_saved_paths) != 2:
                        hard_vote_result = 1  # bad
                        soft_vote_result = 1
                        soft_vote_probs = [0.0, 1.0]
                    elif bolt_model is not None and 'bolt_transform' in locals() and locals()['bolt_transform'] is not None:
                        preds, probs = classify_images(bolt_model, locals()['bolt_transform'], crops_saved_paths)
                        per_crop_preds = preds
                        per_crop_probs = probs
                        # Hard voting: both must be good (0) to be good
                        hard_vote_result = 0 if all(p == 0 for p in preds) else 1
                        # Soft voting: average probabilities
                        avg_probs = np.mean(np.array(probs), axis=0).tolist()
                        soft_vote_result = int(np.argmax(avg_probs))
                        soft_vote_probs = avg_probs
                    else:
                        print("[WARN] Bolt classification skipped (model or transform missing).")

                    # Annotate with classification
                    for tag, box in crops_saved_regions:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    # If we have per-crop preds, overlay text near each crop
                    for idx, (tag, box) in enumerate(crops_saved_regions):
                        x1, y1, x2, y2 = map(int, box)
                        label_txt = f"{tag}"
                        if idx < len(per_crop_probs):
                            prob = per_crop_probs[idx]
                            label_txt += f" good:{prob[0]:.2f} bad:{prob[1]:.2f} -> {['good','bad'][per_crop_preds[idx]]}"
                        draw_text(im0, label_txt, (x1, max(0, y1 - 5)), (0, 255, 255))

                    # Save results text
                    results_path = out_dir / "results_bolt.txt"
                    with open(results_path, "a") as rf:
                        rf.write(json.dumps({
                            "frame": frame_idx,
                            "time": get_kst_timestamp(),
                            "num_crops": len(crops_saved_paths),
                            "per_crop_probs": per_crop_probs,
                            "hard_vote": int(hard_vote_result) if hard_vote_result is not None else None,
                            "soft_vote": int(soft_vote_result) if soft_vote_result is not None else None,
                            "soft_probs": soft_vote_probs,
                        }) + "\n")

                else:  # door
                    # Expect exactly one per class 0,1,2
                    region_map = {0: "high", 1: "mid", 2: "low"}
                    region_boxes: Dict[str, np.ndarray] = {}
                    ok = True
                    for c in [0, 1, 2]:
                        boxes = class_to_boxes.get(c, [])
                        if len(boxes) != 1:
                            ok = False
                            break
                        region_boxes[region_map[c]] = boxes[0]

                    if ok:
                        # Save crops for each region
                        region_crop_paths: Dict[str, Path] = {}
                        for region, b in region_boxes.items():
                            x1, y1, x2, y2 = map(int, b)
                            crop_img = im0[y1:y2, x1:x2]
                            crop_name = f"frame_{frame_idx:06d}_{region}.jpg"
                            crop_path = crops_dir / crop_name
                            cv2.imwrite(str(crop_path), crop_img)
                            region_crop_paths[region] = crop_path
                            crops_saved_paths.append(crop_path)
                            crops_saved_regions.append((region, b))

                        # Region-wise classification
                        region_preds: Dict[str, int] = {}
                        region_probs: Dict[str, List[float]] = {}
                        for region in ["high", "mid", "low"]:
                            model_obj = door_models.get(region)
                            transform_obj = locals().get("transforms_per_region", {}).get(region)  # type: ignore
                            if model_obj is None or transform_obj is None:
                                continue
                            preds, probs = classify_images(model_obj, transform_obj, [region_crop_paths[region]])
                            region_preds[region] = preds[0]
                            region_probs[region] = probs[0]

                        # Voting
                        hard_vote = None
                        soft_vote = None
                        soft_probs = None
                        if len(region_preds) == 3:
                            hard_vote = 0 if all(region_preds[r] == 0 for r in ["high", "mid", "low"]) else 1
                            # Soft: average per-class probs across regions
                            arr = np.array([region_probs[r] for r in ["high", "mid", "low"]])
                            avg = np.mean(arr, axis=0).tolist()
                            soft_vote = int(np.argmax(avg))
                            soft_probs = avg

                        # RandomForest (optional)
                        rf_vote = None
                        if rf_model is not None and len(region_probs) == 3:
                            feats = [
                                region_probs["high"][0], region_probs["high"][1],
                                region_probs["mid"][0], region_probs["mid"][1],
                                region_probs["low"][0], region_probs["low"][1],
                            ]
                            try:
                                rf_vote = int(rf_model.predict([feats])[0])
                            except Exception:
                                rf_vote = None

                        # Annotate
                        for region, box in crops_saved_regions:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            if region in region_probs:
                                prob = region_probs[region]
                                draw_text(
                                    im0,
                                    f"{region} good:{prob[0]:.2f} bad:{prob[1]:.2f} -> {['good','bad'][region_preds[region]]}",
                                    (x1, max(0, y1 - 5)),
                                    (0, 255, 255),
                                )

                        # Save results
                        results_path = out_dir / "results_door.txt"
                        with open(results_path, "a") as rf:
                            rf.write(json.dumps({
                                "frame": frame_idx,
                                "time": get_kst_timestamp(),
                                "region_probs": region_probs,
                                "hard_vote": int(hard_vote) if hard_vote is not None else None,
                                "soft_vote": int(soft_vote) if soft_vote is not None else None,
                                "soft_probs": soft_probs,
                                "rf_vote": rf_vote if rf_vote is not None else "N/A",
                            }) + "\n")

            # Save annotated frame image
            annotated_path = annotated_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(annotated_path), annotator.result())

        # Optional: press Ctrl+C to stop; loop runs indefinitely for webcam


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live test pipeline (webcam)")
    parser.add_argument("--config", type=str, default=str(CURRENT_DIR / "live_config.yaml"), help="Path to live_config.yaml")
    args = parser.parse_args()
    run_live_pipeline(args.config)


if __name__ == "__main__":
    main()


