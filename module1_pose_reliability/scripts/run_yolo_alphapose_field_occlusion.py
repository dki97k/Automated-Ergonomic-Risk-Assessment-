#!/usr/bin/env python3
"""Estimate field occlusion severity with YOLO person boxes + AlphaPose confidence.

This script creates a first-pass occlusion severity manifest for the field
construction images. It does not claim to classify occlusion source
automatically; object/self/mixed source labels should be manually reviewed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHAPOSE_ROOT = PROJECT_ROOT / "vendor" / "AlphaPose"
if str(ALPHAPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(ALPHAPOSE_ROOT))

from alphapose.models import builder  # noqa: E402
from alphapose.utils.config import update_config  # noqa: E402
from alphapose.utils.presets import SimpleTransform  # noqa: E402
from alphapose.utils.transforms import get_func_heatmap_to_coord  # noqa: E402
from ultralytics import YOLO  # noqa: E402


FIELD_SEQUENCES = (
    "MansoryBrickLaying_00",
    "MansoryBrickLaying_01",
    "MansoryBrickLaying_02",
    "MansoryCement_02",
    "RebarPlacement_00",
    "RebarTying_01",
    "RebarTying_02",
    "WallPlacement_00",
)

BODY_JOINT_INDICES = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)


@dataclass(frozen=True)
class FieldFrame:
    sequence: str
    frame_number: int
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("<private_workspace>/data"))
    parser.add_argument("--samples-per-sequence", type=int, default=24)
    parser.add_argument(
        "--sequences",
        nargs="*",
        default=list(FIELD_SEQUENCES),
        help="Field sequence directory names to evaluate. Defaults to all known field sequences.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--keypoint-conf-threshold", type=float, default=0.40)
    parser.add_argument(
        "--alphapose-config",
        type=Path,
        default=ALPHAPOSE_ROOT / "configs" / "coco" / "resnet" / "256x192_res50_lr1e-3_1x.yaml",
    )
    parser.add_argument(
        "--alphapose-checkpoint",
        type=Path,
        default=ALPHAPOSE_ROOT / "pretrained_models" / "fast_res50_256x192.pth",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "occlusion_distribution" / "field_yolo_alphapose_occlusion_severity.csv",
    )
    return parser.parse_args()


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match:
        return int(match.group(1))
    try:
        return int(path.stem)
    except ValueError:
        return -1


def iter_sequence_frames(data_root: Path, sequence: str) -> list[FieldFrame]:
    seq_dir = data_root / sequence
    if not seq_dir.exists():
        return []
    frames = []
    for path in sorted(seq_dir.glob("*.jpg")):
        if path.name.startswith("._"):
            continue
        frames.append(FieldFrame(sequence=sequence, frame_number=frame_number(path), image_path=path))
    return sorted(frames, key=lambda item: (item.frame_number, str(item.image_path)))


def stratified_sample(frames: list[FieldFrame], n: int) -> list[FieldFrame]:
    if n <= 0 or len(frames) <= n:
        return frames
    indices = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[index] for index in indices]


def severity_from_visible_ratio(ratio: float) -> str:
    if ratio >= 0.90:
        return "none"
    if ratio >= 0.70:
        return "mild"
    if ratio >= 0.50:
        return "moderate"
    if ratio >= 0.20:
        return "severe"
    return "full"


def choose_primary_person_box(result) -> tuple[np.ndarray | None, float]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, float("nan")
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    person_mask = cls == 0
    if not person_mask.any():
        return None, float("nan")
    xyxy = xyxy[person_mask]
    conf = conf[person_mask]
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(np.argmax(areas))
    return xyxy[idx].astype(np.float32), float(conf[idx])


def load_alphapose(config_path: Path, checkpoint_path: Path):
    cfg = update_config(str(config_path))
    device = torch.device("cpu")
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    transform = SimpleTransform(
        pose_dataset,
        scale_factor=0,
        input_size=cfg.DATA_PRESET.IMAGE_SIZE,
        output_size=cfg.DATA_PRESET.HEATMAP_SIZE,
        rot=0,
        sigma=cfg.DATA_PRESET.SIGMA,
        train=False,
        add_dpg=False,
        gpu_device=device,
    )
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    state = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return cfg, transform, model


def run_alphapose_on_box(cfg, transform, model, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inp, cropped_box = transform.test_transform(image_rgb, torch.from_numpy(box_xyxy.astype(np.float32)))
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    with torch.no_grad():
        heatmap = model(inp.unsqueeze(0)).cpu()[0]
    coords, scores = heatmap_to_coord(
        heatmap,
        cropped_box,
        hm_shape=cfg.DATA_PRESET.HEATMAP_SIZE,
        norm_type=cfg.LOSS.get("NORM_TYPE", None),
    )
    return np.asarray(coords, dtype=np.float64), np.asarray(scores, dtype=np.float64).reshape(-1)


def main() -> None:
    opts = parse_args()
    opts.output.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for sequence in opts.sequences:
        frames.extend(stratified_sample(iter_sequence_frames(opts.data_root, sequence), opts.samples_per_sequence))
    if opts.max_frames is not None:
        frames = frames[: opts.max_frames]

    print(f"loading_yolo={opts.yolo_model}", flush=True)
    yolo = YOLO(opts.yolo_model)
    print("loading_alphapose=COCO17 FastPose", flush=True)
    ap_cfg, ap_transform, ap_model = load_alphapose(opts.alphapose_config, opts.alphapose_checkpoint)

    fieldnames = [
        "sequence",
        "frame_number",
        "frame_path",
        "status",
        "yolo_conf",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "visible_joint_ratio_body12",
        "visible_joint_ratio_all17",
        "mean_keypoint_conf_body12",
        "mean_keypoint_conf_all17",
        "auto_severity",
        "occlusion_present_auto",
        "source",
        "review_status",
    ]
    rows = []
    handle = opts.output.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()

    for idx, frame in enumerate(frames, start=1):
        image_bgr = cv2.imread(str(frame.image_path))
        if image_bgr is None:
            row = {
                    "sequence": frame.sequence,
                    "frame_number": frame.frame_number,
                    "frame_path": str(frame.image_path),
                    "status": "image_read_failed",
                    "yolo_conf": "",
                    "bbox_x1": "",
                    "bbox_y1": "",
                    "bbox_x2": "",
                    "bbox_y2": "",
                    "visible_joint_ratio_body12": "",
                    "visible_joint_ratio_all17": "",
                    "mean_keypoint_conf_body12": "",
                    "mean_keypoint_conf_all17": "",
                    "auto_severity": "unknown",
                    "occlusion_present_auto": "unknown",
                    "source": "needs_manual_review",
                    "review_status": "needs_manual_review",
            }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        yolo_result = yolo.predict(image_rgb, classes=[0], conf=opts.yolo_conf, verbose=False)[0]
        box, det_conf = choose_primary_person_box(yolo_result)
        if box is None:
            row = {
                    "sequence": frame.sequence,
                    "frame_number": frame.frame_number,
                    "frame_path": str(frame.image_path),
                    "status": "no_person_detected",
                    "yolo_conf": "",
                    "bbox_x1": "",
                    "bbox_y1": "",
                    "bbox_x2": "",
                    "bbox_y2": "",
                    "visible_joint_ratio_body12": "0.000",
                    "visible_joint_ratio_all17": "0.000",
                    "mean_keypoint_conf_body12": "",
                    "mean_keypoint_conf_all17": "",
                    "auto_severity": "full",
                    "occlusion_present_auto": "yes",
                    "source": "needs_manual_review",
                    "review_status": "needs_manual_review",
            }
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            if opts.log_every > 0 and (idx % opts.log_every == 0 or idx == len(frames)):
                print(f"[{idx}/{len(frames)}] no_person {frame.sequence}/{frame.frame_number:05d}", flush=True)
            continue

        coords, scores = run_alphapose_on_box(ap_cfg, ap_transform, ap_model, image_rgb, box)
        body_scores = scores[list(BODY_JOINT_INDICES)]
        body_ratio = float((body_scores >= opts.keypoint_conf_threshold).mean())
        all_ratio = float((scores >= opts.keypoint_conf_threshold).mean())
        severity = severity_from_visible_ratio(body_ratio)
        row = {
                "sequence": frame.sequence,
                "frame_number": frame.frame_number,
                "frame_path": str(frame.image_path),
                "status": "ok",
                "yolo_conf": f"{det_conf:.4f}",
                "bbox_x1": f"{box[0]:.2f}",
                "bbox_y1": f"{box[1]:.2f}",
                "bbox_x2": f"{box[2]:.2f}",
                "bbox_y2": f"{box[3]:.2f}",
                "visible_joint_ratio_body12": f"{body_ratio:.3f}",
                "visible_joint_ratio_all17": f"{all_ratio:.3f}",
                "mean_keypoint_conf_body12": f"{float(body_scores.mean()):.4f}",
                "mean_keypoint_conf_all17": f"{float(scores.mean()):.4f}",
                "auto_severity": severity,
                "occlusion_present_auto": "no" if severity == "none" else "yes",
                "source": "needs_manual_review",
                "review_status": "needs_manual_review",
        }
        rows.append(row)
        writer.writerow(row)
        if idx % 25 == 0:
            handle.flush()
        if opts.log_every > 0 and (idx % opts.log_every == 0 or idx == len(frames)):
            print(
                f"[{idx}/{len(frames)}] ok {frame.sequence}/{frame.frame_number:05d} "
                f"det={det_conf:.2f} visible_body={body_ratio:.2f} severity={severity}",
                flush=True,
            )

    handle.close()

    print(f"output={opts.output}", flush=True)
    print(f"rows={len(rows)}", flush=True)
    for severity in ["none", "mild", "moderate", "severe", "full", "unknown"]:
        count = sum(row["auto_severity"] == severity for row in rows)
        if count:
            print(f"auto_severity_{severity}={count}", flush=True)


if __name__ == "__main__":
    main()
