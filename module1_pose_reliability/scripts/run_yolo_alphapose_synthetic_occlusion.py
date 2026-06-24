#!/usr/bin/env python3
"""Validate synthetic occlusion levels with YOLO + AlphaPose confidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALPHAPOSE_ROOT = PROJECT_ROOT / "vendor" / "AlphaPose"
if str(ALPHAPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(ALPHAPOSE_ROOT))

from alphapose.models import builder  # noqa: E402
from alphapose.utils.config import update_config  # noqa: E402
from alphapose.utils.presets import SimpleTransform  # noqa: E402
from alphapose.utils.transforms import get_func_heatmap_to_coord  # noqa: E402
from ultralytics import YOLO  # noqa: E402


BODY_JOINT_INDICES = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
BODY_JOINT_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_occlusion" / "video_cases_c1" / "synthetic_video_case_manifest.csv",
    )
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
        default=PROJECT_ROOT / "results" / "synthetic_occlusion" / "video_cases_c1_yolo_alphapose_severity.csv",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_occlusion" / "video_cases_c1",
    )
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def severity_from_visible_ratio(ratio: float) -> str:
    if ratio >= 0.90:
        return "none"
    if ratio >= 0.50:
        return "partial"
    return "severe"


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


def write_summaries(rows: list[dict[str, str]], summary_dir: Path) -> None:
    import pandas as pd

    summary_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["visible_joint_ratio_body12"] = pd.to_numeric(df["visible_joint_ratio_body12"], errors="coerce")
    df["yolo_conf"] = pd.to_numeric(df["yolo_conf"], errors="coerce")
    grouped = df.groupby("synthetic_level").agg(
        n=("synthetic_image_path", "count"),
        detected=("status", lambda s: int((s == "ok").sum())),
        detection_rate_percent=("status", lambda s: round(100.0 * (s == "ok").mean(), 2)),
        mean_visible_body12=("visible_joint_ratio_body12", "mean"),
        median_visible_body12=("visible_joint_ratio_body12", "median"),
        mean_yolo_conf=("yolo_conf", "mean"),
    ).reset_index()
    grouped.to_csv(summary_dir / "synthetic_level_visibility_summary.csv", index=False)
    pd.crosstab(df["synthetic_level"], df["auto_severity"]).to_csv(summary_dir / "synthetic_level_by_auto_severity.csv")
    pd.crosstab(df["case_id"], df["synthetic_level"], values=df["visible_joint_ratio_body12"], aggfunc="mean").to_csv(
        summary_dir / "synthetic_case_by_level_mean_visible_ratio.csv"
    )

    joint_cols = [f"visible_{name}" for name in BODY_JOINT_NAMES]
    joint_rows = []
    for level, group in df.groupby("synthetic_level"):
        out = {"synthetic_level": level, "n": len(group)}
        for col in joint_cols:
            out[col] = pd.to_numeric(group[col], errors="coerce").mean()
        joint_rows.append(out)
    pd.DataFrame(joint_rows).to_csv(summary_dir / "synthetic_level_joint_visibility_summary.csv", index=False)

    make_montage(df, summary_dir / "synthetic_yolo_alphapose_validation_montage.jpg")


def make_montage(df, output_path: Path) -> None:
    # Middle frame from each case/level.
    selected = []
    for _, group in df.groupby(["case_id", "synthetic_level"], sort=True):
        mid = group.iloc[(group["frame_offset"].astype(int) - 22).abs().argmin()]
        selected.append(mid)
    thumb = (260, 180)
    label_h = 62
    cols = 3
    canvas = Image.new("RGB", (cols * thumb[0], int(np.ceil(len(selected) / cols)) * (thumb[1] + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        bold = font
    for idx, row in enumerate(selected):
        col = idx % cols
        grid_row = idx // cols
        x0 = col * thumb[0]
        y0 = grid_row * (thumb[1] + label_h)
        img = Image.open(row["synthetic_image_path"]).convert("RGB")
        ow, oh = img.size
        im = ImageOps.contain(img, thumb)
        tx = x0 + (thumb[0] - im.width) // 2
        canvas.paste(im, (tx, y0))
        if row["status"] == "ok":
            sx, sy = im.width / ow, im.height / oh
            box = [
                tx + float(row["bbox_x1"]) * sx,
                y0 + float(row["bbox_y1"]) * sy,
                tx + float(row["bbox_x2"]) * sx,
                y0 + float(row["bbox_y2"]) * sy,
            ]
            draw.rectangle(box, outline=(255, 215, 0), width=3)
        draw.rectangle([x0, y0, x0 + thumb[0] - 1, y0 + thumb[1] - 1], outline=(190, 190, 190))
        draw.text((x0 + 5, y0 + thumb[1] + 4), f"{row['case_id']} | {row['synthetic_level']}", font=bold, fill=(130, 45, 20))
        draw.text((x0 + 5, y0 + thumb[1] + 24), f"vis={row['visible_joint_ratio_body12']} | {row['auto_severity']}", font=font, fill=(0, 0, 0))
        draw.text((x0 + 5, y0 + thumb[1] + 42), f"status={row['status']}", font=font, fill=(70, 70, 70))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    opts = parse_args()
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = list(csv.DictReader(opts.synthetic_manifest.open(newline="")))
    print(f"rows={len(manifest_rows)}", flush=True)
    print(f"loading_yolo={opts.yolo_model}", flush=True)
    yolo = YOLO(opts.yolo_model)
    print("loading_alphapose=COCO17 FastPose", flush=True)
    ap_cfg, ap_transform, ap_model = load_alphapose(opts.alphapose_config, opts.alphapose_checkpoint)

    rows = []
    for idx, row in enumerate(manifest_rows, start=1):
        image_bgr = cv2.imread(row["synthetic_image_path"])
        out = dict(row)
        if image_bgr is None:
            out.update({"status": "image_read_failed", "yolo_conf": "", "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "", "visible_joint_ratio_body12": "0.000", "auto_severity": "unknown"})
            rows.append(out)
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        yolo_result = yolo.predict(image_rgb, classes=[0], conf=opts.yolo_conf, verbose=False)[0]
        box, det_conf = choose_primary_person_box(yolo_result)
        if box is None:
            out.update({"status": "no_person_detected", "yolo_conf": "", "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "", "visible_joint_ratio_body12": "0.000", "auto_severity": "severe"})
            for name in BODY_JOINT_NAMES:
                out[f"visible_{name}"] = "0"
            rows.append(out)
        else:
            coords, scores = run_alphapose_on_box(ap_cfg, ap_transform, ap_model, image_rgb, box)
            body_scores = scores[list(BODY_JOINT_INDICES)]
            visible = body_scores >= opts.keypoint_conf_threshold
            ratio = float(visible.mean())
            out.update(
                {
                    "status": "ok",
                    "yolo_conf": f"{det_conf:.4f}",
                    "bbox_x1": f"{box[0]:.2f}",
                    "bbox_y1": f"{box[1]:.2f}",
                    "bbox_x2": f"{box[2]:.2f}",
                    "bbox_y2": f"{box[3]:.2f}",
                    "visible_joint_ratio_body12": f"{ratio:.3f}",
                    "auto_severity": severity_from_visible_ratio(ratio),
                }
            )
            for name, is_visible in zip(BODY_JOINT_NAMES, visible, strict=True):
                out[f"visible_{name}"] = "1" if bool(is_visible) else "0"
            rows.append(out)
        if opts.log_every > 0 and (idx % opts.log_every == 0 or idx == len(manifest_rows)):
            print(f"processed={idx}/{len(manifest_rows)} level={row['synthetic_level']} status={rows[-1]['status']} vis={rows[-1]['visible_joint_ratio_body12']}", flush=True)

    with opts.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_summaries(rows, opts.summary_dir)
    print(f"output={opts.output}", flush=True)
    print(f"summary_dir={opts.summary_dir}", flush=True)


if __name__ == "__main__":
    main()
