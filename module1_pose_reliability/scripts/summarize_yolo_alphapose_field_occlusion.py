#!/usr/bin/env python3
"""Summarize YOLO + AlphaPose field occlusion severity results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "occlusion_distribution" / "field_yolo_alphapose_occlusion_severity.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "occlusion_distribution"

SEVERITY_COLORS = {
    "none": "#2ca02c",
    "mild": "#ffbf00",
    "moderate": "#ff7f0e",
    "severe": "#d62728",
    "full": "#7f0000",
    "unknown": "#777777",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def write_count_table(df: pd.DataFrame, column: str, path: Path) -> None:
    table = df[column].value_counts(dropna=False).rename_axis(column).reset_index(name="n")
    table["percent"] = (table["n"] / len(df) * 100).round(1)
    table.to_csv(path, index=False)


def draw_overlay_montage(df: pd.DataFrame, output_path: Path) -> None:
    thumb_size = (300, 210)
    label_h = 54
    cols = 4
    rows = int(np.ceil(len(df) / cols))
    canvas = Image.new("RGB", (cols * thumb_size[0], rows * (thumb_size[1] + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 13)
        font_bold = ImageFont.truetype("Arial Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font

    for idx, row in df.reset_index(drop=True).iterrows():
        col = idx % cols
        grid_row = idx // cols
        x0 = col * thumb_size[0]
        y0 = grid_row * (thumb_size[1] + label_h)
        image = Image.open(row["frame_path"]).convert("RGB")
        original_w, original_h = image.size
        thumb = ImageOps.contain(image, thumb_size)
        tx = x0 + (thumb_size[0] - thumb.width) // 2
        ty = y0
        canvas.paste(thumb, (tx, ty))

        severity = str(row["auto_severity"])
        color = SEVERITY_COLORS.get(severity, "#777777")
        if row["status"] == "ok":
            sx = thumb.width / original_w
            sy = thumb.height / original_h
            bx1 = tx + float(row["bbox_x1"]) * sx
            by1 = ty + float(row["bbox_y1"]) * sy
            bx2 = tx + float(row["bbox_x2"]) * sx
            by2 = ty + float(row["bbox_y2"]) * sy
            draw.rectangle([bx1, by1, bx2, by2], outline=color, width=3)

        draw.rectangle([x0, y0, x0 + thumb_size[0] - 1, y0 + thumb_size[1] - 1], outline="#CCCCCC")
        label = f"{row['sequence']} / {int(row['frame_number']):05d}"
        metric = (
            f"{severity} | status={row['status']} | visible={row['visible_joint_ratio_body12']}"
            if row["status"] == "ok"
            else f"{severity} | status={row['status']}"
        )
        draw.text((x0 + 5, y0 + thumb_size[1] + 5), label, fill="#111111", font=font)
        draw.text((x0 + 5, y0 + thumb_size[1] + 25), metric, fill=color, font=font_bold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    opts = parse_args()
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(opts.input)

    write_count_table(df, "status", opts.out_dir / "field_yolo_alphapose_status_summary.csv")
    write_count_table(df, "auto_severity", opts.out_dir / "field_yolo_alphapose_severity_summary.csv")
    pd.crosstab(df["sequence"], df["auto_severity"]).to_csv(
        opts.out_dir / "field_yolo_alphapose_sequence_by_severity.csv"
    )
    pd.crosstab(df["sequence"], df["status"]).to_csv(
        opts.out_dir / "field_yolo_alphapose_sequence_by_status.csv"
    )

    draw_overlay_montage(df, opts.out_dir / "field_yolo_alphapose_severity_overlay_montage.jpg")

    memo = opts.out_dir / "field_yolo_alphapose_occlusion_memo.md"
    status_counts = df["status"].value_counts()
    severity_counts = df["auto_severity"].value_counts()
    ok_df = df[df["status"] == "ok"].copy()
    ok_counts = ok_df["auto_severity"].value_counts()
    memo.write_text(
        f"""# YOLO + AlphaPose Field Occlusion Severity Draft

This is a first-pass automatic severity estimate for 120 stratified field
frames. YOLO provides the primary person bounding box, and AlphaPose COCO-17
keypoint confidence is used to compute visible joint ratio.

## Primary Automatic Measure

- Primary ratio: `visible_joint_ratio_body12`
- Body joints: shoulders, elbows, wrists, hips, knees, ankles
- Visibility threshold: keypoint confidence >= 0.40

## Severity Rule

| Severity | Visible body-joint ratio |
| --- | ---: |
| none | >= 0.90 |
| mild | 0.70-0.89 |
| moderate | 0.50-0.69 |
| severe | 0.20-0.49 |
| full | < 0.20 |

## Draft Results

All sampled frames:

{severity_counts.to_string()}

Detection status:

{status_counts.to_string()}

Frames with successful YOLO detection only:

{ok_counts.to_string()}

## Interpretation Caveat

`no_person_detected` frames are not automatically treated as confirmed full
occlusion in the manuscript. They should be manually reviewed because they may
reflect small scale, unusual posture, low detector confidence, or true severe
occlusion.

Object/self/mixed source labels are not inferred automatically from confidence
alone and should be manually reviewed using the overlay montage.

## Outputs

- `field_yolo_alphapose_occlusion_severity.csv`
- `field_yolo_alphapose_severity_summary.csv`
- `field_yolo_alphapose_sequence_by_severity.csv`
- `field_yolo_alphapose_severity_overlay_montage.jpg`
""",
        encoding="utf-8",
    )
    print(f"input={opts.input}")
    print(f"rows={len(df)}")
    print(f"memo={memo}")
    print(f"montage={opts.out_dir / 'field_yolo_alphapose_severity_overlay_montage.jpg'}")


if __name__ == "__main__":
    main()
