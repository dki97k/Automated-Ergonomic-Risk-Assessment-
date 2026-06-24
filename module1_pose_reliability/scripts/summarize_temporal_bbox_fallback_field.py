#!/usr/bin/env python3
"""Create tables and montage for temporal bbox fallback ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "occlusion_distribution" / "field_temporal_bbox_fallback_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "occlusion_distribution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def make_montage(df: pd.DataFrame, output_path: Path) -> None:
    thumb_size = (320, 220)
    label_h = 64
    cols = 3
    rows = (len(df) + cols - 1) // cols
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

        sx = thumb.width / original_w
        sy = thumb.height / original_h
        if row["fallback_status"] == "recovered":
            bx1 = tx + float(row["fallback_bbox_x1"]) * sx
            by1 = ty + float(row["fallback_bbox_y1"]) * sy
            bx2 = tx + float(row["fallback_bbox_x2"]) * sx
            by2 = ty + float(row["fallback_bbox_y2"]) * sy
            draw.rectangle([bx1, by1, bx2, by2], outline="#ffcc00", width=3)

        draw.rectangle([x0, y0, x0 + thumb_size[0] - 1, y0 + thumb_size[1] - 1], outline="#CCCCCC")
        label = f"{row['sequence']} / {int(row['frame_number']):05d}"
        status = (
            f"{row['fallback_status']} | donor={int(row['donor_frame_number'])} | "
            f"dist={int(row['frame_distance'])}"
        )
        draw.text((x0 + 5, y0 + thumb_size[1] + 5), label, fill="#111111", font=font_bold)
        draw.text((x0 + 5, y0 + thumb_size[1] + 25), status, fill="#7A5A00", font=font)
        draw.text((x0 + 5, y0 + thumb_size[1] + 43), "yellow box = borrowed temporal bbox", fill="#7A5A00", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    opts = parse_args()
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(opts.summary)

    ablation = pd.DataFrame(
        [
            {
                "condition": "YOLO-only",
                "n_frames": len(df),
                "detected_or_recovered": 0,
                "failed": len(df),
                "recovery_rate_percent": 0.0,
            },
            {
                "condition": "YOLO + temporal bbox fallback",
                "n_frames": len(df),
                "detected_or_recovered": int((df["fallback_status"] == "recovered").sum()),
                "failed": int((df["fallback_status"] != "recovered").sum()),
                "recovery_rate_percent": round(float((df["fallback_status"] == "recovered").mean() * 100.0), 1),
            },
        ]
    )
    ablation.to_csv(opts.out_dir / "field_temporal_bbox_fallback_ablation_table.csv", index=False)

    distance = df["frame_distance"].astype(float)
    distance_summary = pd.DataFrame(
        [
            {
                "n_recovered": int((df["fallback_status"] == "recovered").sum()),
                "mean_frame_distance": round(float(distance.mean()), 1),
                "median_frame_distance": round(float(distance.median()), 1),
                "min_frame_distance": int(distance.min()),
                "max_frame_distance": int(distance.max()),
            }
        ]
    )
    distance_summary.to_csv(opts.out_dir / "field_temporal_bbox_fallback_distance_summary.csv", index=False)
    pd.crosstab(df["sequence"], df["fallback_status"]).to_csv(
        opts.out_dir / "field_temporal_bbox_fallback_by_sequence.csv"
    )

    make_montage(df, opts.out_dir / "field_temporal_bbox_fallback_montage.jpg")

    memo = opts.out_dir / "field_temporal_bbox_fallback_memo.md"
    memo.write_text(
        f"""# Temporal Bbox Fallback Ablation

This ablation evaluates the Module 1 temporal bbox fallback on field frames
where YOLO did not detect a person. The fallback borrows the nearest valid
person bbox from the same sequence and runs SAM-3DB body inference.

## Result

- YOLO-only failure frames: {len(df)}
- Recovered with temporal bbox fallback: {int((df['fallback_status'] == 'recovered').sum())}/{len(df)}
- Recovery rate among YOLO failure frames: {float((df['fallback_status'] == 'recovered').mean() * 100.0):.1f}%

## Donor Frame Distance

- Mean distance: {float(distance.mean()):.1f} frames
- Median distance: {float(distance.median()):.1f} frames
- Range: {int(distance.min())}-{int(distance.max())} frames

## Interpretation

These cases support describing Module 1 as more than a direct YOLO + SAM-3DB
assembly. YOLO-only failed under substantial object/self occlusion or unusual
posture, while the temporal bbox fallback allowed SAM-3DB inference to continue
for all reviewed failure frames.

The fallback does not prove pose correctness in these occluded frames. It should
be reported as a detection-continuity/recovery mechanism and paired with
occlusion severity labels and downstream caution/reliability reporting.

## Outputs

- `field_temporal_bbox_fallback_summary.csv`
- `field_temporal_bbox_fallback_ablation_table.csv`
- `field_temporal_bbox_fallback_distance_summary.csv`
- `field_temporal_bbox_fallback_montage.jpg`
""",
        encoding="utf-8",
    )

    print(f"ablation={opts.out_dir / 'field_temporal_bbox_fallback_ablation_table.csv'}")
    print(f"distance={opts.out_dir / 'field_temporal_bbox_fallback_distance_summary.csv'}")
    print(f"montage={opts.out_dir / 'field_temporal_bbox_fallback_montage.jpg'}")
    print(f"memo={memo}")


if __name__ == "__main__":
    main()
