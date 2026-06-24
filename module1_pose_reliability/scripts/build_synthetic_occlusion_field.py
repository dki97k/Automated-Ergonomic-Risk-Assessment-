#!/usr/bin/env python3
"""Build synthetic occlusion variants from clean field frames.

This creates a controlled stress-test set: the underlying field image stays
fixed, while the occlusion level is changed by drawing construction-like
occluders over the person bbox.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--field-manifest",
        type=Path,
        default=PROJECT_ROOT / "results" / "occlusion_distribution" / "field_yolo_alphapose_occlusion_severity_allframes.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "synthetic_occlusion" / "field_c1",
    )
    parser.add_argument("--samples-per-sequence", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260617)
    return parser.parse_args()


def choose_base_frames(df: pd.DataFrame, samples_per_sequence: int) -> pd.DataFrame:
    clean = df[
        df["status"].eq("ok")
        & df["auto_severity"].eq("none")
        & (pd.to_numeric(df["visible_joint_ratio_body12"], errors="coerce") >= 1.0)
        & (pd.to_numeric(df["yolo_conf"], errors="coerce") >= 0.70)
    ].copy()
    parts = []
    for _, group in clean.sort_values(["sequence", "frame_number"]).groupby("sequence", sort=True):
        if len(group) <= samples_per_sequence:
            parts.append(group)
            continue
        idx = [round(i * (len(group) - 1) / (samples_per_sequence - 1)) for i in range(samples_per_sequence)]
        parts.append(group.iloc[idx])
    return pd.concat(parts).reset_index(drop=True)


def occluder_box(bbox: tuple[float, float, float, float], level: str, pattern_index: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if level == "none":
        return None
    if level == "partial":
        patterns = (
            (x1 + 0.52 * w, y1 + 0.10 * h, x1 + 0.92 * w, y1 + 0.78 * h),
            (x1 + 0.08 * w, y1 + 0.40 * h, x1 + 0.90 * w, y1 + 0.68 * h),
            (x1 + 0.05 * w, y1 + 0.18 * h, x1 + 0.42 * w, y1 + 0.88 * h),
        )
    elif level == "severe":
        patterns = (
            (x1 + 0.32 * w, y1 + 0.02 * h, x1 + 0.98 * w, y1 + 0.96 * h),
            (x1 + 0.02 * w, y1 + 0.35 * h, x1 + 0.98 * w, y1 + 0.92 * h),
            (x1 + 0.00 * w, y1 + 0.08 * h, x1 + 0.62 * w, y1 + 0.98 * h),
        )
    else:
        raise ValueError(level)
    return patterns[pattern_index % len(patterns)]


def draw_occluder(image: Image.Image, box: tuple[float, float, float, float] | None, level: str, pattern_index: int) -> Image.Image:
    out = image.copy()
    if box is None:
        return out
    draw = ImageDraw.Draw(out)
    colors = {
        "partial": [(104, 102, 96), (132, 98, 62), (94, 111, 122)],
        "severe": [(82, 82, 78), (120, 86, 50), (72, 88, 96)],
    }
    fill = colors[level][pattern_index % 3]
    draw.rectangle(box, fill=fill, outline=(35, 35, 35), width=2)
    # Add faint stripes to make the overlay read as material/equipment rather than an image artifact.
    x1, y1, x2, y2 = [int(v) for v in box]
    for x in range(x1, x2, 18):
        draw.line([(x, y1), (x + (y2 - y1), y2)], fill=tuple(min(255, c + 28) for c in fill), width=1)
    return out


def make_montage(rows: list[dict[str, str]], output_path: Path) -> None:
    sample = rows[: min(len(rows), 36)]
    thumb_size = (280, 190)
    label_h = 58
    cols = 3
    canvas = Image.new("RGB", (cols * thumb_size[0], int(np.ceil(len(sample) / cols)) * (thumb_size[1] + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font
    for idx, row in enumerate(sample):
        col = idx % cols
        grid_row = idx // cols
        x0 = col * thumb_size[0]
        y0 = grid_row * (thumb_size[1] + label_h)
        img = Image.open(row["synthetic_image_path"]).convert("RGB")
        thumb = ImageOps.contain(img, thumb_size)
        tx = x0 + (thumb_size[0] - thumb.width) // 2
        canvas.paste(thumb, (tx, y0))
        draw.rectangle([x0, y0, x0 + thumb_size[0] - 1, y0 + thumb_size[1] - 1], outline=(190, 190, 190))
        draw.text((x0 + 5, y0 + thumb_size[1] + 5), f"{row['base_sequence']} f{int(row['base_frame_number']):05d}", font=font, fill=(0, 0, 0))
        draw.text((x0 + 5, y0 + thumb_size[1] + 25), row["synthetic_level"], font=font_bold, fill=(140, 45, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    opts = parse_args()
    image_dir = opts.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(opts.field_manifest)
    bases = choose_base_frames(df, opts.samples_per_sequence)
    rows: list[dict[str, str]] = []
    levels = ("none", "partial", "severe")

    for base_idx, row in bases.reset_index(drop=True).iterrows():
        image = Image.open(row["frame_path"]).convert("RGB")
        bbox = (float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"]))
        for level in levels:
            box = occluder_box(bbox, level, base_idx)
            out_img = draw_occluder(image, box, level, base_idx)
            synthetic_id = f"syn_{base_idx + 1:03d}_{level}"
            out_path = image_dir / f"{synthetic_id}.jpg"
            out_img.save(out_path, quality=94)
            rows.append(
                {
                    "synthetic_id": synthetic_id,
                    "synthetic_level": level,
                    "synthetic_image_path": str(out_path),
                    "base_sequence": row["sequence"],
                    "base_frame_number": int(row["frame_number"]),
                    "base_frame_path": row["frame_path"],
                    "base_bbox_x1": f"{bbox[0]:.2f}",
                    "base_bbox_y1": f"{bbox[1]:.2f}",
                    "base_bbox_x2": f"{bbox[2]:.2f}",
                    "base_bbox_y2": f"{bbox[3]:.2f}",
                    "occluder_x1": "" if box is None else f"{box[0]:.2f}",
                    "occluder_y1": "" if box is None else f"{box[1]:.2f}",
                    "occluder_x2": "" if box is None else f"{box[2]:.2f}",
                    "occluder_y2": "" if box is None else f"{box[3]:.2f}",
                }
            )

    manifest_path = opts.output_dir / "synthetic_occlusion_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_montage(rows, opts.output_dir / "synthetic_occlusion_montage.jpg")
    print(f"base_frames={len(bases)}")
    print(f"synthetic_images={len(rows)}")
    print(f"manifest={manifest_path}")
    print(f"montage={opts.output_dir / 'synthetic_occlusion_montage.jpg'}")


if __name__ == "__main__":
    main()
