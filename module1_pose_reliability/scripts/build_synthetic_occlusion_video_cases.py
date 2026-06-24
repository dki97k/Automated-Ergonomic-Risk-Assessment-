#!/usr/bin/env python3
"""Build case-level synthetic occlusion video clips from field sequences."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SEQUENCE_RULES = {
    "MansoryBrickLaying_00": {"visible": 1.00, "conf": 0.70},
    "MansoryBrickLaying_01": {"visible": 1.00, "conf": 0.70},
    "MansoryBrickLaying_02": {"visible": 1.00, "conf": 0.70},
    "MansoryCement_02": {"visible": 1.00, "conf": 0.70},
    "RebarPlacement_00": {"visible": 1.00, "conf": 0.70},
    "RebarTying_01": {"visible": 1.00, "conf": 0.70},
    # This sequence is naturally more cluttered; use a relaxed clean-enough base.
    "RebarTying_02": {"visible": 0.83, "conf": 0.40},
    "WallPlacement_00": {"visible": 1.00, "conf": 0.70},
}


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
        default=PROJECT_ROOT / "outputs" / "synthetic_occlusion" / "video_cases_c1",
    )
    parser.add_argument("--clip-length", type=int, default=45)
    parser.add_argument("--write-images", action="store_true", default=True)
    return parser.parse_args()


def consecutive_runs(numbers: list[int]) -> list[tuple[int, int, int]]:
    if not numbers:
        return []
    runs = []
    start = prev = numbers[0]
    for number in numbers[1:]:
        if number == prev + 1:
            prev = number
            continue
        runs.append((start, prev, prev - start + 1))
        start = prev = number
    runs.append((start, prev, prev - start + 1))
    return runs


def pick_clip(df: pd.DataFrame, sequence: str, clip_length: int) -> pd.DataFrame:
    rule = DEFAULT_SEQUENCE_RULES[sequence]
    seq = df[df["sequence"].eq(sequence)].copy()
    seq["visible"] = pd.to_numeric(seq["visible_joint_ratio_body12"], errors="coerce")
    seq["conf"] = pd.to_numeric(seq["yolo_conf"], errors="coerce")
    clean = seq[
        seq["status"].eq("ok")
        & (seq["visible"] >= rule["visible"])
        & (seq["conf"] >= rule["conf"])
    ].copy()
    numbers = sorted(clean["frame_number"].astype(int).tolist())
    runs = [run for run in consecutive_runs(numbers) if run[2] >= clip_length]
    if not runs:
        raise RuntimeError(f"no {clip_length}-frame clean run found for {sequence}")
    run = max(runs, key=lambda item: item[2])
    start = run[0] + max(0, (run[2] - clip_length) // 2)
    end = start + clip_length - 1
    clip = clean[(clean["frame_number"] >= start) & (clean["frame_number"] <= end)].sort_values("frame_number")
    if len(clip) != clip_length:
        raise RuntimeError(f"selected clip for {sequence} has {len(clip)} frames, expected {clip_length}")
    return clip


def occluder_box(bbox: tuple[float, float, float, float], level: str, t: int, total: int, case_index: int) -> tuple[float, float, float, float] | None:
    if level == "none":
        return None
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    phase = 0.0 if total <= 1 else t / (total - 1)
    sway = math.sin(2.0 * math.pi * phase) * 0.05
    if level == "partial":
        # V2 moderate occlusion: hide one body side or lower body while leaving
        # enough head/torso evidence for person detection to remain plausible.
        if case_index % 3 == 0:
            ox1 = x1 + (0.52 + sway) * w
            ox2 = x1 + (0.98 + sway) * w
            oy1 = y1 + 0.18 * h
            oy2 = y1 + 0.96 * h
        elif case_index % 3 == 1:
            ox1 = x1 + 0.02 * w
            ox2 = x1 + 0.98 * w
            oy1 = y1 + (0.50 + 0.03 * math.sin(2.0 * math.pi * phase)) * h
            oy2 = y1 + 1.02 * h
        else:
            ox1 = x1 + (0.02 + sway) * w
            ox2 = x1 + (0.48 + sway) * w
            oy1 = y1 + 0.18 * h
            oy2 = y1 + 0.96 * h
    elif level == "severe":
        # Strong occlusion: large object/material slab covering most of the body.
        if case_index % 3 == 0:
            ox1 = x1 + (0.18 + sway) * w
            ox2 = x1 + (0.98 + sway) * w
            oy1 = y1 + 0.02 * h
            oy2 = y1 + 1.00 * h
        elif case_index % 3 == 1:
            ox1 = x1 + 0.02 * w
            ox2 = x1 + 0.98 * w
            oy1 = y1 + (0.25 + 0.03 * math.sin(2.0 * math.pi * phase)) * h
            oy2 = y1 + 1.02 * h
        else:
            ox1 = x1 + (0.00 + sway) * w
            ox2 = x1 + (0.78 + sway) * w
            oy1 = y1 + 0.02 * h
            oy2 = y1 + 1.00 * h
    else:
        raise ValueError(level)
    return (max(0.0, ox1), max(0.0, oy1), max(0.0, ox2), max(0.0, oy2))


def draw_occluder(image: Image.Image, box: tuple[float, float, float, float] | None, level: str, case_index: int) -> Image.Image:
    out = image.copy()
    if box is None:
        return out
    draw = ImageDraw.Draw(out)
    palette = {
        "partial": [(104, 102, 96), (132, 98, 62), (94, 111, 122)],
        "severe": [(82, 82, 78), (120, 86, 50), (72, 88, 96)],
    }
    fill = palette[level][case_index % 3]
    draw.rectangle(box, fill=fill, outline=(35, 35, 35), width=2)
    x1, y1, x2, y2 = [int(v) for v in box]
    for x in range(x1, x2 + 1, 20):
        draw.line([(x, y1), (x + (y2 - y1), y2)], fill=tuple(min(255, c + 25) for c in fill), width=1)
    return out


def make_montage(rows: list[dict[str, str]], output_path: Path) -> None:
    # Use the middle frame of each case/level.
    middle = {}
    for row in rows:
        key = (row["case_id"], row["synthetic_level"])
        if key not in middle or abs(int(row["frame_offset"]) - 22) < abs(int(middle[key]["frame_offset"]) - 22):
            middle[key] = row
    sample = list(middle.values())
    thumb_size = (260, 180)
    label_h = 55
    cols = 3
    canvas = Image.new("RGB", (cols * thumb_size[0], math.ceil(len(sample) / cols) * (thumb_size[1] + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        bold = font
    for idx, row in enumerate(sample):
        col = idx % cols
        grid_row = idx // cols
        x0 = col * thumb_size[0]
        y0 = grid_row * (thumb_size[1] + label_h)
        img = Image.open(row["synthetic_image_path"]).convert("RGB")
        thumb = ImageOps.contain(img, thumb_size)
        canvas.paste(thumb, (x0 + (thumb_size[0] - thumb.width) // 2, y0))
        draw.rectangle([x0, y0, x0 + thumb_size[0] - 1, y0 + thumb_size[1] - 1], outline=(190, 190, 190))
        draw.text((x0 + 5, y0 + thumb_size[1] + 4), f"{row['case_id']} | {row['synthetic_level']}", font=bold, fill=(130, 45, 20))
        draw.text((x0 + 5, y0 + thumb_size[1] + 24), row["sequence"], font=font, fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> None:
    opts = parse_args()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    image_root = opts.output_dir / "images"
    if opts.write_images:
        image_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(opts.field_manifest)
    sequences = [seq for seq in DEFAULT_SEQUENCE_RULES if seq in set(df["sequence"])]
    rows: list[dict[str, str]] = []
    case_rows = []
    levels = ("none", "partial", "severe")

    for case_index, sequence in enumerate(sequences, start=1):
        clip = pick_clip(df, sequence, opts.clip_length)
        case_id = f"case_{case_index:02d}_{sequence}"
        case_rows.append(
            {
                "case_id": case_id,
                "sequence": sequence,
                "start_frame": int(clip["frame_number"].iloc[0]),
                "end_frame": int(clip["frame_number"].iloc[-1]),
                "clip_length": len(clip),
                "clean_visible_threshold": DEFAULT_SEQUENCE_RULES[sequence]["visible"],
                "clean_conf_threshold": DEFAULT_SEQUENCE_RULES[sequence]["conf"],
            }
        )
        for offset, (_, row) in enumerate(clip.iterrows()):
            image = Image.open(row["frame_path"]).convert("RGB")
            bbox = (float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"]))
            for level in levels:
                box = occluder_box(bbox, level, offset, opts.clip_length, case_index)
                rel_dir = Path(case_id) / level
                out_path = image_root / rel_dir / f"image_{int(row['frame_number']):05d}.jpg"
                if opts.write_images:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_img = draw_occluder(image, box, level, case_index)
                    out_img.save(out_path, quality=94)
                rows.append(
                    {
                        "case_id": case_id,
                        "sequence": sequence,
                        "frame_offset": offset,
                        "base_frame_number": int(row["frame_number"]),
                        "base_frame_path": row["frame_path"],
                        "synthetic_level": level,
                        "synthetic_image_path": str(out_path),
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

    manifest_path = opts.output_dir / "synthetic_video_case_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    case_path = opts.output_dir / "synthetic_video_case_summary.csv"
    with case_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)
    if opts.write_images:
        make_montage(rows, opts.output_dir / "synthetic_video_case_montage.jpg")

    print(f"cases={len(case_rows)}")
    print(f"rows={len(rows)}")
    print(f"manifest={manifest_path}")
    print(f"case_summary={case_path}")
    if opts.write_images:
        print(f"montage={opts.output_dir / 'synthetic_video_case_montage.jpg'}")


if __name__ == "__main__":
    main()
