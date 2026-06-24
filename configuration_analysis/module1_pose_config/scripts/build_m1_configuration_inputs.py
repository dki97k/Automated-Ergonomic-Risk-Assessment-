#!/usr/bin/env python3
"""Build fair Module 1 configuration inputs from AlphaPose-MotionBERT and SAM-3DB.

Outputs:
- shared-angle CSVs for posture/duration metrics using only common body joints.
- REP++ JSONL files for repetition estimation using the same common body joints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMMON14 = (
    "pelvis",
    "neck",
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
JI = {name: idx for idx, name in enumerate(COMMON14)}

MHR70_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    "right_wrist": 41,
    "left_wrist": 62,
    "neck": 69,
}

DISPLAY_NAMES = {
    "pelvis": "Pelvis (Origin)",
    "neck": "Neck",
    "left_shoulder": "Shoulder (L)",
    "right_shoulder": "Shoulder (R)",
    "left_elbow": "Elbow (L)",
    "right_elbow": "Elbow (R)",
    "left_wrist": "Wrist (L)",
    "right_wrist": "Wrist (R)",
    "left_hip": "Hip (L)",
    "right_hip": "Hip (R)",
    "left_knee": "Knee (L)",
    "right_knee": "Knee (R)",
    "left_ankle": "Ankle (L)",
    "right_ankle": "Ankle (R)",
}


def mhr70_to_common14(joints_mhr70: np.ndarray) -> np.ndarray:
    joints_mhr70 = np.asarray(joints_mhr70, dtype=np.float64)
    mapped = []
    for name in COMMON14:
        if name == "pelvis":
            mapped.append((joints_mhr70[MHR70_INDEX["left_hip"]] + joints_mhr70[MHR70_INDEX["right_hip"]]) / 2.0)
        else:
            mapped.append(joints_mhr70[MHR70_INDEX[name]])
    return np.stack(mapped, axis=0)


def load_alpha(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        str(sample_id): np.asarray(pose, dtype=np.float64)
        for sample_id, pose in zip(data["sample_ids"], data["pred_common14"], strict=True)
    }


def load_sam_by_sequence(sam_dir: Path) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    for path in sorted(sam_dir.glob("*.npz")):
        if path.name.startswith("._"):
            continue
        data = np.load(path, allow_pickle=True)
        sequence = str(data["sequence"][0])
        frames = data["frame_index"]
        poses = data["keypoints3d_mhr70_m"]
        out[sequence] = {int(frame): mhr70_to_common14(pose) for frame, pose in zip(frames, poses, strict=True)}
    return out


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))))


def vector_onto_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(n)
    if norm == 0:
        return v
    unit = n / norm
    return v - np.dot(v, unit) * unit


SEGMENT_DATA = {
    "Trunk": (0.497, 0.500),
    "UpperArm_L": (0.028, 0.436),
    "UpperArm_R": (0.028, 0.436),
    "Forearm_L": (0.016, 0.430),
    "Forearm_R": (0.016, 0.430),
    "Thigh_L": (0.100, 0.433),
    "Thigh_R": (0.100, 0.433),
    "Shank_L": (0.0465, 0.433),
    "Shank_R": (0.0465, 0.433),
}


def joint_dict(pose: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Pelvis": pose[JI["pelvis"]],
        "NeckBase": pose[JI["neck"]],
        "L_Shoulder": pose[JI["left_shoulder"]],
        "R_Shoulder": pose[JI["right_shoulder"]],
        "L_Elbow": pose[JI["left_elbow"]],
        "R_Elbow": pose[JI["right_elbow"]],
        "L_Wrist": pose[JI["left_wrist"]],
        "R_Wrist": pose[JI["right_wrist"]],
        "L_Hip": pose[JI["left_hip"]],
        "R_Hip": pose[JI["right_hip"]],
        "L_Knee": pose[JI["left_knee"]],
        "R_Knee": pose[JI["right_knee"]],
        "L_Ankle": pose[JI["left_ankle"]],
        "R_Ankle": pose[JI["right_ankle"]],
    }


def total_com(joints: dict[str, np.ndarray]) -> np.ndarray | None:
    weighted = np.zeros(3, dtype=np.float64)
    total = 0.0
    pairs = [
        ("Trunk", "NeckBase", "Pelvis"),
        ("UpperArm_L", "L_Shoulder", "L_Elbow"),
        ("UpperArm_R", "R_Shoulder", "R_Elbow"),
        ("Forearm_L", "L_Elbow", "L_Wrist"),
        ("Forearm_R", "R_Elbow", "R_Wrist"),
        ("Thigh_L", "L_Hip", "L_Knee"),
        ("Thigh_R", "R_Hip", "R_Knee"),
        ("Shank_L", "L_Knee", "L_Ankle"),
        ("Shank_R", "R_Knee", "R_Ankle"),
    ]
    for seg, a, b in pairs:
        if a in joints and b in joints:
            ratio, com_ratio = SEGMENT_DATA[seg]
            weighted += (joints[a] + (joints[b] - joints[a]) * com_ratio) * ratio
            total += ratio
    if total == 0:
        return None
    return weighted / total


def estimate_gravity(poses: list[np.ndarray]) -> np.ndarray:
    candidates = []
    for pose in poses:
        joints = joint_dict(pose)
        com = total_com(joints)
        if com is None:
            continue
        ankle_center = (joints["L_Ankle"] + joints["R_Ankle"]) / 2.0
        vec = ankle_center - com
        norm = np.linalg.norm(vec)
        if norm > 0:
            candidates.append(vec / norm)
    if not candidates:
        return np.array([0.0, -1.0, 0.0])
    med = np.median(np.asarray(candidates), axis=0)
    norm = np.linalg.norm(med)
    return med / norm if norm > 0 else np.array([0.0, -1.0, 0.0])


def trunk_angles(joints: dict[str, np.ndarray], gravity_vec: np.ndarray) -> tuple[float, float, float]:
    top = joints["NeckBase"]
    pelvis = joints["Pelvis"]
    trunk_vector = top - pelvis
    up_vec = -gravity_vec
    raw = angle_between(trunk_vector, up_vec)
    shoulder_axis = joints["R_Shoulder"] - joints["L_Shoulder"]
    forward_vec = np.cross(up_vec, shoulder_axis)
    flexion = raw if np.dot(trunk_vector, forward_vec) > 0 else -raw
    hip_axis = joints["R_Hip"] - joints["L_Hip"]
    bending = angle_between(trunk_vector, hip_axis) - 90.0
    shoulder_line = joints["R_Shoulder"] - joints["L_Shoulder"]
    twisting = angle_between(vector_onto_plane(hip_axis, trunk_vector), vector_onto_plane(shoulder_line, trunk_vector))
    return flexion, bending, twisting


def upper_arm_angles(joints: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    sa = joints["R_Shoulder"] - joints["L_Shoulder"]
    tv = joints["NeckBase"] - joints["Pelvis"]
    ca = np.cross(sa, tv)

    def calc(side: str) -> tuple[float, float]:
        arm = joints[f"{side}_Elbow"] - joints[f"{side}_Shoulder"]
        sign = -1 if side == "R" else 1
        if np.dot(arm, tv) < 0:
            return angle_between(arm, ca) - 90.0, angle_between(arm, sign * sa) - 90.0
        return 270.0 - angle_between(arm, ca), 270.0 - angle_between(arm, sign * sa)

    left_flex, left_abd = calc("L")
    right_flex, right_abd = calc("R")
    return left_flex, right_flex, left_abd, right_abd


def elbow_flexion(joints: dict[str, np.ndarray]) -> tuple[float, float]:
    left = angle_between(joints["L_Shoulder"] - joints["L_Elbow"], joints["L_Wrist"] - joints["L_Elbow"]) - 90.0
    right = angle_between(joints["R_Shoulder"] - joints["R_Elbow"], joints["R_Wrist"] - joints["R_Elbow"]) - 90.0
    return left, right


def knee_angles(joints: dict[str, np.ndarray]) -> tuple[float, float]:
    left = angle_between(joints["L_Hip"] - joints["L_Knee"], joints["L_Knee"] - joints["L_Ankle"])
    right = angle_between(joints["R_Hip"] - joints["R_Knee"], joints["R_Knee"] - joints["R_Ankle"])
    return left, right


def leg_support_ratio(joints: dict[str, np.ndarray]) -> float:
    dl = np.linalg.norm(joints["Pelvis"] - joints["L_Ankle"])
    dr = np.linalg.norm(joints["Pelvis"] - joints["R_Ankle"])
    return float(dr / (dl + dr)) if (dl + dr) > 0 else 0.5


ANGLE_COLUMNS = [
    ("trunk", "flexion"),
    ("trunk", "bending"),
    ("trunk", "twisting"),
    ("upperarm", "left_flexion"),
    ("upperarm", "right_flexion"),
    ("upperarm", "left_abduction"),
    ("upperarm", "right_abduction"),
    ("lower arm", "left_flexion"),
    ("lower arm", "right_flexion"),
    ("knee", "left_flexion"),
    ("knee", "right_flexion"),
    ("leg_support", "ratio"),
]


def angle_row(frame: int, pose: np.ndarray, gravity_vec: np.ndarray) -> dict:
    joints = joint_dict(pose)
    trunk_f, trunk_b, trunk_t = trunk_angles(joints, gravity_vec)
    luf, ruf, lua, rua = upper_arm_angles(joints)
    llf, rlf = elbow_flexion(joints)
    lk, rk = knee_angles(joints)
    return {
        "frame": frame,
        ("trunk", "flexion"): trunk_f,
        ("trunk", "bending"): trunk_b,
        ("trunk", "twisting"): trunk_t,
        ("upperarm", "left_flexion"): luf,
        ("upperarm", "right_flexion"): ruf,
        ("upperarm", "left_abduction"): lua,
        ("upperarm", "right_abduction"): rua,
        ("lower arm", "left_flexion"): llf,
        ("lower arm", "right_flexion"): rlf,
        ("knee", "left_flexion"): lk,
        ("knee", "right_flexion"): rk,
        ("leg_support", "ratio"): leg_support_ratio(joints),
    }


def rep_joints(pose: np.ndarray) -> dict[str, dict[str, float]]:
    joints = {}
    for name, display in DISPLAY_NAMES.items():
        point = pose[JI[name]]
        joints[display] = {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
    spine = (pose[JI["pelvis"]] + pose[JI["neck"]]) / 2.0
    joints["Spine"] = {"x": float(spine[0]), "y": float(spine[1]), "z": float(spine[2])}
    return joints


def write_angle_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    frames = [row["frame"] for row in rows]
    data = [[row[col] for col in ANGLE_COLUMNS] for row in rows]
    df = pd.DataFrame(data, index=frames, columns=pd.MultiIndex.from_tuples(ANGLE_COLUMNS))
    df.index.name = "frame"
    df = df.sort_index()
    for col in df.columns:
        if len(df) > 1:
            delta = df[col].diff().fillna(0.0)
            sd = float(delta.std())
            if sd > 1e-9:
                z = (delta - float(delta.mean())) / sd
                df.loc[z.abs() > 3.0, col] = np.nan
    df.interpolate(method="linear", limit_direction="both", inplace=True)
    df.fillna(0.0, inplace=True)
    df.to_csv(path)


def write_rep_jsonl(rows: list[tuple[int, np.ndarray]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame, pose in sorted(rows, key=lambda item: item[0]):
            handle.write(json.dumps({"frame": int(frame), "joints": rep_joints(pose)}, separators=(",", ":")) + "\n")


def build_condition(
    *,
    condition: str,
    manifest: pd.DataFrame,
    pose_lookup,
    angle_out: Path,
    rep_out: Path,
) -> list[dict]:
    summary = []
    for sequence, group in manifest.groupby("sequence", sort=True):
        angle_rows = []
        rep_rows = []
        poses_for_gravity = []
        frame_pose_pairs = []
        for row in group.itertuples(index=False):
            pose = pose_lookup(row)
            if pose is None:
                continue
            frame = int(row.frame_number)
            frame_pose_pairs.append((frame, pose))
            poses_for_gravity.append(pose)
        gravity = estimate_gravity(poses_for_gravity)
        for frame, pose in frame_pose_pairs:
            angle_rows.append(angle_row(frame, pose, gravity))
            rep_rows.append((frame, pose))

        write_angle_csv(angle_rows, angle_out / condition / f"{sequence}_angle.csv")
        write_rep_jsonl(rep_rows, rep_out / condition / f"{sequence}.jsonl")
        summary.append(
            {
                "condition": condition,
                "sequence": sequence,
                "manifest_rows": len(group),
                "prepared_rows": len(frame_pose_pairs),
                "missing_rows": len(group) - len(frame_pose_pairs),
                "angle_csv": str(angle_out / condition / f"{sequence}_angle.csv"),
                "repetition_jsonl": str(rep_out / condition / f"{sequence}.jsonl"),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("<private_workspace>/configuration_analysis/m1"))
    args = parser.parse_args()

    root = args.root
    manifest_path = root / "inputs/m1_pose/manifest/field_plausibility_by_severity_allframes_temporal_bbox_manifest.csv"
    alpha_path = root / "inputs/m1_pose/alphapose_motionbert/field_plausibility_alphapose_motionbert_allframes_temporal_bbox.npz"
    sam_dir = root / "inputs/m1_pose/sam3db_mhr70_by_sequence"
    angle_out = root / "inputs/shared_angle_csv"
    rep_out = root / "inputs/repetition_jsonl"

    manifest = pd.read_csv(manifest_path)
    alpha = load_alpha(alpha_path)
    sam = load_sam_by_sequence(sam_dir)

    def alpha_lookup(row) -> np.ndarray | None:
        return alpha.get(str(row.sample_id))

    def sam_lookup(row) -> np.ndarray | None:
        return sam.get(str(row.sequence), {}).get(int(row.frame_number))

    summary = []
    summary.extend(build_condition(condition="alphapose_motionbert", manifest=manifest, pose_lookup=alpha_lookup, angle_out=angle_out, rep_out=rep_out))
    summary.extend(build_condition(condition="sam3db", manifest=manifest, pose_lookup=sam_lookup, angle_out=angle_out, rep_out=rep_out))

    out_dir = root / "results/prepared_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out_dir / "m1_configuration_input_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"[ok] wrote {out_dir / 'm1_configuration_input_summary.csv'}")


if __name__ == "__main__":
    main()
