import sys
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from tkinter import Tk, filedialog
from scipy.stats import zscore

# =========================================================
# Part 0. Gravity Estimation using Center of Mass (CoM)
# =========================================================

# Dempster's Anthropometric Data
# "Space requirements of the seated operator"
SEGMENT_DATA = {
    "Head": (0.081, 0.500),
    "Trunk": (0.497, 0.500),
    "UpperArm_L": (0.028, 0.436),
    "UpperArm_R": (0.028, 0.436),
    "Forearm_L": (0.016, 0.430),
    "Forearm_R": (0.016, 0.430),
    "Hand_L": (0.006, 0.506),
    "Hand_R": (0.006, 0.506),
    "Thigh_L": (0.100, 0.433),
    "Thigh_R": (0.100, 0.433),
    "Shank_L": (0.0465, 0.433),
    "Shank_R": (0.0465, 0.433),
    "Foot_L": (0.0145, 0.500),
    "Foot_R": (0.0145, 0.500),
}


def calculate_total_com(joints):
    weighted_pos_sum = np.array([0.0, 0.0, 0.0])
    total_mass = 0.0

    if "NeckBase" in joints and "Pelvis" in joints:
        p1, p2 = joints["NeckBase"], joints["Pelvis"]
        ratio, com_ratio = SEGMENT_DATA["Trunk"]
        weighted_pos_sum += (p1 + (p2 - p1) * com_ratio) * ratio
        total_mass += ratio

    if "Head" in joints and "NeckBase" in joints:
        p1, p2 = joints["Head"], joints["NeckBase"]
        ratio, com_ratio = SEGMENT_DATA["Head"]
        weighted_pos_sum += (p1 + (p2 - p1) * com_ratio) * ratio
        total_mass += ratio

    for side in ["L", "R"]:
        if f"{side}_Shoulder" in joints and f"{side}_Elbow" in joints:
            p1, p2 = joints[f"{side}_Shoulder"], joints[f"{side}_Elbow"]
            ratio, com_ratio = SEGMENT_DATA[f"UpperArm_{side}"]
            weighted_pos_sum += (p1 + (p2 - p1) * com_ratio) * ratio
            total_mass += ratio
        if f"{side}_Elbow" in joints and f"{side}_Wrist" in joints:
            p1, p2 = joints[f"{side}_Elbow"], joints[f"{side}_Wrist"]
            ratio, com_ratio = SEGMENT_DATA[f"Forearm_{side}"]
            weighted_pos_sum += (p1 + (p2 - p1) * com_ratio) * ratio
            total_mass += ratio
        if f"{side}_Hip" in joints and f"{side}_Knee" in joints:
            p1, p2 = joints[f"{side}_Hip"], joints[f"{side}_Knee"]
            ratio, com_ratio = SEGMENT_DATA[f"Thigh_{side}"]
            weighted_pos_sum += (p1 + (p2 - p1) * com_ratio) * ratio
            total_mass += ratio
        if f"{side}_Knee" in joints and f"{side}_Ankle" in joints:
            p1, p2 = joints[f"{side}_Knee"], joints[f"{side}_Ankle"]
            ratio, com_ratio = SEGMENT_DATA[f"Shank_{side}"]
            weighted_pos_sum += (p1 + (p2 - p1) * com_ratio) * ratio
            total_mass += ratio

    if total_mass == 0:
        return None
    return weighted_pos_sum / total_mass


def estimate_calibration_data(file_list):
    """
    1. 중력 벡터: CoM -> AnkleCenter (Median 사용)
    """
    gravity_candidates = []
    print("🌍 [Pre-Process] 무게중심(CoM) 분석 중...")

    for filepath in tqdm(file_list, desc="Scanning Gravity"):
        with open(filepath, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    raw_joints = data.get("joints", {})

                    def get_kpt(idx):
                        if isinstance(raw_joints, dict):
                            v = raw_joints.get(str(idx)) or raw_joints.get(idx)
                        elif isinstance(raw_joints, list) and idx < len(raw_joints):
                            v = raw_joints[idx]
                        else:
                            return None
                        if v and isinstance(v, dict):
                            return np.array([v["x"], v["y"], v["z"]])
                        if v:
                            return np.array(v)
                        return None

                    joints = {}
                    indices = {
                        "L_Hip": 9,
                        "R_Hip": 10,
                        "L_Knee": 11,
                        "R_Knee": 12,
                        "L_Ankle": 13,
                        "R_Ankle": 14,
                        "L_Shoulder": 5,
                        "R_Shoulder": 6,
                        "L_Elbow": 7,
                        "R_Elbow": 8,
                        "L_Wrist": 62,
                        "R_Wrist": 41,
                        "Neck": 69,
                        "L_Ear": 3,
                        "R_Ear": 4,
                    }
                    for name, idx in indices.items():
                        coord = get_kpt(idx)
                        if coord is not None:
                            joints[name] = coord

                    if "L_Hip" in joints and "R_Hip" in joints:
                        joints["Pelvis"] = (joints["L_Hip"] + joints["R_Hip"]) / 2
                    if "Neck" in joints:
                        joints["NeckBase"] = joints["Neck"]
                    elif "L_Shoulder" in joints and "R_Shoulder" in joints:
                        joints["NeckBase"] = (
                            joints["L_Shoulder"] + joints["R_Shoulder"]
                        ) / 2
                    if "L_Ear" in joints and "R_Ear" in joints:
                        joints["Head"] = (joints["L_Ear"] + joints["R_Ear"]) / 2

                    if (
                        "Pelvis" not in joints
                        or "NeckBase" not in joints
                        or "L_Ankle" not in joints
                        or "R_Ankle" not in joints
                    ):
                        continue

                    # 중력 추정
                    com = calculate_total_com(joints)
                    if com is None:
                        continue
                    ankle_center = (joints["L_Ankle"] + joints["R_Ankle"]) / 2
                    grav_vec = ankle_center - com
                    gravity_candidates.append(grav_vec)

                except Exception:
                    continue

    if not gravity_candidates:
        print("⚠️ 데이터 부족으로 기본값 사용")
        return np.array([0, -1, 0])

    vecs_array = np.array(gravity_candidates)
    median_vec = np.median(vecs_array, axis=0)
    gravity_vec = median_vec / np.linalg.norm(median_vec)
    print(f"✅ CoM 기반 중력 벡터: {np.round(gravity_vec, 3)}")

    return gravity_vec


# =========================================================
# Part 1. Angle Analysis Modules
# =========================================================
def get_angle_between(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(v1 / norm_v1, v2 / norm_v2), -1.0, 1.0)))


def vector_onto_plane(v, n):
    return v - np.dot(v, n) * n


# ---------------------------------------------------------
# Trunk Angles (Direction Logic Fixed)
# ---------------------------------------------------------
def get_trunk_angles(joints, gravity_vec):
    if "NeckBase" in joints:
        top = joints["NeckBase"]
    else:
        top = (joints["L_Shoulder"] + joints["R_Shoulder"]) / 2
    pelvis = joints["Pelvis"]
    trunk_vector = top - pelvis

    # 1. 절대 각도 (0 ~ 180도)
    up_vec = -gravity_vec
    raw_angle = get_angle_between(trunk_vector, up_vec)

    # 2. 방향 판별 (Forward vs Backward)
    shoulder_axis = joints["R_Shoulder"] - joints["L_Shoulder"]  # Left -> Right

    # [수정됨] Cross Product 순서 변경: (Up) x (Shoulder) = Forward
    # 오른손 법칙: 검지(Up) x 중지(Right) = 엄지(Forward/Front)
    forward_vec = np.cross(up_vec, shoulder_axis)

    # 내적(Dot Product): Trunk Vector가 Forward와 같은 방향이면 양수
    if np.dot(trunk_vector, forward_vec) > 0:
        final_flexion = raw_angle  # Flexion (Forward) -> Positive (+)
    else:
        final_flexion = -raw_angle  # Extension (Backward) -> Negative (-)

    # Bending & Twisting
    hip_axis = joints["R_Hip"] - joints["L_Hip"]
    bending = get_angle_between(trunk_vector, hip_axis) - 90
    shoulder_line = joints["R_Shoulder"] - joints["L_Shoulder"]
    twisting = get_angle_between(
        vector_onto_plane(hip_axis, trunk_vector),
        vector_onto_plane(shoulder_line, trunk_vector),
    )

    return final_flexion, bending, twisting


# (나머지 함수 동일)
def get_leg_support_ratio(joints):
    dl = np.linalg.norm(joints["Pelvis"] - joints["L_Ankle"])
    dr = np.linalg.norm(joints["Pelvis"] - joints["R_Ankle"])
    return dr / (dl + dr) if (dl + dr) > 0 else 0.5


def get_knee_angles(joints):
    # [컨벤션 2026-06-21] 반환값 = 0=직선(다리 폄), 굴곡↑(쪼그릴수록 +).
    #   ⚠️ elbow(get_elbow_flexion)는 (내각−90) 컨벤션(직선→+90)이라 부위 간 0점 위치가 다름.
    #   knee는 REBA leg score(60/30 임계)가 raw 그대로 소비 → 환산 불필요(R1).
    l = get_angle_between(
        joints["L_Hip"] - joints["L_Knee"], joints["L_Knee"] - joints["L_Ankle"]
    )
    r = get_angle_between(
        joints["R_Hip"] - joints["R_Knee"], joints["R_Knee"] - joints["R_Ankle"]
    )
    return l, r


def get_wrist_angles(joints):
    def flex(s):
        return get_angle_between(
            joints[f"{s}_Middle_3rd_Joint"] - joints[f"{s}_Wrist"],
            joints[f"{s}_Wrist"] - joints[f"{s}_Elbow"],
        )

    def twist(s):
        la = joints[f"{s}_Wrist"] - joints[f"{s}_Elbow"]
        fa = joints[f"{s}_Thumb_3rd_Joint"] - joints[f"{s}_Pinky_3rd_Joint"]
        ea = joints[f"{s}_Cubital_Fossa"] - joints[f"{s}_Olecranon"]
        return (
            get_angle_between(vector_onto_plane(fa, la), vector_onto_plane(ea, la)) - 40
        )

    return flex("L"), flex("R"), twist("L"), twist("R")


def get_elbow_flexion(joints):
    # [주의 2026-06-21] 반환값 = (내각 − 90) 컨벤션: 직선 팔 → +90, 직각 → 0, 완전굽힘 → −90.
    #   knee(get_knee_angles)는 0=직선 컨벤션이라 부위 간 불일치가 있음.
    #   실제 REBA forearm 굴곡 φ = 90 − (반환값). REBA_table.score_lower_arm이 φ로 환산해 채점함.
    #   [향후 클린업] 일관성을 위해 (180 − 내각) = (90 − 현재반환값)로 바꿔 0=직선으로 통일 권장
    #   (단 그 경우 angle CSV 재생성 필요 → 본 복사본은 기존 CSV 호환 위해 컨벤션 유지).
    l = (
        get_angle_between(
            joints["L_Shoulder"] - joints["L_Elbow"],
            joints["L_Wrist"] - joints["L_Elbow"],
        )
        - 90
    )
    r = (
        get_angle_between(
            joints["R_Shoulder"] - joints["R_Elbow"],
            joints["R_Wrist"] - joints["R_Elbow"],
        )
        - 90
    )
    return l, r


def get_upper_arm_angles(joints):
    sa = joints["R_Shoulder"] - joints["L_Shoulder"]
    tv = joints["NeckBase"] - joints["Pelvis"]
    ca = np.cross(sa, tv)

    def calc(s):
        av = joints[f"{s}_Elbow"] - joints[f"{s}_Shoulder"]
        sign = -1 if s == "R" else 1
        if np.dot(av, tv) < 0:
            return get_angle_between(av, ca) - 90, get_angle_between(av, sign * sa) - 90
        else:
            return 270 - get_angle_between(av, ca), 270 - get_angle_between(
                av, sign * sa
            )

    lf, la = calc("L")
    rf, ra = calc("R")
    return lf, rf, la, ra


def get_neck_angles(joints):
    nv = joints["Head"] - joints["NeckBase"]
    tv = joints["NeckBase"] - joints["Pelvis"]
    sa = joints["R_Shoulder"] - joints["L_Shoulder"]
    ca = np.cross(sa, tv)
    f = (
        get_angle_between(nv, ca) - 90
        if np.dot(nv, tv) > 0
        else 270 - get_angle_between(nv, ca)
    )
    b = get_angle_between(nv, sa) - 90
    t = get_angle_between(
        vector_onto_plane(sa, nv),
        vector_onto_plane(joints["R_Ear"] - joints["L_Ear"], nv),
    )
    return f, b, t


# =========================================================
# Part 2. Main Execution
# =========================================================

RAW_INDICES = {
    "L_Ear": 3,
    "R_Ear": 4,
    "L_Shoulder": 5,
    "R_Shoulder": 6,
    "L_Elbow": 7,
    "R_Elbow": 8,
    "L_Hip": 9,
    "R_Hip": 10,
    "L_Knee": 11,
    "R_Knee": 12,
    "L_Ankle": 13,
    "R_Ankle": 14,
    "R_Wrist": 41,
    "L_Wrist": 62,
    "Neck": 69,
    "L_Thumb_3rd_Joint": 45,
    "R_Thumb_3rd_Joint": 24,
    "L_Pinky_3rd_Joint": 61,
    "R_Pinky_3rd_Joint": 40,
    "L_Middle_3rd_Joint": 53,
    "R_Middle_3rd_Joint": 32,
    "L_Cubital_Fossa": 65,
    "R_Cubital_Fossa": 66,
    "L_Olecranon": 63,
    "R_Olecranon": 64,
}


def process_jsonl_file(filepath, gravity_vec):
    results = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                frame_id = int(data["frame"])
                joints_data = data["joints"]
                raw_kpts = {}
                if isinstance(joints_data, dict):
                    for k, v in joints_data.items():
                        if isinstance(v, dict):
                            raw_kpts[int(k)] = np.array([v["x"], v["y"], v["z"]])
                        else:
                            raw_kpts[int(k)] = np.array(v)
                elif isinstance(joints_data, list):
                    for i, v in enumerate(joints_data):
                        if isinstance(v, dict):
                            raw_kpts[i] = np.array([v["x"], v["y"], v["z"]])
                        else:
                            raw_kpts[i] = np.array(v)

                joints = {}
                for name, idx in RAW_INDICES.items():
                    if idx in raw_kpts:
                        joints[name] = raw_kpts[idx]

                if "L_Hip" in joints and "R_Hip" in joints:
                    joints["Pelvis"] = (joints["L_Hip"] + joints["R_Hip"]) / 2
                if "Neck" in joints:
                    joints["NeckBase"] = joints["Neck"]
                elif "L_Shoulder" in joints and "R_Shoulder" in joints:
                    joints["NeckBase"] = (
                        joints["L_Shoulder"] + joints["R_Shoulder"]
                    ) / 2
                if "L_Ear" in joints and "R_Ear" in joints:
                    joints["Head"] = (joints["L_Ear"] + joints["R_Ear"]) / 2

                req = [
                    "Pelvis",
                    "NeckBase",
                    "Head",
                    "L_Shoulder",
                    "R_Shoulder",
                    "L_Elbow",
                    "R_Elbow",
                    "L_Knee",
                    "R_Knee",
                    "L_Wrist",
                    "R_Wrist",
                ]
                if not all(k in joints for k in req):
                    continue

                # Calculations
                # [partial REBA — pose 비관측 항목은 여기서 애초에 미산출(명시)]:
                #   force/load, coupling(grip), shoulder-raise(+1), supported(−1),
                #   upper-arm rotation = 키네마틱으로 추정 불가 → ② REBA에서 0(neutral) 처리.
                #   (§0 partial-REBA 논지와 정합. 산출 각도는 전부 pose-admissible만.)
                neck_f, neck_b, neck_t = get_neck_angles(joints)
                trunk_f, trunk_b, trunk_t = get_trunk_angles(joints, gravity_vec)
                luf, ruf, lua, rua = get_upper_arm_angles(joints)
                llf, rlf = get_elbow_flexion(joints)
                lk, rk = get_knee_angles(joints)
                lsr = get_leg_support_ratio(joints)
                lwf, rwf, lwt, rwt = get_wrist_angles(joints)

                results.append(
                    {
                        "frame": frame_id,
                        "neck_flexion": neck_f,
                        "neck_bending": neck_b,
                        "neck_twsting": neck_t,
                        "trunk_flexion": trunk_f,
                        "trunk_bending": trunk_b,
                        "trunk_twisting": trunk_t,
                        "left_upper_arm_flexion": luf,
                        "right_upper_arm_flexion": ruf,
                        "left_upper_arm_abduction": lua,
                        "right_upper_arm_abduction": rua,
                        "left_lower_arm_flexion": llf,
                        "right_lower_arm_flexion": rlf,
                        "left_wrist_flexion": lwf,
                        "right_wrist_flexion": rwf,
                        "left_wrist_twisting": lwt,
                        "right_wrist_twisting": rwt,
                        "left_knee_flexion": lk,
                        "right_knee_flexion": rk,
                        "leg_support_ratio": lsr,
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(results)


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    print("STEP 1. JSONL 파일 선택")
    input_files = filedialog.askopenfilenames(filetypes=[("JSONL", "*.jsonl")])
    if not input_files:
        sys.exit(1)
    print("STEP 2. 저장 폴더 선택")
    target_output_dir = filedialog.askdirectory()
    if not target_output_dir:
        sys.exit(1)

    global_gravity_vec = estimate_calibration_data(input_files)

    all_frames_data = []
    print(f"총 {len(input_files)}개 파일 처리 시작...")

    for input_file in tqdm(input_files):
        df = process_jsonl_file(input_file, global_gravity_vec)
        if not df.empty:
            all_frames_data.append(df)

    if all_frames_data:
        print("\n🔄 병합 및 후처리...")
        combined_df = pd.concat(all_frames_data, ignore_index=True)
        combined_df = combined_df.sort_values(by="frame").set_index("frame")

        cols = [
            ("neck", "flexion"),
            ("neck", "bending"),
            ("neck", "twisting"),
            ("trunk", "flexion"),
            ("trunk", "bending"),
            ("trunk", "twisting"),
            ("upperarm", "left_flexion"),
            ("upperarm", "right_flexion"),
            ("upperarm", "left_abduction"),
            ("upperarm", "right_abduction"),
            ("lower arm", "left_flexion"),
            ("lower arm", "right_flexion"),
            ("wrist", "left_flexion"),
            ("wrist", "right_flexion"),
            ("wrist", "left_twisting"),
            ("wrist", "right_twisting"),
            ("knee", "left_flexion"),
            ("knee", "right_flexion"),
            ("leg_support", "ratio"),
        ]
        if len(combined_df.columns) == len(cols):
            combined_df.columns = pd.MultiIndex.from_tuples(cols)

        # [후처리 2026-06-21] 트래킹 jitter 완화: 프레임간 변화량(diff)의 |z|>3인
        #   급변 프레임만 NaN→선형보간→0. = 측정 노이즈의 *보간*이지 데이터 합성 아님(R5 무결성 유지).
        for col in combined_df.columns:
            if len(combined_df) > 1:
                delta = combined_df[col].diff().fillna(0)
                if np.std(delta) > 1e-6:
                    combined_df.loc[
                        np.abs(zscore(delta, nan_policy="omit")) > 3.0, col
                    ] = np.nan
        combined_df.interpolate(method="linear", limit_direction="both", inplace=True)
        combined_df.fillna(0, inplace=True)

        path = os.path.join(
            target_output_dir,
            f"{os.path.basename(os.path.dirname(input_files[0]))}_angle.csv",
        )
        combined_df.to_csv(path)
        print(f"\n✅ 저장 완료: {path}")
    else:
        print("❌ 데이터 없음")
