# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog

ANGLE_THRESHOLD = 20.0
FPS = 30  # 영상의 프레임 레이트


# -------------------------------
# 점수 계산 함수
# -------------------------------
def score_neck(flex, bend, twist):
    if flex < -5:
        s_flex = 2
    elif -5 <= flex <= 20:
        s_flex = 1
    else:
        s_flex = 2

    # [FIX 2026-06-21] REBA 정규: 측굴(bend) OR 비틀림(twist) 시 +1 (둘 다여도 단일 +1).
    # 기존: s_bend+s_twist 각각 +1 → 최대 +2로 REBA 위반(정규는 단일 +1).
    adj = 1 if (abs(bend) >= ANGLE_THRESHOLD or abs(twist) >= ANGLE_THRESHOLD) else 0

    return s_flex + adj


def score_trunk(flex, bend, twist):
    if flex < -5:
        s_flex = 2
    elif -5 <= flex <= 5:
        s_flex = 1
    elif 5 < flex <= 20:
        s_flex = 2
    elif 20 < flex <= 60:
        s_flex = 3
    else:
        s_flex = 4

    # [FIX 2026-06-21] REBA 정규: 측굴(bend) OR 비틀림(twist) 시 +1 (둘 다여도 단일 +1).
    # 기존: s_bend+s_twist 각각 +1 → 최대 +2로 REBA 위반(정규는 단일 +1).
    adj = 1 if (abs(bend) >= ANGLE_THRESHOLD or abs(twist) >= ANGLE_THRESHOLD) else 0

    return s_flex + adj


def score_upper_arm(flex, abduction, shoulder_raised=False, supported=False):
    if flex < -20:
        s_flex = 2
    elif -20 <= flex <= 20:
        s_flex = 1
    elif 20 < flex <= 45:
        s_flex = 2
    elif 45 < flex <= 90:
        s_flex = 3
    else:
        s_flex = 4

    s_abd = 1 if abduction > 45 else 0
    s_shoulder = 1 if shoulder_raised else 0
    s_supported = -1 if supported else 0

    return s_flex + s_abd + s_shoulder + s_supported


def score_lower_arm(flex):
    # [FIX 2026-06-21] REBA 하완: 굴곡 60–100° = 1, 그 외 = 2.
    # joint_angle.py의 elbow 'flex'는 (내각 − 90) 컨벤션 → 실제 forearm 굴곡 φ = 90 − flex.
    # 기존 abs(flex)<20 (= 내각[70,110])은 REBA[80,120] 대비 10° 어긋나고
    # knee(0=직선)와 컨벤션도 불일치 → φ 기준 REBA 정범위로 교정.
    phi = 90.0 - flex
    return 1 if 60.0 <= phi <= 100.0 else 2


def score_wrist(flex, twist):
    if -15 <= flex <= 15:
        s_flex = 1
    else:
        s_flex = 2

    s_twist = 1 if not abs(twist) < 45 else 0

    return s_flex + s_twist


def score_leg(knee_angle, support_ratio):
    if knee_angle >= 60:
        s_flex = 2
    elif 30 <= knee_angle < 60:
        s_flex = 1
    else:
        s_flex = 0
    if support_ratio < 0.48 or support_ratio > 0.52:
        s_balance = 2
    else:
        s_balance = 1
    return s_flex + s_balance


def score_coupling(grip="not_assessed"):
    # [FIX 2026-06-21] REBA coupling: good=0, fair=+1, poor=+2, unacceptable=+3.
    # 기존 매핑 키 오류("fit/poor/possible/none") → 기본 grip="fair"가 매핑에 없어
    #   .get(...,1)로 '항상 +1' 상수가 B에 더해지는 버그(매 프레임 B+1 과대편향).
    # partial REBA: coupling/grip은 pose 비관측 → 미평가 시 0(neutral)으로 제외·명시.
    mapping = {"good": 0, "fair": 1, "poor": 2, "unacceptable": 3}
    return mapping.get(grip, 0)


def score_activity(held=False, repeated=False, unstable=False):
    return int(held) + int(repeated) + int(unstable)


# -------------------------------
# REBA Look up Tables``
# -------------------------------
# Table_A = neck, trunk, leg
TABLE_A = np.array(
    [
        [
            [1, 2, 3, 4],  # Trunk 1
            [2, 3, 4, 5],  # Trunk 2
            [2, 4, 5, 6],  # Trunk 3
            [3, 5, 6, 7],  # Trunk 4
            [4, 6, 7, 8],  # Trunk 5
        ],
        [
            [1, 2, 3, 4],  # Trunk 1
            [3, 4, 5, 6],  # Trunk 2
            [4, 5, 6, 7],  # Trunk 3
            [5, 6, 7, 8],  # Trunk 4
            [6, 7, 8, 9],  # Trunk 5
        ],
        [
            [3, 3, 5, 6],  # Trunk 1
            [4, 5, 6, 7],  # Trunk 2
            [5, 6, 7, 8],  # Trunk 3
            [6, 7, 8, 9],  # Trunk 4
            [7, 8, 9, 9],  # Trunk 5
        ],
    ]
)

# Table_B = lower arm, upper arm, wrist
TABLE_B = np.array(
    [
        [[1, 2, 2], [1, 2, 3], [3, 4, 5], [4, 5, 5], [6, 7, 8], [7, 8, 8]],
        [[1, 2, 3], [2, 3, 4], [4, 5, 5], [5, 6, 7], [7, 8, 8], [8, 9, 9]],
    ]
)

# Table_C = Tanle A, Table B
TABLE_C = np.array(
    [
        [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
        [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
        [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
        [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
        [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
        [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
        [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
        [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
        [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
        [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
        [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    ]
)


def reba_frame(row, grip="not_assessed", activity_flags=None):
    # [partial REBA — pose 비관측 항목 제외(명시)]: force/load(A에 미가산),
    # coupling(grip 미평가→0), activity(held/repeated/unstable 기본 0),
    # shoulder-raised(+1)/supported(−1)/upper-arm rotation = 미산출.
    # duration·repetition은 별도 모듈에서 정량(REBA에 합산하지 않음 = 모듈 유기관계 유지).
    # [추가할 코드] activity_flags가 없으면 기본값(False)으로 딕셔너리 생성
    if activity_flags is None:
        activity_flags = {"held": False, "repeated": False, "unstable": False}

    neck = score_neck(
        row[("neck", "flexion")], row[("neck", "bending")], row[("neck", "twisting")]
    )
    trunk = score_trunk(
        row[("trunk", "flexion")], row[("trunk", "bending")], row[("trunk", "twisting")]
    )

    leg = score_leg(
        max(row[("knee", "left_flexion")], row[("knee", "right_flexion")]),
        row[("leg_support", "ratio")],
    )

    # ---------------------------------------------------------
    # Group B: Upper Arms, Lower Arms, Wrists
    #  좌/우 각각 점수를 계산한 뒤, 더 높은 점수(Worst Case)를 채택
    # ---------------------------------------------------------
    upper_L = score_upper_arm(
        row[("upperarm", "left_flexion")], row[("upperarm", "left_abduction")]
    )
    upper_R = score_upper_arm(
        row[("upperarm", "right_flexion")], row[("upperarm", "right_abduction")]
    )
    upper = max(upper_L, upper_R)

    lower_L = score_lower_arm(row[("lower arm", "left_flexion")])
    lower_R = score_lower_arm(row[("lower arm", "right_flexion")])
    lower = max(lower_L, lower_R)

    wrist_L = score_wrist(
        row[("wrist", "left_flexion")], row[("wrist", "left_twisting")]
    )
    wrist_R = score_wrist(
        row[("wrist", "right_flexion")], row[("wrist", "right_twisting")]
    )
    wrist = max(wrist_L, wrist_R)

    # TABLE_A 구조: [Neck][Trunk][Leg]
    idx_neck = min(max(neck, 1), 3) - 1
    idx_trunk = min(max(trunk, 1), 5) - 1
    idx_leg = min(max(leg, 1), 4) - 1

    A = TABLE_A[idx_neck, idx_trunk, idx_leg]

    # TABLE_B 구조: [Upper][Lower][Wrist]
    idx_upper = min(max(upper, 1), 6) - 1
    idx_lower = min(max(lower, 1), 2) - 1  # Lower max는 2
    idx_wrist = min(max(wrist, 1), 3) - 1

    B_raw = TABLE_B[idx_lower, idx_upper, idx_wrist]
    B = B_raw + score_coupling(grip)

    # TABLE_C 구조: [Score A][Score B]
    # 점수 합산 후 Table C 조회 (Load/Coupling 점수는 여기선 제외된 상태, 필요시 추가)
    C_raw = TABLE_C[min(A - 1, 11), min(B - 1, 11)]

    C = C_raw + score_activity(**activity_flags)

    return {
        "Neck": neck,
        "Trunk": trunk,
        "Leg": leg,
        "UpperArm": upper,
        "LowerArm": lower,
        "Wrist": wrist,
        "A": A,
        "B": B,
        "C": C,
        "Final": C,
    }


def reba_timeseries(df):
    return pd.DataFrame([reba_frame(row) for _, row in df.iterrows()])


# -------------------------------
# 📊 통계 요약 함수
# -------------------------------
def calculate_summary(video_nm, df_scores):
    mean_score = df_scores["Final"].mean()
    peak_score = df_scores["Final"].max()
    p90_score = df_scores["Final"].quantile(0.9)

    # High Risk (8~10) — [T14-FIX] REBA 기준 8-10 포함(≥8). 기존 >8은 score=8 누락.
    high_risk_count = df_scores[
        (df_scores["Final"] >= 8) & (df_scores["Final"] <= 10)
    ].shape[0]
    time_high_risk = high_risk_count / FPS

    # Very High Risk (11+)
    very_high_risk_count = df_scores[df_scores["Final"] >= 11].shape[0]
    time_very_high_risk = very_high_risk_count / FPS

    # Overall Risk Level
    if mean_score < 2:
        risk_level = "Negligible"
    elif mean_score < 4:
        risk_level = "Low"
    elif mean_score < 8:
        risk_level = "Medium"
    elif mean_score < 11:
        risk_level = "High"
    else:
        risk_level = "VeryHigh"  # [T3-FIX] 통일: "Very High" → "VeryHigh" (시스템 전체 ordinal 일치)

    # [설계 주의] 여기 risk_level은 MEAN 기반 (summary CSV용).
    # schema JSON (build_schema.py)의 pose.risk_level = P90 기반 — 둘은 다른 집계이며 의도적.
    return {
        "VIDEO_NM": video_nm,
        "Mean REBA score": round(mean_score, 2),
        "Peak REBA score": peak_score,
        "90th percentile score": round(p90_score, 2),
        "Time in high risk (s)": round(time_high_risk, 2),
        "Time in very high risk (s)": round(time_very_high_risk, 2),
        "Overall posture risk level": risk_level,
    }


# -------------------------------
# 파일 처리 로직 (output_dir을 인자로 받음)
# -------------------------------
def process_file(filepath, output_dir):
    try:
        print(f"\n--- 🔄 {os.path.basename(filepath)} 처리 중 ---")

        df_raw = pd.read_csv(filepath, header=[0, 1], index_col=0)
        scores = reba_timeseries(df_raw)
        scores.insert(0, "frame", df_raw.index)

        # 파일명 처리
        input_filename = os.path.basename(filepath)
        name, ext = os.path.splitext(input_filename)
        if name.endswith("_ANGLES"):
            name = name.replace("_ANGLES", "")

        score_filename = (
            f"{name}_SCORE{ext}"  # 파일명: _angle suffix 유지(build_schema.py 의존)
        )

        # [T3-FIX] VIDEO_NM은 _angle suffix 제거(가독성). 파일명과 분리.
        display_name = name[: -len("_angle")] if name.endswith("_angle") else name

        # ✅ 전달받은 output_dir 사용
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, score_filename)

        scores.to_csv(out_path, index=False)
        print(f"✅ 시계열 점수 저장됨: {out_path}")

        summary_data = calculate_summary(display_name, scores)
        return summary_data

    except Exception as e:
        print(f"❌ {os.path.basename(filepath)} 처리 중 오류 발생: {e}")
        return None


# -------------------------------
# 메인 실행 블록
# -------------------------------
if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    # 1. 입력 파일 선택
    filepaths = filedialog.askopenfilenames(
        title="CSV 파일 선택 (여러 개 선택 가능)", filetypes=[("CSV files", "*.csv")]
    )

    if filepaths:
        print(f"✅ 총 {len(filepaths)}개의 파일을 선택했습니다.")

        # 2.  저장할 폴더 선택 (GUI)
        target_output_dir = filedialog.askdirectory(
            title="결과(CSV)를 저장할 폴더를 선택하세요"
        )

        if target_output_dir:
            print(f"📂 결과 저장 경로: {target_output_dir}")

            all_summaries = []

            for filepath in filepaths:
                # ✅ 선택된 폴더 경로를 함수에 전달
                summary = process_file(filepath, target_output_dir)
                if summary:
                    all_summaries.append(summary)

            # 전체 요약 리포트 저장
            if all_summaries:
                df_summary = pd.DataFrame(all_summaries)
                cols = [
                    "VIDEO_NM",
                    "Mean REBA score",
                    "Peak REBA score",
                    "90th percentile score",
                    "Time in high risk (s)",
                    "Time in very high risk (s)",
                    "Overall posture risk level",
                ]
                # 컬럼 재정렬
                df_summary = df_summary.reindex(columns=cols)

                # ✅ 전체 요약 리포트도 사용자가 선택한 폴더에 저장
                summary_path = os.path.join(
                    target_output_dir, "TOTAL_SUMMARY_REPORT.csv"
                )

                df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
                print(f"\n📊 전체 요약 리포트 저장 완료: {summary_path}")

            print("\n--- 모든 작업 완료 ---")

        else:
            print("❌ 폴더를 선택하지 않아 작업을 취소합니다.")

    else:
        print("❌ CSV 파일을 선택하지 않았습니다.")
