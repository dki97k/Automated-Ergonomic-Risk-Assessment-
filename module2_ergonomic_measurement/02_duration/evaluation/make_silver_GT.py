# =====================================================
# 파일명: make_silver_GT.py
# 역할: duration 평가용 *silver(자동 1차) GT* 생성 — 사람이 검수·수정할 출발점.
#       2단계: (1)전신 static 구간(postural core=trunk OR knee 유지) → (2)그 안에서 *관절별* static.
#       ★분석기(SD-속도 튜닝)와 독립이도록 'range 기반'(각도가 t_min 동안 δ° 이내=정지).
# 입력: 00_joint_angle/*_angle.csv
# 출력: evaluation/silver_GT/<clip>_silverGT_intervals.csv (검수용, 관절별 컬럼) + _silverGT_summary.csv
# 의존: numpy, pandas. 검수 방법 = silver_GT/README.md.
# =====================================================
import os, glob, numpy as np, pandas as pd
from pathlib import Path

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)

FPS = 30.0
T_MIN = 120  # frames(4s, ISO 11226 하한)
MIN_RUN = 30  # frames(1s) — 너무 짧은 전신-static 구간 제외

# ── 관절별 static 판정 (개별 각도열). δ = 그 관절이 t_min 동안 이 범위 이내면 static. ──
# ★현재 전 관절 동일 δ=15°(분석기-독립·단순). 관절별로 다르게 두려면 값만 바꾸면 됨(검수 시 조정).
#   (분석기 duration_analyzer는 부위계수가 다르지만, GT는 그 튜닝과 *독립*이어야 순환을 피함.)
JOINTS = {
    "neck_flex": (("neck", "flexion"), 15.0),
    "trunk_flex": (("trunk", "flexion"), 15.0),
    "trunk_bend": (("trunk", "bending"), 15.0),
    "trunk_twist": (("trunk", "twisting"), 15.0),
    "knee_L": (("knee", "left_flexion"), 15.0),
    "knee_R": (("knee", "right_flexion"), 15.0),
    "uparm_L": (("upperarm", "left_flexion"), 15.0),
    "uparm_R": (("upperarm", "right_flexion"), 15.0),
    "lowarm_L": (("lower arm", "left_flexion"), 15.0),
    "lowarm_R": (("lower arm", "right_flexion"), 15.0),
    "wrist_L": (("wrist", "left_flexion"), 15.0),
    "wrist_R": (("wrist", "right_flexion"), 15.0),
}
CORE = [
    "trunk_flex",
    "knee_L",
    "knee_R",
]  # 전신 static = postural core 유지(몸통 또는 무릎)


def joint_static(df, col, delta):
    """centered t_min 창 내 (max-min)<δ → True(그 관절 held still). range 기반(속도 아님)."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    r = df[col].rolling(T_MIN, center=True, min_periods=max(2, T_MIN // 2))
    return (r.max() - r.min()) < delta


def runs(mask):
    out, s = [], None
    arr = mask.to_numpy()
    for i, v in enumerate(arr):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(arr) - 1))
    return out


outdir = Path("02_duration/evaluation/silver_GT")
outdir.mkdir(parents=True, exist_ok=True)
summary = []
for f in sorted(glob.glob("00_joint_angle/*_angle.csv")):
    clip = os.path.basename(f).replace("_angle.csv", "")
    df = pd.read_csv(f, header=[0, 1], index_col=0)
    jstat = {
        name: joint_static(df, col, d) for name, (col, d) in JOINTS.items()
    }  # 관절별 static
    jmat = pd.DataFrame(jstat)
    wb = jmat[CORE].any(axis=1)  # 전신 static = core 중 하나라도 유지
    rows = []
    for s, e in runs(wb):
        if (e - s + 1) < MIN_RUN:
            continue
        seg = jmat.iloc[s : e + 1]
        rows.append(
            dict(
                interval_id=len(rows) + 1,
                start_frame=int(df.index[s]),
                end_frame=int(df.index[e]),
                start_sec=round(int(df.index[s]) / FPS, 1),
                end_sec=round(int(df.index[e]) / FPS, 1),
                dur_s=round((e - s + 1) / FPS, 1),
                look_at_frame=int(
                    df.index[(s + e) // 2]
                ),  # 검수 시 이 프레임 영상 확인
                # 관절별: 구간 내 50%↑ static이면 1 (검수 시 영상보고 0/1 수정)
                **{name: int(seg[name].mean() >= 0.5) for name in JOINTS},
                verified="NO",
            )
        )
    pd.DataFrame(rows).to_csv(
        outdir / f"{clip}_silverGT_intervals.csv", index=False, encoding="utf-8-sig"
    )
    summary.append(
        dict(
            clip=clip,
            n_frames=len(df),
            n_intervals=len(rows),
            wb_static_sec=round(float(wb.sum()) / FPS, 1),
            wb_static_ratio_pct=round(100 * wb.sum() / len(df), 1),
        )
    )
pd.DataFrame(summary).to_csv(
    outdir / "_silverGT_summary.csv", index=False, encoding="utf-8-sig"
)
print("=== silver GT (range 기반·관절별·분석기 독립) — 검수 전 초안 ===")
print(pd.DataFrame(summary).to_string(index=False))
print(
    f"\n[저장] {outdir}/<clip>_silverGT_intervals.csv (관절별 컬럼) + _silverGT_summary.csv"
)
print(
    "[형식] interval당: 시작/끝(frame+sec)·look_at_frame·관절12개 static(0/1)·verified."
)
print(
    "[δ] 현재 전 관절 15°(동일). 관절별로 바꾸려면 JOINTS dict의 값 수정. 검수→eval_vs_GT.py."
)
