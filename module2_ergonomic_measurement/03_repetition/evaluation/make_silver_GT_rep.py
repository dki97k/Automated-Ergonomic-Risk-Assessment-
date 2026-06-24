# =====================================================
# 파일명: make_silver_GT_rep.py
# 역할: repetition 평가용 *silver(자동 1차) GT* — 사람이 '한 반복의 시작/끝 구간을 1로' 체크하는 표준 interval 방식의
#       검수 출발점. 시스템(SSM/RepNet) 파이프라인과 *독립*이도록, 가장 많이 움직이는 관절 각도의
#       주기적 극값(find_peaks)으로 rep 구간(연속 peak 사이=1 cycle)을 생성. → 사람이 영상보고 verified 수정.
# 입력: 00_joint_angle/*_angle.csv (+ 03_repetition/rep_period_frozen.csv: 주기 힌트로 peak 최소간격)
# 출력: 03_repetition/evaluation/silver_GT_rep/<clip>_silverGT_reps.csv (interval당 start/end/verified) + _summary.csv
# 의존: numpy, pandas, scipy.signal.find_peaks. 검수 방법 = 동 폴더 README.
# =====================================================
import os, glob, numpy as np, pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # module root
os.chdir(ROOT)
FPS = 30.0

# 반복 동작이 가장 잘 드러나는 후보 관절(원위 상지 우선) — 그중 변동(std) 최대 관절 자동선택
CAND = [
    ("wrist", "left_flexion"),
    ("wrist", "right_flexion"),
    ("lower arm", "left_flexion"),
    ("lower arm", "right_flexion"),
    ("upperarm", "left_flexion"),
    ("upperarm", "right_flexion"),
    ("trunk", "flexion"),
]
_fr = pd.read_csv("03_repetition/rep_period_frozen.csv")
PERIOD = dict(zip(_fr["clip"], _fr["cycle_sec"]))

outdir = Path("03_repetition/evaluation/silver_GT_rep")
outdir.mkdir(parents=True, exist_ok=True)
summary = []
for f in sorted(glob.glob("00_joint_angle/*_angle.csv")):
    clip = os.path.basename(f).replace("_angle.csv", "")
    df = pd.read_csv(f, header=[0, 1], index_col=0)
    cols = [c for c in CAND if c in df.columns]
    # 변동 최대 관절 선택(가장 활발히 움직이는 = 반복 주체)
    j = max(cols, key=lambda c: df[c].std())
    sig = df[j].rolling(5, center=True, min_periods=1).mean().to_numpy()  # 평활
    per_f = PERIOD.get(clip, 6.0) * FPS  # 주기 힌트(frame)
    min_dist = max(10, int(per_f * 0.45))  # 과검출 방지(주기의 ~절반)
    prom = max(3.0, np.nanstd(sig) * 0.5)  # 최소 prominence
    peaks, _ = find_peaks(sig, distance=min_dist, prominence=prom)
    frames = df.index.to_numpy()
    rows = []
    for i in range(len(peaks) - 1):  # 연속 peak 사이 = 1 rep cycle 구간
        s, e = peaks[i], peaks[i + 1]
        rows.append(
            dict(
                rep_id=len(rows) + 1,
                start_frame=int(frames[s]),
                end_frame=int(frames[e]),
                start_sec=round(frames[s] / FPS, 1),
                end_sec=round(frames[e] / FPS, 1),
                dur_s=round((e - s) / FPS, 2),
                look_at_frame=int(frames[(s + e) // 2]),  # 검수 시 이 프레임 영상 확인
                source_joint=f"{j[0]}.{j[1]}",
                verified="NO",
            )
        )
    pd.DataFrame(rows).to_csv(
        outdir / f"{clip}_silverGT_reps.csv", index=False, encoding="utf-8-sig"
    )
    summary.append(
        dict(
            clip=clip,
            source_joint=f"{j[0]}.{j[1]}",
            n_reps_silver=len(rows),
            n_frames=len(df),
        )
    )
pd.DataFrame(summary).to_csv(
    outdir / "_silverGT_rep_summary.csv", index=False, encoding="utf-8-sig"
)
print(
    "=== repetition silver GT (분석기 독립·연속 peak 사이=1 rep 구간) — 검수 전 초안 ==="
)
print(pd.DataFrame(summary).to_string(index=False))
print(
    f"\n[저장] {outdir}/<clip>_silverGT_reps.csv (rep당 start/end/verified) + _silverGT_rep_summary.csv"
)
print(
    "[검수] look_at_frame 영상 확인 → 진짜 반복 구간만 verified=YES, start/end 보정. → eval_vs_GT_rep.py"
)
