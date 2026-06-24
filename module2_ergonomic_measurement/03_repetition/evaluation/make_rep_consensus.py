# =====================================================
# 파일명: make_rep_consensus.py
# 역할: 3인(R1/R2/R3)이 독립적으로 친 repetition 구간 → 자동 GT_consensus.
#       ★repetition은 reps가 back-to-back이라 duration식 frame-다수결을 쓰면 한 덩어리로 붕괴 →
#       *rep 이벤트(개별 반복) 단위 매칭*: 각 rater의 rep 중심을 시간축에서 군집화, ≥2 rater 지지 군집 = consensus rep.
#       (철학은 duration과 동일: 3인 독립 → 자동 consensus → eval. 단 count 보존 위해 event-기반.)
# 입력: 03_repetition/evaluation/rep_GT/<clip>__R{1,2,3}.csv (keep==YES), 03_repetition/rep_period_frozen.csv(주기 tol)
# 출력: rep_GT/<clip>_consensus.csv (consensus rep 구간) + _consensus_reliability.csv
# 의존: numpy, pandas.
# =====================================================
import os, glob, numpy as np, pandas as pd
from pathlib import Path

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # module root
os.chdir(ROOT)
FPS = 30.0
GT = Path("03_repetition/evaluation/rep_GT")
RATERS = ["R1", "R2", "R3"]
_fr = pd.read_csv("03_repetition/rep_period_frozen.csv")
PERIOD = dict(zip(_fr["clip"], _fr["cycle_sec"]))


def rater_reps(clip, R):
    """rater R의 rep 이벤트 목록 = [(center, start, end), ...]. 구간 개수 무관."""
    fp = GT / f"{clip}__{R}.csv"
    if not fp.exists():
        return None
    d = pd.read_csv(fp)
    if "keep" in d.columns:
        d = d[d["keep"].astype(str).str.upper() == "YES"]
    return [
        (
            (int(r["start_frame"]) + int(r["end_frame"])) / 2,
            int(r["start_frame"]),
            int(r["end_frame"]),
        )
        for _, r in d.iterrows()
    ]


rel = []
for f in sorted(glob.glob("00_joint_angle/*_angle.csv")):
    clip = os.path.basename(f).replace("_angle.csv", "")
    per_reps = {R: rater_reps(clip, R) for R in RATERS}
    per_reps = {R: v for R, v in per_reps.items() if v is not None}
    if len(per_reps) < 2:
        continue
    tol = PERIOD.get(clip, 6.0) * FPS * 0.5  # 군집 허용오차 = 주기의 절반
    # 모든 rater의 rep 이벤트를 (center, rater, start, end)로 모아 center 순 정렬 후 그리디 군집화
    events = sorted((c, R, s, e) for R, v in per_reps.items() for (c, s, e) in v)
    clusters, cur = [], []
    for ev in events:
        if cur and ev[0] - cur[-1][0] > tol:
            clusters.append(cur)
            cur = []
        cur.append(ev)
    if cur:
        clusters.append(cur)
    cons = []
    for cl in clusters:
        raters = {r for _, r, _, _ in cl}
        if len(raters) >= 2:  # ≥2 rater 지지 = consensus rep (count 보존)
            ss = [s for _, _, s, _ in cl]
            ee = [e for _, _, _, e in cl]
            cons.append((int(np.median(ss)), int(np.median(ee)), len(raters)))
    pd.DataFrame(
        [
            dict(
                rep_id=i + 1,
                start_frame=s,
                end_frame=e,
                support_raters=k,
                start_sec=round(s / FPS, 1),
                end_sec=round(e / FPS, 1),
            )
            for i, (s, e, k) in enumerate(cons)
        ]
    ).to_csv(GT / f"{clip}_consensus.csv", index=False, encoding="utf-8-sig")
    nrep = {R: len(v) for R, v in per_reps.items()}
    rel.append(
        dict(
            clip=clip,
            n_raters=len(per_reps),
            rater_nrep=str(list(nrep.values())),
            consensus_nrep=len(cons),
            mean_support=round(np.mean([k for *_, k in cons]), 2) if cons else 0.0,
        )
    )

pd.DataFrame(rel).to_csv(
    GT / "_consensus_reliability.csv", index=False, encoding="utf-8-sig"
)
print("=== repetition GT consensus (event-기반: ≥2 rater 지지 rep, count 보존) ===")
print(pd.DataFrame(rel).to_string(index=False) if rel else "  (rater 파일 부족)")
print(
    f"\n[저장] {GT}/<clip>_consensus.csv + _consensus_reliability.csv → eval_vs_GT_rep.py가 이 consensus로 계산"
)
print(
    "[차이] duration=frame 다수결(gap 있음) / repetition=event 군집 다수결(back-to-back이라 count 보존)."
)
