# =====================================================
# 파일명: make_dur_consensus.py
# 역할: 3인(R1/R2/R3)이 독립적으로 친 duration static 구간 → *자동* GT_consensus 생성.
#       frame-level 다수결(≥2/3) → rater마다 구간 개수가 달라도 합의 가능.
#       consensus 구간 + 신뢰도(pairwise frame IoU, Fleiss κ) 산출. 이 consensus로 eval_vs_GT가 계산.
# 입력: 02_duration/evaluation/dur_GT/<clip>__R{1,2,3}.csv (keep==YES 구간), 00_joint_angle(프레임수)
# 출력: dur_GT/<clip>_consensus.csv (합의 구간) + _consensus_reliability.csv
# 의존: numpy, pandas.
# =====================================================
import os, glob, numpy as np, pandas as pd
from pathlib import Path

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # module root
os.chdir(ROOT)
FPS = 30.0
GT = Path("02_duration/evaluation/dur_GT")
RATERS = ["R1", "R2", "R3"]


def rater_mask(clip, R, n, f0):
    fp = GT / f"{clip}__{R}.csv"
    m = np.zeros(n, dtype=bool)
    if not fp.exists():
        return None
    d = pd.read_csv(fp)
    if "keep" in d.columns:
        d = d[d["keep"].astype(str).str.upper() == "YES"]
    for _, r in d.iterrows():  # 구간 개수 무관 — 각 구간을 프레임으로 펼침
        s, e = int(r["start_frame"]) - f0, int(r["end_frame"]) - f0
        m[max(0, s) : min(n - 1, e) + 1] = True
    return m


def mask_to_intervals(m, f0):
    out, s = [], None
    for i, v in enumerate(m):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s + f0, i - 1 + f0))
            s = None
    if s is not None:
        out.append((s + f0, len(m) - 1 + f0))
    return out


def fleiss_binary(masks):
    # 3 rater binary/frame → Fleiss κ (category={static,not})
    M = np.vstack(masks).astype(int)  # (3, n)
    n_r = M.shape[0]
    s = M.sum(0)  # static 표 수/frame
    P_i = (s * (s - 1) + (n_r - s) * (n_r - s - 1)) / (n_r * (n_r - 1))
    Pbar = P_i.mean()
    p_static = s.mean() / n_r
    Pe = p_static**2 + (1 - p_static) ** 2
    return (Pbar - Pe) / (1 - Pe) if (1 - Pe) else 1.0


rows_c, rel = [], []
for f in sorted(glob.glob("00_joint_angle/*_angle.csv")):
    clip = os.path.basename(f).replace("_angle.csv", "")
    idx = pd.read_csv(f, header=[0, 1], index_col=0).index.to_numpy()
    n, f0 = len(idx), int(idx[0])
    masks = [rater_mask(clip, R, n, f0) for R in RATERS]
    masks = [m for m in masks if m is not None]
    if len(masks) < 2:
        continue
    votes = np.sum(masks, axis=0)
    consensus = votes >= 2  # 다수결(3인 중 2+ → static 합의)
    iv = mask_to_intervals(consensus, f0)
    pd.DataFrame(
        [
            dict(
                interval_id=i + 1,
                start_frame=s,
                end_frame=e,
                start_sec=round(s / FPS, 1),
                end_sec=round(e / FPS, 1),
                dur_s=round((e - s + 1) / FPS, 1),
            )
            for i, (s, e) in enumerate(iv)
        ]
    ).to_csv(GT / f"{clip}_consensus.csv", index=False, encoding="utf-8-sig")
    # 신뢰도: pairwise frame IoU + Fleiss κ + rater별 구간수(개수 차이 확인용)
    pious = []
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            inter = (masks[a] & masks[b]).sum()
            uni = (masks[a] | masks[b]).sum()
            pious.append(inter / uni if uni else 1.0)
    nseg = [len(mask_to_intervals(m, f0)) for m in masks]
    rel.append(
        dict(
            clip=clip,
            n_raters=len(masks),
            rater_nseg=str(nseg),
            consensus_nseg=len(iv),
            pairwise_frame_IoU=round(float(np.mean(pious)), 3),
            fleiss_kappa=round(float(fleiss_binary(masks)), 3),
        )
    )

pd.DataFrame(rel).to_csv(
    GT / "_consensus_reliability.csv", index=False, encoding="utf-8-sig"
)
print("=== duration GT consensus (frame-level 다수결, rater 구간수 무관) ===")
print(
    pd.DataFrame(rel).to_string(index=False)
    if rel
    else "  (rater 파일 부족: dur_GT/<clip>__R{1,2,3}.csv 필요)"
)
print(f"\n[저장] {GT}/<clip>_consensus.csv + _consensus_reliability.csv")
print(
    "[핵심] rater별 구간 개수(rater_nseg)가 달라도 frame 다수결로 consensus 생성 → eval_vs_GT가 이 consensus로 계산."
)
