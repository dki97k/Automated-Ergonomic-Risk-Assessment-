# =====================================================
# 파일명: eval_vs_GT_rep.py
# 역할: repetition 검출을 *사람이 친 rep 구간 GT*(표준 interval 방식)에 대해 평가 — count·localization F1·frame IoU.
#       (구 consistency=검출 peak 평가는 precision-only·종속 → 본 방식=독립 GT 평가로 전환.)
# 입력: 03_repetition/evaluation/silver_GT_rep/<clip>_silverGT_reps.csv (GT 구간; verified 우선),
#       시스템 검출 peak = rep_peak_eval/Repetition frames_kk.xlsx (clip별 peak_frame), 00_joint_angle(프레임수).
# 출력: stdout 표(clip별 + overall). 검수 후 USE_VERIFIED_ONLY=True로 최종.
# 의존: numpy, pandas, openpyxl.
# =====================================================
import os, glob, numpy as np, pandas as pd
from pathlib import Path

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # module root
os.chdir(ROOT)
FPS = 30.0
USE_VERIFIED_ONLY = False  # 검수(verified=YES) 후 True로

GTDIR = Path("03_repetition/evaluation/silver_GT_rep")
CONSENSUS_DIR = Path(
    "03_repetition/evaluation/rep_GT"
)  # 3인 자동 consensus(우선). 없으면 silver fallback.
XLSX = Path("03_repetition/evaluation/rep_peak_eval/Repetition frames_kk.xlsx")
# clip name -> spreadsheet sheet name
XMAP = {"RebarTying_00": "RebarTying_01", "RebarTying_01": "RebarTying_02"}
_x = pd.read_excel(XLSX, sheet_name=None, engine="openpyxl")


def sys_peaks(clip):
    sh = XMAP.get(clip, clip)
    if sh not in _x:
        return []
    d = _x[sh]
    col = next((c for c in d.columns if "frame" in str(c).lower()), None)
    return (
        sorted(int(v) for v in pd.to_numeric(d[col], errors="coerce").dropna())
        if col
        else []
    )


def to_mask(intervals, n, frames0):
    m = np.zeros(n, dtype=bool)
    for s, e in intervals:
        i0, i1 = max(0, s - frames0), min(n - 1, e - frames0)
        if i1 >= i0:
            m[i0 : i1 + 1] = True
    return m


rows = []
gt_src = "silver"
for f in sorted(glob.glob("00_joint_angle/*_angle.csv")):
    clip = os.path.basename(f).replace("_angle.csv", "")
    cf = CONSENSUS_DIR / f"{clip}_consensus.csv"  # 3인 자동 consensus 우선
    sf = GTDIR / f"{clip}_silverGT_reps.csv"
    gtf = cf if cf.exists() else sf
    if cf.exists():
        gt_src = "consensus(3인)"
    if not gtf.exists():
        continue
    gt = pd.read_csv(gtf)
    if gtf is sf and USE_VERIFIED_ONLY and "verified" in gt.columns:
        gt = gt[gt["verified"].astype(str).str.upper() == "YES"]
    gt_iv = list(zip(gt["start_frame"], gt["end_frame"]))
    sp = sys_peaks(clip)
    n_gt, n_sys = len(gt_iv), len(sp)
    # localization: 시스템 peak이 GT 구간 안 = TP(precision); GT 구간에 peak 1+ = hit(recall)
    tp = sum(any(s <= p <= e for s, e in gt_iv) for p in sp)
    prec = tp / n_sys if n_sys else 0.0
    hit = sum(any(s <= p <= e for p in sp) for s, e in gt_iv)
    rec = hit / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # frame-level IoU: rep-active mask (GT 구간 union vs 시스템 peak±tol union)
    idx = pd.read_csv(f, header=[0, 1], index_col=0).index.to_numpy()
    n, f0 = len(idx), int(idx[0])
    per_f = 6 * FPS
    tol = int(per_f * 0.25)
    gm = to_mask(gt_iv, n, f0)
    sm = to_mask([(p - tol, p + tol) for p in sp], n, f0)
    inter = (gm & sm).sum()
    union = (gm | sm).sum()
    iou = inter / union if union else 0.0
    rows.append(
        dict(
            clip=clip,
            n_GT=n_gt,
            n_sys=n_sys,
            count_err=n_sys - n_gt,
            prec=round(prec, 3),
            rec=round(rec, 3),
            F1=round(f1, 3),
            frame_IoU=round(iou, 3),
        )
    )

R = pd.DataFrame(rows)
print(f"=== repetition 검출 정확도 vs GT 구간 (GT={gt_src}) ===")
print(R.to_string(index=False))
if len(R):
    print(
        f"\n[overall] count MAE={R['count_err'].abs().mean():.2f} | "
        f"mean F1={R['F1'].mean():.3f} | mean frame_IoU={R['frame_IoU'].mean():.3f}"
    )
print(
    "[주의] silver GT(미검수) 데모값. look_at_frame 영상검수→verified=YES 후 USE_VERIFIED_ONLY=True로 최종."
)
