# =====================================================
# 파일명: tune_params.py
# 역할: duration 분석기 파라미터를 (검수전) silver GT 기준으로 *5단위 sweep* 튜닝 — frame F1/IoU 최대화.
#       1차 = sd_window ∈ {5,10,15,20,25,30}(5단위). 보조 = t_min ∈ {90,120,150}.
# 입력: silver_GT/<clip>_silverGT_intervals.csv, 00_joint_angle/*_angle.csv, duration_analyzer.py
# 출력: stdout 표 + silver_GT/_tune_result.csv. ⚠️ silver(미검수) 기준 = 잠정. 검수 후 재실행.
# 의존: numpy/pandas + duration_analyzer.
# =====================================================
import importlib.util, os, glob, shutil, tempfile, numpy as np, pandas as pd
from pathlib import Path
from dataclasses import replace

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)
GTDIR = Path("02_duration/evaluation/silver_GT")
USE_VERIFIED_ONLY = False  # 검수(verified=YES) 후 True 권장


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


dur = load("02_duration/duration_analyzer.py", "dur")
base = dur.AnalysisConfig()
clips = [
    os.path.basename(f).replace("_angle.csv", "")
    for f in sorted(glob.glob("00_joint_angle/*_angle.csv"))
]


def mask_iv(df_iv, n):
    m = np.zeros(n, bool)
    for _, r in df_iv.iterrows():
        m[int(r["start_frame"]) : int(r["end_frame"]) + 1] = True
    return m


# GT 로드
gtm = {}
for clip in clips:
    fp = GTDIR / f"{clip}_silverGT_intervals.csv"
    iv = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    if USE_VERIFIED_ONLY and "verified" in iv.columns:
        iv = iv[iv["verified"].astype(str).str.upper() == "YES"]
    n = len(pd.read_csv(f"00_joint_angle/{clip}_angle.csv", header=[0, 1], index_col=0))
    gtm[clip] = (mask_iv(iv, n) if len(iv) else np.zeros(n, bool), n)


def evaluate(cfg):
    an = dur.StaticPostureAnalyzer(cfg)
    P = R = F = I = 0.0
    for clip in clips:
        gt, n = gtm[clip]
        tmp = Path(tempfile.mkdtemp(prefix="_tune_"))
        try:
            an.process_file(Path(f"00_joint_angle/{clip}_angle.csv"), tmp)
            iv = pd.read_csv(tmp / "duration" / f"{clip}_angle_integrated_analysis.csv")
            wb = (
                iv[iv["Part"] == "Whole Body"] if "Part" in iv.columns else iv.iloc[0:0]
            )
            det = mask_iv(wb, n) if len(wb) else np.zeros(n, bool)
        except Exception:
            det = np.zeros(n, bool)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        tp = int((gt & det).sum())
        fp = int((~gt & det).sum())
        fn = int((gt & ~det).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        P += pr
        R += rc
        F += f1
        I += iou
    k = len(clips)
    return P / k, R / k, F / k, I / k


print("=== duration 파라미터 튜닝 (silver GT 기준, 5단위) ===")
print(f"{'param':18s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'IoU':>6s}")
out = []
# 1차: sd_window 5단위
for w in [5, 10, 15, 20, 25, 30]:
    pr, rc, f1, iou = evaluate(replace(base, sd_window=w))
    print(f"sd_window={w:<7d} {pr:>6.3f} {rc:>6.3f} {f1:>6.3f} {iou:>6.3f}")
    out.append(
        dict(
            param=f"sd_window={w}",
            precision=round(pr, 3),
            recall=round(rc, 3),
            F1=round(f1, 3),
            IoU=round(iou, 3),
        )
    )
# 보조: t_min(초) sweep
for tm in [90, 120, 150]:
    pr, rc, f1, iou = evaluate(replace(base, t_min=tm))
    print(f"t_min={tm:<10d} {pr:>6.3f} {rc:>6.3f} {f1:>6.3f} {iou:>6.3f}")
    out.append(
        dict(
            param=f"t_min={tm}",
            precision=round(pr, 3),
            recall=round(rc, 3),
            F1=round(f1, 3),
            IoU=round(iou, 3),
        )
    )
res = pd.DataFrame(out)
res.to_csv(GTDIR / "_tune_result.csv", index=False, encoding="utf-8-sig")
best = res.loc[res["F1"].idxmax()]
print(
    f"\n[best by F1] {best['param']}  F1={best['F1']}  IoU={best['IoU']} (현행 baseline sd_window={base.sd_window})"
)
print("[주의] silver(미검수) 기준 잠정값. 검수 후 USE_VERIFIED_ONLY=True 재실행→확정.")
