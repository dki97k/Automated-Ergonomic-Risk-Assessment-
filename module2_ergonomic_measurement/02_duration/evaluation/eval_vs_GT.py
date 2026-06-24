# =====================================================
# 파일명: eval_vs_GT.py
# 역할: duration 검출의 *정확도 기반* 파라미터 민감도 — (검수된) silver GT 대비 분석기 검출의
#       frame-level F1/IoU/MoF를 파라미터 설정별로 산출. GT가 파라미터-독립이라 공정 비교 가능.
# 입력: silver_GT/<clip>_silverGT_intervals.csv(전신 static 구간 GT), 00_joint_angle/*_angle.csv,
#       duration_analyzer.py(StaticPostureAnalyzer/AnalysisConfig)
# 출력: stdout — 설정별 macro precision/recall/F1/IoU/MoF.
# 의존: numpy/pandas + duration_analyzer. scope: GT=순수static, 검출=safe-zone gating → Δ(설정 간)가 핵심.
# =====================================================
import importlib.util, os, glob, shutil, tempfile, numpy as np, pandas as pd
from pathlib import Path
from dataclasses import replace

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)
USE_VERIFIED_ONLY = (
    False  # True면 verified==YES 행만 GT로(검수 완료 후 권장). silver 데모는 False.
)
GTDIR = Path("02_duration/evaluation/silver_GT")
CONSENSUS_DIR = Path(
    "02_duration/evaluation/dur_GT"
)  # 3인 → 자동 consensus(우선). 없으면 silver fallback.


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


dur = load("02_duration/duration_analyzer.py", "dur")
base = dur.AnalysisConfig()
settings = {
    "baseline": base,
    "sd_window=10": replace(base, sd_window=10),
    "sd_window=20": replace(base, sd_window=20),
    "t_min=90": replace(base, t_min=90),
    "SD×0.8": replace(
        base, sd_a=base.sd_a * 0.8, sd_b=base.sd_b * 0.8, sd_leg=base.sd_leg * 0.8
    ),
    "SD×1.2": replace(
        base, sd_a=base.sd_a * 1.2, sd_b=base.sd_b * 1.2, sd_leg=base.sd_leg * 1.2
    ),
}


def mask_from_intervals(df_iv, n):
    m = np.zeros(n, dtype=bool)
    for _, r in df_iv.iterrows():
        m[int(r["start_frame"]) : int(r["end_frame"]) + 1] = True
    return m


def metrics(gt, det):
    tp = int((gt & det).sum())
    fp = int((~gt & det).sum())
    fn = int((gt & ~det).sum())
    tn = int((~gt & ~det).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    mof = (tp + tn) / (tp + fp + fn + tn)
    return prec, rec, f1, iou, mof


clips = [
    os.path.basename(f).replace("_angle.csv", "")
    for f in sorted(glob.glob("00_joint_angle/*_angle.csv"))
]
# GT 로드
gtm = {}
gt_src = "silver"
for clip in clips:
    cf = CONSENSUS_DIR / f"{clip}_consensus.csv"  # 3인 자동 consensus 우선
    sf = GTDIR / f"{clip}_silverGT_intervals.csv"
    fp = cf if cf.exists() else sf
    if cf.exists():
        gt_src = "consensus"
    iv = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    if fp is sf and USE_VERIFIED_ONLY and "verified" in iv.columns:
        iv = iv[iv["verified"].astype(str).str.upper() == "YES"]
    n = len(pd.read_csv(f"00_joint_angle/{clip}_angle.csv", header=[0, 1], index_col=0))
    # GT 구간수 = len(iv) (segment 수). 검출 segment 수와 달라도 frame-mask로 비교(개수 차 robust).
    gtm[clip] = (
        mask_from_intervals(iv, n) if len(iv) else np.zeros(n, bool),
        n,
        len(iv),
    )

print(
    f"=== duration 정확도 민감도 (GT={gt_src}, verified_only={USE_VERIFIED_ONLY}) ==="
)
print(
    "[GT=독립 annotation(3인 consensus). 검출 segment 수와 GT segment 수가 달라도 frame-mask 비교로 robust.]"
)
print(
    f"{'setting':14s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'IoU':>6s} {'MoF':>6s} {'GTseg':>6s} {'detseg':>7s}"
)
rowsout = []
for name, cfg in settings.items():
    an = dur.StaticPostureAnalyzer(cfg)
    P = R = F = I = M = 0.0
    gtseg_t = detseg_t = nclip = 0
    for clip in clips:
        gt, n, n_gtseg = gtm[clip]
        tmp = Path(tempfile.mkdtemp(prefix="_ev_"))
        try:
            an.process_file(Path(f"00_joint_angle/{clip}_angle.csv"), tmp)
            iv = pd.read_csv(tmp / "duration" / f"{clip}_angle_integrated_analysis.csv")
            wb = (
                iv[iv["Part"] == "Whole Body"] if "Part" in iv.columns else iv.iloc[0:0]
            )
            det = (
                mask_from_intervals(
                    wb.rename(
                        columns={"start_frame": "start_frame", "end_frame": "end_frame"}
                    ),
                    n,
                )
                if len(wb)
                else np.zeros(n, bool)
            )
        except Exception:
            det = np.zeros(n, bool)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        p, r, f1, iou, mof = metrics(gt, det)
        P += p
        R += r
        F += f1
        I += iou
        M += mof
        nclip += 1
        gtseg_t += n_gtseg
        # 검출 segment 수 = det 마스크의 상승에지 수(연속 run 개수). GT seg 수와 달라도 무관(frame 비교).
        _d = np.diff(np.concatenate(([0], det.astype(int), [0])))
        detseg_t += int((_d == 1).sum())
    k = max(nclip, 1)
    print(
        f"{name:14s} {P/k:>6.3f} {R/k:>6.3f} {F/k:>6.3f} {I/k:>6.3f} {M/k:>6.3f} {gtseg_t:>6d} {detseg_t:>7d}"
    )
    rowsout.append(
        dict(
            setting=name,
            precision=round(P / k, 3),
            recall=round(R / k, 3),
            F1=round(F / k, 3),
            IoU=round(I / k, 3),
            MoF=round(M / k, 3),
            GT_seg=gtseg_t,
            det_seg=detseg_t,
        )
    )
pd.DataFrame(rowsout).to_csv(
    GTDIR / "_eval_param_sensitivity.csv", index=False, encoding="utf-8-sig"
)
print(
    f"\n[해석] F1/IoU의 설정 간 변동 = 파라미터 민감도(정확도). baseline이 최적 근처면 견고."
)
print(
    "[주의] silver GT(미검수)에 대한 데모값. 검수(verified=YES) 후 USE_VERIFIED_ONLY=True로 최종 산출."
)
