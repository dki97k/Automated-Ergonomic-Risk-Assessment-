# =====================================================
# 파일명: _activity_sensitivity.py
# 역할: REBA activity score 반영(full) vs 미반영(partial=현행)을 98프레임 전문가 GT에 대해 *경험적* 비교.
#       "activity를 넣는 게 맞나"를 inflation 논리가 아니라 GT 일치도로 판정(사용자 지적 반영).
# 입력: REBA_table.py(reba_frame), 00_joint_angle/*_angle.csv, 03_3_repetition/results/rep_risk.csv(freq),
#       03_4_integration/results/schema_summary.csv(dur_static_sec=held 판정)
# 출력: stdout — partial/full 각각 ICC/Acc/QWK/MAE/meanPred + activity 산정내역.
# 의존: numpy/pandas/scipy/sklearn. REBA activity: held(정적>1min)+repeated(>4/min)+unstable(미산출=0).
# =====================================================
import importlib.util, numpy as np, pandas as pd, os
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy import stats

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # module root
os.chdir(ROOT)


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


reba = load("01_pose/REBA_table.py", "reba")

# 98프레임 전문가 GT (frame, REBA Final) — _selffix_eval.py와 동일
GT = {
    "MansoryBrickLaying_00": ([212, 652, 1053, 1448], [4, 4, 3, 1]),
    "MansoryBrickLaying_01": (
        [226, 693, 1078, 1520, 1840, 2410, 2814, 3145, 3591],
        [1, 4, 1, 3, 1, 1, 6, 4, 2],
    ),
    "MansoryBrickLaying_02": (
        [
            318,
            584,
            947,
            1443,
            1709,
            2156,
            2581,
            3010,
            3509,
            3742,
            4236,
            4662,
            4907,
            5473,
            5749,
            6121,
            6781,
            7010,
        ],
        [1, 1, 1, 4, 1, 3, 2, 2, 3, 2, 4, 1, 1, 1, 1, 3, 1, 1],
    ),
    "MansoryCement_02": (
        [231, 676, 1052, 1413, 1915, 2247, 2852, 3145, 3653, 4007, 4476, 4798],
        [1, 1, 4, 1, 4, 3, 3, 1, 1, 6, 1, 1],
    ),
    "RebarPlacement_00": (
        [
            203,
            757,
            1147,
            1451,
            1966,
            2087,
            2529,
            2883,
            3383,
            3810,
            4175,
            4770,
            5014,
            5394,
            5888,
            6156,
            6786,
            6944,
            7659,
            7991,
            8165,
        ],
        [6, 6, 6, 1, 3, 1, 4, 3, 4, 2, 7, 4, 4, 4, 6, 4, 3, 4, 4, 3, 3],
    ),
    "RebarTying_00": ([251, 728, 1185, 1664], [6, 8, 6, 8]),
    "RebarTying_01": ([251, 728, 1185, 1664], [7, 7, 4, 10]),
    "WallPlacement_00": (
        [
            397,
            663,
            1172,
            1402,
            1658,
            2312,
            2736,
            3056,
            3548,
            3836,
            4252,
            4616,
            4851,
            5290,
            5662,
            6320,
            6589,
            7074,
            7646,
            7774,
            8245,
            8795,
            8977,
            9318,
            9811,
            10134,
        ],
        [10, 7, 1, 1, 1, 6, 6, 6, 6, 4, 4, 4, 6, 4, 4, 4, 3, 1, 1, 1, 4, 1, 7, 1, 1, 1],
    ),
}
lvl = lambda s: (
    0 if s == 1 else (1 if s <= 3 else (2 if s <= 7 else (3 if s <= 10 else 4)))
)


def icc(a, b):
    d = np.array([a, b], float).T
    n, k = d.shape
    gm = d.mean()
    SSR = k * np.sum((d.mean(1) - gm) ** 2)
    SST = np.sum((d - gm) ** 2)
    SSC = n * np.sum((d.mean(0) - gm) ** 2)
    BMS = SSR / (n - 1)
    EMS = (SST - SSR - SSC) / ((n - 1) * (k - 1))
    return (BMS - EMS) / (BMS + (k - 1) * EMS)


# --- clip별 REBA activity score 산정 (held + repeated; unstable=미산출) ---
# freq=반복 측정(repetition_measurement), static=duration 측정(integrated Whole Body).
# risk files are intentionally excluded; this evaluation uses measurement sources.
import glob as _glob

_rm = pd.read_csv("03_repetition/results/repetition_measurement.csv")
freq = dict(zip(_rm["clip"], _rm["repetition_rate_cycle_per_min"]))
held = {}
for _f in _glob.glob("02_duration/results/duration/*_integrated_analysis.csv"):
    _clip = os.path.basename(_f).replace("_angle_integrated_analysis.csv", "")
    _d = pd.read_csv(_f)
    _wb = _d[_d["Part"] == "Whole Body"] if "Part" in _d.columns else _d.iloc[0:0]
    held[_clip] = float(_wb["duration_sec"].sum()) if not _wb.empty else 0.0
activity = {}
for clip in GT:
    rep = (
        1 if freq.get(clip, 0) > 4 else 0
    )  # REBA: 동작 4회/min 초과 반복 → +1 (전 클립 해당)
    hld = 1 if held.get(clip, 0) > 60 else 0  # REBA: 1부위라도 정적 >1min 유지 → +1
    activity[clip] = rep + hld  # unstable(급변)은 pose로 미산출 → 0


def evaluate(use_activity):
    g, p = [], []
    for clip, (frs, gts) in GT.items():
        df = pd.read_csv(f"00_joint_angle/{clip}_angle.csv", header=[0, 1], index_col=0)
        act = activity[clip] if use_activity else 0
        for fr, gt in zip(frs, gts):
            if fr in df.index:
                pred = (
                    reba.reba_frame(df.loc[fr])["Final"] + act
                )  # full=partial+activity
                p.append(pred)
                g.append(gt)
    gl, pl = [lvl(s) for s in g], [lvl(s) for s in p]
    return dict(
        ICC=icc(g, p),
        Spear=stats.spearmanr(g, p)[0],
        Acc=accuracy_score(gl, pl),
        QWK=cohen_kappa_score(gl, pl, weights="quadratic"),
        MAE=float(np.mean(np.abs(np.array(g) - np.array(p)))),
        meanPred=float(np.mean(p)),
        meanGT=float(np.mean(g)),
    )


print(
    "=== REBA activity 민감도: partial(현행) vs full(activity 반영) — 98프레임 GT ==="
)
print("clip별 activity score (held>60s + repeated>4/min):")
for c, a in activity.items():
    print(
        f"  {c:22s} activity=+{a}  (freq={freq.get(c):.1f}/min, static={held.get(c):.0f}s)"
    )
pa, fu = evaluate(False), evaluate(True)
print(f"\n{'metric':10s} {'partial(현행)':>14s} {'full(+activity)':>16s}")
for k in ["ICC", "Spear", "Acc", "QWK", "MAE", "meanPred"]:
    print(f"{k:10s} {pa[k]:>14.3f} {fu[k]:>16.3f}")
print(f"{'meanGT':10s} {pa['meanGT']:>14.2f} {'(동일)':>16s}")
better = (
    "partial"
    if abs(pa["meanPred"] - pa["meanGT"]) < abs(fu["meanPred"] - fu["meanGT"])
    else "full"
)
print(
    f"\n[판정] GT mean={pa['meanGT']:.2f}. activity 반영 시 meanPred {pa['meanPred']:.2f}→{fu['meanPred']:.2f}, MAE {pa['MAE']:.3f}→{fu['MAE']:.3f}."
)
print(
    f"        → 전문가 GT에 더 가까운 쪽 = '{better}'. (inflation이 아니라 *GT 일치도*로 판정.)"
)
