# =====================================================
# 파일명: _reba_ablation.py  (model_v2 작업복사본 — 원본 무수정)
# 역할: 기존 angle CSV로 REBA를 재계산하되 두 수정(coupling, elbow)을 toggle해
#       system-vs-GT 평가(ICC/Spearman/ActLvl Acc/Quadratic Kappa) delta를 산출
# 입력: 00_joint_angle/*_angle.csv (기존 중간데이터, jsonl/torch 불필요)
# 출력: _ablation_result.csv + 콘솔 표
# 의존: REBA_table.py 로직 재구현(toggle), REBA_ICC_SC.py 내장 GT
# =====================================================
import numpy as np, pandas as pd, glob, os
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy import stats

# --- 내장 GT (REBA_ICC_SC.py 그대로) ---
gt_database = {
    "MansoryBrickLaying_00": {
        "Frames": [212, 652, 1053, 1448],
        "Final_GT": [4, 4, 3, 1],
    },
    "MansoryBrickLaying_01": {
        "Frames": [226, 693, 1078, 1520, 1840, 2410, 2814, 3145, 3591],
        "Final_GT": [1, 4, 1, 3, 1, 1, 6, 4, 2],
    },
    "MansoryBrickLaying_02": {
        "Frames": [
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
        "Final_GT": [1, 1, 1, 4, 1, 3, 2, 2, 3, 2, 4, 1, 1, 1, 1, 3, 1, 1],
    },
    "MansoryCement_02": {
        "Frames": [
            231,
            676,
            1052,
            1413,
            1915,
            2247,
            2852,
            3145,
            3653,
            4007,
            4476,
            4798,
        ],
        "Final_GT": [1, 1, 4, 1, 4, 3, 3, 1, 1, 6, 1, 1],
    },
    "RebarPlacement_00": {
        "Frames": [
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
        "Final_GT": [6, 6, 6, 1, 3, 1, 4, 3, 4, 2, 7, 4, 4, 4, 6, 4, 3, 4, 4, 3, 3],
    },
    "RebarTying_00": {"Frames": [251, 728, 1185, 1664], "Final_GT": [6, 8, 6, 8]},
    "RebarTying_01": {"Frames": [251, 728, 1185, 1664], "Final_GT": [7, 7, 4, 10]},
    "WallPlacement_00": {
        "Frames": [
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
        "Final_GT": [
            10,
            7,
            1,
            1,
            1,
            6,
            6,
            6,
            6,
            4,
            4,
            4,
            6,
            4,
            4,
            4,
            3,
            1,
            1,
            1,
            4,
            1,
            7,
            1,
            1,
            1,
        ],
    },
}

TABLE_A = np.array(
    [
        [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
        [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
        [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
    ]
)
TABLE_B = np.array(
    [
        [[1, 2, 2], [1, 2, 3], [3, 4, 5], [4, 5, 5], [6, 7, 8], [7, 8, 8]],
        [[1, 2, 3], [2, 3, 4], [4, 5, 5], [5, 6, 7], [7, 8, 8], [8, 9, 9]],
    ]
)
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
TH = 20.0


def s_neck(f, b, t):
    sf = 2 if f < -5 else (1 if f <= 20 else 2)
    return sf + (0 if abs(b) < TH else 1) + (0 if abs(t) < TH else 1)


def s_trunk(f, b, t):
    sf = 2 if f < -5 else (1 if f <= 5 else (2 if f <= 20 else (3 if f <= 60 else 4)))
    return sf + (0 if abs(b) < TH else 1) + (0 if abs(t) < TH else 1)


def s_upper(f, abd):
    sf = 2 if f < -20 else (1 if f <= 20 else (2 if f <= 45 else (3 if f <= 90 else 4)))
    return sf + (1 if abd > 45 else 0)


def s_lower(f, elbow_mode):
    # orig: interior-90 컨벤션, |f|<20 -> 1 (= interior[70,110])
    # reba: forearm flexion φ=90-f, 60<=φ<=100 -> 1 (= REBA 60-100, interior[80,120])
    if elbow_mode == "orig":
        return 1 if abs(f) < 20 else 2
    phi = 90.0 - f
    return 1 if 60 <= phi <= 100 else 2


def s_wrist(f, t):
    return (1 if -15 <= f <= 15 else 2) + (1 if not abs(t) < 45 else 0)


def s_leg(k, sr):
    sf = 2 if k >= 60 else (1 if k >= 30 else 0)
    return sf + (2 if (sr < 0.48 or sr > 0.52) else 1)


def coupling(mode):
    return (
        1 if mode == "fair_bug" else 0
    )  # fair_bug=현행(+1 default bug), exclude=0(partial REBA)


def reba_final(r, coup_mode, elbow_mode):
    neck = s_neck(
        r[("neck", "flexion")], r[("neck", "bending")], r[("neck", "twisting")]
    )
    trunk = s_trunk(
        r[("trunk", "flexion")], r[("trunk", "bending")], r[("trunk", "twisting")]
    )
    leg = s_leg(
        max(r[("knee", "left_flexion")], r[("knee", "right_flexion")]),
        r[("leg_support", "ratio")],
    )
    upper = max(
        s_upper(r[("upperarm", "left_flexion")], r[("upperarm", "left_abduction")]),
        s_upper(r[("upperarm", "right_flexion")], r[("upperarm", "right_abduction")]),
    )
    lower = max(
        s_lower(r[("lower arm", "left_flexion")], elbow_mode),
        s_lower(r[("lower arm", "right_flexion")], elbow_mode),
    )
    wrist = max(
        s_wrist(r[("wrist", "left_flexion")], r[("wrist", "left_twisting")]),
        s_wrist(r[("wrist", "right_flexion")], r[("wrist", "right_twisting")]),
    )
    A = TABLE_A[
        min(max(neck, 1), 3) - 1, min(max(trunk, 1), 5) - 1, min(max(leg, 1), 4) - 1
    ]
    B = TABLE_B[
        min(max(lower, 1), 2) - 1, min(max(upper, 1), 6) - 1, min(max(wrist, 1), 3) - 1
    ] + coupling(coup_mode)
    C = TABLE_C[min(A - 1, 11), min(B - 1, 11)]
    return int(C)


def action_level(s):
    return 0 if s == 1 else (1 if s <= 3 else (2 if s <= 7 else (3 if s <= 10 else 4)))


def icc_consistency(a, b):
    d = np.array([a, b], float).T
    n, k = d.shape
    gm = d.mean()
    SSR = k * np.sum((d.mean(1) - gm) ** 2)
    SST = np.sum((d - gm) ** 2)
    SSC = n * np.sum((d.mean(0) - gm) ** 2)
    BMS = SSR / (n - 1)
    EMS = (SST - SSR - SSC) / ((n - 1) * (k - 1))
    return (BMS - EMS) / (BMS + (k - 1) * EMS)


def evaluate(coup_mode, elbow_mode):
    gt_all, pred_all = [], []
    for f in glob.glob("00_joint_angle/*_angle.csv"):
        vid = next((k for k in gt_database if k in os.path.basename(f)), None)
        if not vid:
            continue
        df = pd.read_csv(f, header=[0, 1], index_col=0)
        finals = {
            fr: reba_final(df.loc[fr], coup_mode, elbow_mode)
            for fr in gt_database[vid]["Frames"]
            if fr in df.index
        }
        for fr, g in zip(gt_database[vid]["Frames"], gt_database[vid]["Final_GT"]):
            gt_all.append(g)
            pred_all.append(finals.get(fr, 0))
    gl = [action_level(s) for s in gt_all]
    pl = [action_level(s) for s in pred_all]
    return dict(
        n=len(gt_all),
        ICC=round(icc_consistency(gt_all, pred_all), 3),
        Spearman=round(stats.spearmanr(gt_all, pred_all)[0], 3),
        ActAcc=round(accuracy_score(gl, pl), 3),
        QWKappa=round(cohen_kappa_score(gl, pl, weights="quadratic"), 3),
        MAE=round(np.mean(np.abs(np.array(gt_all) - np.array(pred_all))), 3),
        meanPred=round(np.mean(pred_all), 2),
        meanGT=round(np.mean(gt_all), 2),
    )


rows = []
for tag, c, e in [
    ("baseline (현행: coupling+1, elbow orig)", "fair_bug", "orig"),
    ("coupling=0 (partial REBA)", "exclude", "orig"),
    ("elbow=REBA[60-100]", "fair_bug", "reba"),
    ("both fixes", "exclude", "reba"),
]:
    r = evaluate(c, e)
    r["config"] = tag
    rows.append(r)
res = pd.DataFrame(rows)[
    ["config", "n", "ICC", "Spearman", "ActAcc", "QWKappa", "MAE", "meanPred", "meanGT"]
]
res.to_csv("_ablation_result.csv", index=False, encoding="utf-8-sig")
print(res.to_string(index=False))
