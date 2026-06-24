# self-fix 평가 하네스: (수정된) REBA_table로 GT 98프레임 재채점 → ICC/Acc/QWK/MAE/meanPred
import sys, glob, importlib.util, numpy as np, pandas as pd


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def cohen_kappa_score(y_true, y_pred, weights=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.arange(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()) + 1)
    n = len(labels)
    idx = {label: i for i, label in enumerate(labels)}
    observed = np.zeros((n, n), dtype=float)
    for a, b in zip(y_true, y_pred):
        observed[idx[a], idx[b]] += 1
    total = observed.sum()
    if total == 0:
        return float("nan")
    observed /= total
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col)
    if weights == "quadratic":
        denom = (n - 1) ** 2
        w = np.fromfunction(lambda i, j: ((i - j) ** 2) / denom, (n, n), dtype=float)
    elif weights is None:
        w = np.ones((n, n), dtype=float) - np.eye(n)
    else:
        raise ValueError(f"Unsupported weights={weights!r}")
    expected_loss = np.sum(w * expected)
    if expected_loss == 0:
        return 1.0
    return 1.0 - (np.sum(w * observed) / expected_loss)


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearmanr(a, b):
    ra = _rankdata(a)
    rb = _rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])

spec = importlib.util.spec_from_file_location(
    "reba", "01_pose/REBA_table.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
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
    # [⚠️ T3 주의] 두 RebarTying 클립의 GT 프레임 인덱스가 동일([251,728,1185,1664]).
    # 원본 gt.py Case 6/7(Rebar tying)에서 동일 시간대 균등 샘플링으로 추정(clip 길이 유사).
    # GT 점수는 다름([6,8,6,8] vs [7,7,4,10]) → 서로 다른 클립 관찰임. 검증 필요 시 gt.py 교차 확인.
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


print(
    "⚠️  GT주의: RebarTying_00/_01 프레임 인덱스 동일([251,728,1185,1664]) — 클립 길이 유사로 균등샘플링이 일치. GT점수는 서로 다름(별개 관찰). 의심시 gt.py Case6/7 교차확인."
)
g, p, parts = [], [], {}
for vid, (frs, gts) in GT.items():
    df = pd.read_csv(f"00_joint_angle/{vid}_angle.csv", header=[0, 1], index_col=0)
    for fr, gt in zip(frs, gts):
        if fr in df.index:
            d = m.reba_frame(df.loc[fr])
            p.append(d["Final"])
            g.append(gt)
            for kk in ["Neck", "Trunk", "Leg", "UpperArm", "LowerArm", "Wrist"]:
                parts.setdefault(kk, []).append(d[kk])
gl = [lvl(s) for s in g]
pl = [lvl(s) for s in p]
tag = sys.argv[1] if len(sys.argv) > 1 else ""
print(
    "%-22s ICC=%.3f Spear=%.3f Acc=%.3f QWK=%.3f MAE=%.3f meanPred=%.2f (GT=%.2f)"
    % (
        tag,
        icc(g, p),
        spearmanr(g, p),
        accuracy_score(gl, pl),
        cohen_kappa_score(gl, pl, weights="quadratic"),
        np.mean(np.abs(np.array(g) - np.array(p))),
        np.mean(p),
        np.mean(g),
    )
)
