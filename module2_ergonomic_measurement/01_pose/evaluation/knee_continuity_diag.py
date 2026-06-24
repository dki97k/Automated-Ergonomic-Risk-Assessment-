# =====================================================
# 파일명: knee_continuity_diag.py
# 역할: knee 고값(>120°)이 (a)실제 깊은 스쿼트/kneeling인지 (b)keypoint flip(~180-θ) 아티팩트인지
#       *시간연속성*으로 정량 분류 — 영상 없이 angle 시계열만으로 1차 판정.
# 입력: 00_joint_angle/*_angle.csv (knee left/right flexion; 0=직선 컨벤션, 값↑=굴곡↑)
# 출력: stdout (클립별 flip% vs real%) + 애매프레임 shortlist CSV(_knee_ambiguous_frames.csv)
# 의존: numpy, pandas. 근거: flip은 인접프레임에서 θ↔180-θ로 점프(생리적 불가능 각속도) →
#       |Δknee/frame|이 사람 무릎 최대각속도(~400°/s=13°/frame@30fps)를 크게 초과(>40°/frame).
# =====================================================
import glob, os, numpy as np, pandas as pd

FPS = 30.0
HIGH = 120.0  # REBA leg high-flex 영역(>=60°는 이미 score 2). 'flip 의심' 관찰대역
IMPOSSIBLE = (
    40.0  # °/frame (=1200°/s) — 생리적 불가능 점프 = flip 마커(인간 무릎 ~400°/s)
)
FAST = 13.0  # °/frame (=400°/s) — 빠른 실제동작 상한(이 사이는 ambiguous)
SUSTAINED = 15  # frames(0.5s) 이상 지속된 high run = 실제자세 가능성↑

# self-fix GT의 high-knee 영향 프레임(클립별 GT frame index) — 26/98 영향 검증용
GT = {
    "MansoryBrickLaying_00": [212, 652, 1053, 1448],
    "MansoryBrickLaying_01": [226, 693, 1078, 1520, 1840, 2410, 2814, 3145, 3591],
    "MansoryBrickLaying_02": [
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
    "MansoryCement_02": [
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
    "RebarPlacement_00": [
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
    "RebarTying_00": [251, 728, 1185, 1664],
    "RebarTying_01": [251, 728, 1185, 1664],
    "WallPlacement_00": [
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
}


def runs_ge(mask):
    """연속 True run의 (start,end) 목록 — high-knee 지속구간 길이 판정용."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


amb_rows, summ = [], []
for f in sorted(glob.glob("00_joint_angle/*_angle.csv")):
    clip = os.path.basename(f).replace("_angle.csv", "")
    df = pd.read_csv(f, header=[0, 1], index_col=0)
    kl, kr = (
        df[("knee", "left_flexion")].to_numpy(),
        df[("knee", "right_flexion")].to_numpy(),
    )
    knee = np.maximum(kl, kr)  # REBA leg와 동일(좌우 worst)
    dabs = np.abs(np.diff(knee, prepend=knee[0]))  # 프레임간 변화량
    high = knee > HIGH
    runs = [r for r in runs_ge(high)]
    n_high = int(high.sum())
    # 분류: high 프레임이 속한 run의 진입점 점프로 flip/real/ambiguous
    flip = real = amb = 0
    frame_class = {}
    for s, e in runs:
        run_len = e - s + 1
        entry_jump = dabs[s]  # run 진입 프레임의 점프량
        if entry_jump >= IMPOSSIBLE:  # 불가능 점프 → flip
            cls = "flip"
        elif run_len >= SUSTAINED and entry_jump < FAST:  # 매끄럽게 지속 → 실제
            cls = "real"
        else:
            cls = "ambiguous"
        for i in range(s, e + 1):
            frame_class[int(df.index[i])] = (
                cls,
                float(knee[i]),
                float(entry_jump),
                int(run_len),
            )
        if cls == "flip":
            flip += run_len
        elif cls == "real":
            real += run_len
        else:
            amb += run_len
    # GT 프레임 중 high-knee 영향분
    gt_high = [
        fr
        for fr in GT.get(clip, [])
        if fr in df.index and knee[df.index.get_loc(fr)] > HIGH
    ]
    summ.append(
        dict(
            clip=clip,
            n_frames=len(df),
            n_high=n_high,
            pct_high=round(100 * n_high / len(df), 1),
            flip_fr=flip,
            real_fr=real,
            amb_fr=amb,
            gt_high=len(gt_high),
        )
    )
    # 애매프레임 shortlist(클립당 최대 3개, run 중앙)
    for s, e in runs:
        cls, _, ej, rl = frame_class[int(df.index[s])]
        if cls == "ambiguous":
            mid = int(df.index[(s + e) // 2])
            amb_rows.append(
                dict(
                    clip=clip,
                    frame=mid,
                    knee_deg=round(float(knee[(s + e) // 2]), 1),
                    entry_jump=round(float(ej), 1),
                    run_len_frames=rl,
                    run_sec=round(rl / FPS, 2),
                )
            )

S = pd.DataFrame(summ)
print("=== knee 시간연속성 진단 (영상 없이 angle 시계열) ===")
print(S.to_string(index=False))
tot_flip, tot_real, tot_amb = S.flip_fr.sum(), S.real_fr.sum(), S.amb_fr.sum()
tot = tot_flip + tot_real + tot_amb
print(
    f"\n[전체 high-knee 프레임 {tot}개] flip={tot_flip}({100*tot_flip/tot:.1f}%)  "
    f"real={tot_real}({100*tot_real/tot:.1f}%)  ambiguous={tot_amb}({100*tot_amb/tot:.1f}%)"
)
print(f"[GT 98프레임 중 high-knee 영향] {S.gt_high.sum()}개")
A = pd.DataFrame(amb_rows)
A.to_csv(
    "01_pose/evaluation/_knee_ambiguous_frames.csv",
    index=False,
    encoding="utf-8-sig",
)
print(
    f"\n[애매프레임 shortlist] {len(A)}개 → _knee_ambiguous_frames.csv (영상 눈확인 대상)"
)
if len(A):
    print(A.to_string(index=False))
