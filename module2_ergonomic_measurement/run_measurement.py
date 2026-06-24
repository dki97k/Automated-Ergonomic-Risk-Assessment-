# =====================================================
# 파일명: run_measurement.py
# 역할: Module 2 = 측정·기호 feature까지만. 위험 해석(risk interpretation)은 LLM(Module 3).
#       per-modality 측정 results + 측정계약(LLM 입력) 생성. iso_duration/rep_risk/build_schema(risk) 없음.
# 입력: 00_joint_angle/*_angle.csv (Tier-1).
# 출력: 01_pose/results/(REBA SCORE+측정요약), 02_duration/results/(정적구간), 03_repetition/results/(반복측정),
#       llm_input/<clip>.json (측정계약 = numerical_only 측정필드; risk_level/co_level 없음).
# 의존: numpy,pandas,scipy,sklearn. 경계근거: docs/../MEASUREMENT_SCOPE.md.
# =====================================================
import os, sys, glob, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).parent
os.chdir(HERE)


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def step(m):
    print(f"\n=== {m} ===", flush=True)


reba = load("01_pose/REBA_table.py", "reba")
dur = load("02_duration/duration_analyzer.py", "dur")
sys.path.insert(0, str(HERE / "llm_input"))
from _contract_common import build_input_summary  # m3 측정-집계(측정전용)

ANG = sorted(glob.glob("00_joint_angle/*_angle.csv"))
clips = [os.path.basename(f).replace("_angle.csv", "") for f in ANG]

# --- 1) POSE 측정: REBA score (표준 기호 feature) ---
step("1) POSE — REBA score (측정)")
os.makedirs("01_pose/results", exist_ok=True)
pose_summ = []
for f in ANG:
    s = reba.process_file(f, "01_pose/results")  # SCORE.csv (측정). 위험라벨은 미사용.
    if s:
        # 측정 요약만(mean/p90/peak); 'Overall posture risk level' 라벨은 제외(→LLM)
        pose_summ.append({k: s[k] for k in s if k != "Overall posture risk level"})
pd.DataFrame(pose_summ).to_csv(
    "01_pose/results/pose_measurement_summary.csv", index=False, encoding="utf-8-sig"
)
print(f"  -> {len(pose_summ)} SCORE + pose_measurement_summary.csv (risk 라벨 제외)")

# --- 2) DURATION 측정: 정적 hold 검출 (위험등급 없음) ---
step("2) DURATION — 정적구간 검출 (측정)")
an = dur.StaticPostureAnalyzer(dur.AnalysisConfig())
outd = Path("02_duration/results")
outd.mkdir(exist_ok=True)
for f in ANG:
    an.process_file(Path(f), outd)
print("  -> segments/ + duration/ (정적구간 측정; ISO graded risk 없음)")

# --- 3) REPETITION 측정: peak count/period/rpm (위험등급 없음) ---
step("3) REPETITION — peak count/period (측정)")
_fr = pd.read_csv("03_repetition/rep_period_frozen.csv")
PER = dict(zip(_fr["clip"], _fr["cycle_sec"]))
STD = dict(
    zip(_fr["clip"], _fr["std_sec"])
)  # std도 frozen에 baking됨(self-contained)
_npk = pd.read_csv("03_repetition/results/_rep_consistency_eval.csv")
NPK = dict(zip(_npk["clip"], _npk["n_peaks"]))


def std_of(c):
    v = STD.get(c)
    return float(v) if v is not None and not pd.isna(v) else None


rep_rows = []
for c in clips:
    per = PER.get(c, 0.0)
    rep_rows.append(
        dict(
            clip=c,
            total_repetitions=int(NPK.get(c, 0)),
            mean_period_sec=round(per, 3),
            repetition_rate_cycle_per_min=round(60.0 / per, 3) if per else 0.0,
            std_period_sec=std_of(c) or 0.0,
        )
    )
pd.DataFrame(rep_rows).to_csv(
    "03_repetition/results/repetition_measurement.csv",
    index=False,
    encoding="utf-8-sig",
)
print(
    f"  -> repetition_measurement.csv (count/period/rpm/std; Silverstein/Rodgers risk 없음)"
)
REP = {r["clip"]: r for r in rep_rows}

# --- 4) 측정계약 (LLM 입력) = numerical_only 측정필드. risk_level/co_level/ISO/Silverstein 없음 ---
step("4) 측정계약 → llm_input/<clip>.json")
REBA_MAP = {
    "Neck": "neck",
    "Trunk": "trunk",
    "Leg": "leg",
    "UpperArm": "upper_arm",
    "LowerArm": "lower_arm",
    "Wrist": "wrist",
    "Final": "final",
}
os.makedirs("llm_input", exist_ok=True)
n_contract = 0
for c in clips:
    adf = pd.read_csv(f"00_joint_angle/{c}_angle.csv", header=[0, 1], index_col=0)
    angles = {}
    for fr, row in adf.iterrows():
        d = {}
        for (t, sub), v in row.items():
            if not pd.isna(v):
                d.setdefault(t, {})[sub] = float(v)
        angles[str(int(fr))] = d
    sdf = pd.read_csv(f"01_pose/results/{c}_angle_SCORE.csv")
    rebad = {
        str(int(r["frame"])): {REBA_MAP[k]: float(r[k]) for k in REBA_MAP if k in r}
        for _, r in sdf.iterrows()
    }
    idf_fp = f"02_duration/results/duration/{c}_angle_integrated_analysis.csv"
    wb = []
    if os.path.exists(idf_fp):
        idf = pd.read_csv(idf_fp)
        wb = [
            {
                "start_frame": int(r["start_frame"]),
                "end_frame": int(r["end_frame"]),
                "duration_sec": float(r["duration_sec"]),
            }
            for _, r in idf[idf["Part"] == "Whole Body"].iterrows()
        ]
    rr = REP[c]
    processed = {
        "meta": {"video_id": c, "fps": 30, "total_frames": len(rebad)},
        "angles": angles,
        "reba": rebad,
        "duration": {"whole_body": wb},
        "repetition": {
            "total_repetitions": rr["total_repetitions"],
            "repetition_rate_cycle_per_min": rr["repetition_rate_cycle_per_min"],
            "mean_period_sec": rr["mean_period_sec"],
            "std_period_sec": rr["std_period_sec"],
        },
    }
    contract = build_input_summary(
        processed
    )  # 측정전용 집계(REBA통계·각도·정적·반복). risk 필드 없음.
    json.dump(
        contract,
        open(f"llm_input/{c}.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    n_contract += 1
print(f"  -> {n_contract} 측정계약 JSON (LLM이 risk_summary+key_factor 해석)")

# --- 경계 검증 ---
step("경계 검증")
import glob as _g

sample = json.load(open(_g.glob("llm_input/*.json")[0], encoding="utf-8"))
blob = json.dumps(sample)
risk_tokens = [
    t
    for t in [
        "risk_level",
        "co_level",
        "duration_risk_iso",
        "repetition_risk_silverstein",
        "freq_risk_rodgers",
        "Negligible",
        "VeryHigh",
    ]
    if t in blob
]
print(f"  측정계약 내 risk-해석 토큰: {risk_tokens if risk_tokens else '없음 ✅'}")
print(
    f"  risk 스크립트(iso_duration/rep_risk/build_schema): {len(_g.glob('**/iso_duration.py', recursive=True)+_g.glob('**/rep_risk.py', recursive=True)+_g.glob('**/build_schema.py', recursive=True))}개(0이어야 ✅)"
)
print(
    "\n[measurement release] 측정 results(pose/dur/rep) + 측정계약 생성 완료. risk 해석 = LLM(Module 3)."
)
