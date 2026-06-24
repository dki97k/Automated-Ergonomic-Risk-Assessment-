# =====================================================
# 파일명: _param_sensitivity.py
# 역할: duration 검출 파라미터(SD창·t_min·SD임계계수)를 흔들어 출력(whole-body static ratio)이
#       얼마나 변하는지 정량 = *구조적 민감도*(GT 불필요, 안정성/robustness 증거).
#       정확도(F1/IoU)는 별도 manual-GT 필요(_param_sensitivity는 안정성만).
# 입력: 00_joint_angle/*_angle.csv, duration_analyzer.py(StaticPostureAnalyzer/AnalysisConfig)
# 출력: stdout — 설정별 clip별 static ratio(%) + 클립별 spread(min~max, CV).
# 의존: numpy/pandas + duration_analyzer. 임시 출력은 _tmp_sens/(자동).
# =====================================================
import importlib.util, os, shutil, tempfile, numpy as np, pandas as pd
from pathlib import Path
from dataclasses import replace

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.chdir(ROOT)


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


dur = load("02_duration/duration_analyzer.py", "dur")
base = dur.AnalysisConfig()

# 흔들 설정: baseline + 창/임계/t_min 섭동 (각각 ±)
settings = {
    "baseline": base,
    "sd_window=10": replace(base, sd_window=10),
    "sd_window=20": replace(base, sd_window=20),
    "t_min=90(3s)": replace(base, t_min=90),
    "t_min=150(5s)": replace(base, t_min=150),
    "SD×0.8(엄격)": replace(
        base, sd_a=base.sd_a * 0.8, sd_b=base.sd_b * 0.8, sd_leg=base.sd_leg * 0.8
    ),
    "SD×1.2(관대)": replace(
        base, sd_a=base.sd_a * 1.2, sd_b=base.sd_b * 1.2, sd_leg=base.sd_leg * 1.2
    ),
}
clips = sorted(Path("00_joint_angle").glob("*_angle.csv"))
res = {}  # setting -> {clip: static_ratio%}
for name, cfg in settings.items():
    an = dur.StaticPostureAnalyzer(cfg)
    tmp = Path(tempfile.mkdtemp(prefix="_sens_"))
    row = {}
    for f in clips:
        try:
            out = an.process_file(f, tmp)
            row[f.stem.replace("_angle", "")] = out.get(
                "Static posture ratio (%)", float("nan")
            )
        except Exception as e:
            row[f.stem.replace("_angle", "")] = float("nan")
    shutil.rmtree(tmp, ignore_errors=True)
    res[name] = row

df = pd.DataFrame(res).T  # rows=setting, cols=clip
print("=== duration 구조적 민감도: whole-body static posture ratio(%) ===")
print(df.round(1).to_string())
# clip별 spread (baseline 대비 변동)
print("\n=== clip별 안정성 (7개 설정 across) ===")
stab = pd.DataFrame(
    {
        "baseline": df.loc["baseline"],
        "min": df.min(),
        "max": df.max(),
        "range_pp": (df.max() - df.min()).round(1),
        "CV%": (df.std() / df.mean() * 100).round(1),
    }
).round(1)
print(stab.to_string())
print(
    f"\n[해석] 전 설정 across 평균 CV = {stab['CV%'].mean():.1f}%. range가 작을수록 파라미터에 견고."
)
print(
    "       정확도(F1/IoU vs GT)는 manual 정적구간 GT 필요 → 본 스크립트는 *안정성*만 정량."
)
