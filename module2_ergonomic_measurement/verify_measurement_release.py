# =====================================================
# 파일명: verify_measurement_release.py
# 역할: reviewer 편의·재현성 검증. 측정 출력·eval을 내장 기대값과 1커맨드로 PASS/FAIL 대조.
# 입력: 없음(run_measurement 산출물 + self-contained eval 호출). 출력: stdout PASS/FAIL + caveat.
# 의존: 표준 + numpy/pandas. 자기완결(외부 참조 0).
# =====================================================
import subprocess, sys, re, json, glob, os
from pathlib import Path

HERE = Path(__file__).parent
os.chdir(HERE)
PY = sys.executable
EXPECT = {"ICC": 0.757, "Acc": 0.459, "QWK": 0.601, "MAE": 1.561}
TOL = {"ICC": 0.005, "Acc": 0.01, "QWK": 0.01, "MAE": 0.02}


def run(cmd):
    return subprocess.run([PY] + cmd, capture_output=True, text=True).stdout


checks = []

# 1) 측정 파이프라인 재생성 + 경계(risk 없음)
out = run(["run_measurement.py"])
checks.append(("측정계약 risk-free", "risk-해석 토큰: 없음" in out))
checks.append(
    ("risk 스크립트 0", "risk 스크립트(iso_duration/rep_risk/build_schema): 0개" in out)
)

# 2) REBA system-vs-GT 기대값
out = run(["01_pose/evaluation/_selffix_eval.py"])
m = {k: float(v) for k, v in re.findall(r"(ICC|Acc|QWK|MAE)=([\d.]+)", out)}
for k, v in EXPECT.items():
    got = m.get(k)
    checks.append((f"REBA {k}={v}", got is not None and abs(got - v) <= TOL[k]))

# 3) repetition consistency
out = run(["03_repetition/evaluation/_rep_eval.py"])
mc = re.search(r"mean_Likert=([\d.]+)", out)
checks.append(
    ("rep consistency≈4.17", mc is not None and abs(float(mc.group(1)) - 4.167) <= 0.01)
)

# 4) knee flip 0%
out = run(["01_pose/evaluation/knee_continuity_diag.py"])
checks.append(("knee flip=0%", "flip=0(0.0%)" in out))

# 5) 측정계약 8개 + 결정성
n_contract = len(glob.glob("llm_input/*.json"))
checks.append(("측정계약 8개", n_contract == 8))

passed = all(ok for _, ok in checks)
print("\n========== measurement release VERIFY ==========")
for name, ok in checks:
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}")
print(
    f"\n{'✅ PASS — measurement release reproduction and scope checks passed' if passed else '❌ FAIL — 위 항목 확인'}"
)
print("\n[caveat — reviewer 유의]")
print(
    "  · duration F1/IoU = silver GT(미검수) 데모값 → verified GT 투입 시 최종(eval_vs_GT.py)."
)
print(
    "  · repetition count/F1 = true-count GT 부재로 미산출(consistency까지). 날조 금지."
)
print(
    "  · REBA GT(98프레임): RebarTying _00/_01 프레임인덱스 동일(균등샘플 일치, 점수는 별개) — gt 교차확인 권장."
)
print(
    "  · 위험 해석(High/Low·등급)은 Module 2 범위 밖 = LLM(Module 3) structured GT로 평가."
)
sys.exit(0 if passed else 1)
