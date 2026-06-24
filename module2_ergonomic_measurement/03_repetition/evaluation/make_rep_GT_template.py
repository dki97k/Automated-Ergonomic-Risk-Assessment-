# =====================================================
# 파일명: make_rep_GT_template.py
# 역할: repetition GT 양식 — 3인(R1/R2/R3)이 각자 독립적으로 '한 반복의 시작/끝 구간'을 적는 템플릿(duration과 통일).
#       rater마다 rep 개수가 달라도 무방. silver rep 초안을 시드로 제공.
# 입력: 03_repetition/evaluation/silver_GT_rep/<clip>_silverGT_reps.csv
# 출력: 03_repetition/evaluation/rep_GT/<clip>__R{1,2,3}.csv → make_rep_consensus.py가 합의.
# =====================================================
import os, glob, pandas as pd
from pathlib import Path

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # module root
os.chdir(ROOT)
SILVER = Path("03_repetition/evaluation/silver_GT_rep")
OUT = Path("03_repetition/evaluation/rep_GT")
OUT.mkdir(parents=True, exist_ok=True)

made = []
for f in sorted(glob.glob("00_joint_angle/*_angle.csv")):
    clip = os.path.basename(f).replace("_angle.csv", "")
    sf = SILVER / f"{clip}_silverGT_reps.csv"
    seed = (
        pd.read_csv(sf)
        if sf.exists()
        else pd.DataFrame(columns=["start_frame", "end_frame"])
    )
    tmpl = pd.DataFrame(
        {
            "start_frame": seed.get("start_frame", []),
            "end_frame": seed.get("end_frame", []),
            "look_at_frame": seed.get("look_at_frame", []),
            "keep": "YES",  # rater: 진짜 반복이면 YES, 아니면 행 삭제/NO. 누락 반복은 행 추가.
        }
    )
    for R in ["R1", "R2", "R3"]:
        tmpl.to_csv(OUT / f"{clip}__{R}.csv", index=False, encoding="utf-8-sig")
    made.append((clip, len(tmpl)))

print("=== repetition GT 템플릿(3인 독립) 생성 ===")
for c, n in made:
    print(f"  {c:22s} seed rep {n}개 → R1/R2/R3")
print(
    f"\n[저장] {OUT}/<clip>__R{{1,2,3}}.csv  → make_rep_consensus.py → eval_vs_GT_rep.py"
)
