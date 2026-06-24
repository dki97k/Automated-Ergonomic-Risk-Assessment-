# =====================================================
# 파일명: make_dur_GT_template.py
# 역할: duration GT 양식 생성 — 3인(R1/R2/R3)이 *각자 독립적으로* 전신 static 구간(start/end)을 적는 템플릿.
#       rater마다 구간 개수가 달라도 무방(독립 GT). silver 초안을 시드로 깔아 검수 시작점 제공.
# 입력: 02_duration/evaluation/silver_GT/<clip>_silverGT_intervals.csv (시드)
# 출력: 02_duration/evaluation/dur_GT/<clip>__R{1,2,3}.csv (각 rater 편집) — make_dur_consensus.py가 합의 생성.
# 의존: pandas. 검수: dur_GT/README.md.
# =====================================================
import os, glob, pandas as pd
from pathlib import Path

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # module root
os.chdir(ROOT)
FPS = 30.0
SILVER = Path("02_duration/evaluation/silver_GT")
OUT = Path("02_duration/evaluation/dur_GT")
OUT.mkdir(parents=True, exist_ok=True)

made = []
for f in sorted(glob.glob("00_joint_angle/*_angle.csv")):
    clip = os.path.basename(f).replace("_angle.csv", "")
    sf = SILVER / f"{clip}_silverGT_intervals.csv"
    seed = (
        pd.read_csv(sf)
        if sf.exists()
        else pd.DataFrame(columns=["start_frame", "end_frame"])
    )
    # 양식: rater가 채울 전신 static 구간. 모델 검출구간에 종속되지 않음 = 독립.
    tmpl = pd.DataFrame(
        {
            "start_frame": seed.get("start_frame", []),
            "end_frame": seed.get("end_frame", []),
            "look_at_frame": seed.get("look_at_frame", []),
            "part": "Whole Body",  # (확장) 추후 joint별로 분리 가능
            "keep": "YES",  # rater: 진짜 static이면 YES, 아니면 행 삭제/NO. 누락구간은 행 추가.
        }
    )
    for R in ["R1", "R2", "R3"]:
        tmpl.to_csv(OUT / f"{clip}__{R}.csv", index=False, encoding="utf-8-sig")
    made.append((clip, len(tmpl)))

print("=== duration GT 템플릿(3인 독립) 생성 ===")
for c, n in made:
    print(f"  {c:22s} seed 구간 {n}개 → R1/R2/R3 템플릿")
print(f"\n[저장] {OUT}/<clip>__R{{1,2,3}}.csv")
print(
    "[작성법] 각 rater가 영상보고 독립적으로: look_at_frame 확인→진짜 static만 keep=YES, start/end 보정, 누락구간 행 추가."
)
print(
    "[다음] make_dur_consensus.py → frame-level 다수결 자동 consensus(개수 달라도 OK) → eval_vs_GT.py"
)
