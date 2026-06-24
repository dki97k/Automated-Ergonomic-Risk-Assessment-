# duration GT (3인 독립 → 자동 consensus) 안내

**철학**: **모델 검출 구간에 종속되지 않는 독립 GT**. 3인이 *각자* 전신 static 구간(start/end)을 표시 → frame-level 다수결로 자동 consensus → 그 consensus로 평가.

## 파일
- `<clip>__R1.csv`, `__R2.csv`, `__R3.csv`: 3인 각자 작성(독립). 컬럼 = `start_frame, end_frame, look_at_frame, part, keep`. **rater마다 구간 개수가 달라도 됨.**
  - 시드 = silver 초안(편집 시작점). `look_at_frame` 영상 확인 → 진짜 static이면 `keep=YES`, 아니면 행 삭제, 누락 구간은 행 추가.
- `<clip>_consensus.csv`: `make_dur_consensus.py`가 자동 생성(다수결 ≥2/3). **수정 금지(자동산출)**.
- `_consensus_reliability.csv`: rater별 구간수(rater_nseg)·pairwise frame IoU·Fleiss κ.

## 워크플로
1. (편집) 3인이 `__R{1,2,3}.csv` 독립 작성.
2. `python 02_duration/evaluation/make_dur_consensus.py` → consensus + 신뢰도 자동.
3. `python 02_duration/evaluation/eval_vs_GT.py` → 검출 vs **consensus**, frame-level F1/IoU/MoF + GTseg/detseg(개수 차 표시).

## 핵심(평가모델 변경점)
- GT가 **독립**(모델 검출 무관) → 검출 segment 수와 GT segment 수가 **다를 수 있음** → eval은 **frame-mask 비교**라 개수 차에 robust(상승에지로 segment 수만 별도 보고).
- consensus는 **frame 다수결** → rater 구간수가 달라도 합의 생성.
- repetition도 동일 철학: `03_repetition/evaluation/{make_silver_GT_rep,eval_vs_GT_rep}.py` (사람이 친 rep 구간 GT).
