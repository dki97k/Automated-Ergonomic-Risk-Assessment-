# repetition GT 검수 안내 (표준 interval 방식)

**방식**: 사람이 **한 반복(1 cycle)의 시작/끝 프레임을 표시**해 그 구간을 rep으로 인정(=1). duration GT(다인 consensus)와 동일한 *독립 interval GT* 철학.

## 파일
- `<clip>_silverGT_reps.csv`: 자동 1차(silver) rep 구간. 컬럼 = `rep_id, start_frame, end_frame, start_sec, end_sec, dur_s, look_at_frame, source_joint, verified`.
  - silver = 가장 활발한 관절 각도의 주기적 극값(find_peaks, 시스템 SSM/RepNet과 **독립**)으로 연속 peak 사이를 1 cycle로 둔 초안.

## 검수 절차
1. `look_at_frame`(구간 중앙) 영상 확인 → 진짜 반복이면 `verified=YES`, 아니면 행 삭제/`NO`.
2. 필요시 `start_frame`/`end_frame` 보정(반복의 실제 시작/끝).
3. (권장) 2인 이상 독립 검수 후 consensus.
4. 검수 끝나면 `eval_vs_GT_rep.py`에서 `USE_VERIFIED_ONLY=True`로 최종 산출.

## 평가 (eval_vs_GT_rep.py)
- count_err(검출수−GT수), localization precision/recall/F1(시스템 peak이 GT 구간 안인지), frame-level rep-active IoU.
- 현재 silver(미검수) 기준 overall: count MAE≈5.6, F1≈0.77, IoU≈0.34 (데모; 검수 후 갱신).

## 구 방식과의 차이
구 `_rep_eval.py`(consistency) = *검출된 peak*를 사람이 Likert로 평가(precision-only·검출 종속). → 본 interval 방식 = **사람이 독립적으로 친 GT 구간**에 대해 평가(recall·count 포함). consistency는 보조 지표로 유지.
