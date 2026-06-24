# Measurement Performance Summary

This note summarizes the released Module 2 measurement checks. The comparison is
limited to measurement accuracy for REBA scoring, static-duration detection, and
repetition detection. Final risk interpretation is evaluated in Module 3.

## 1. POSE / REBA — system vs 전문가 GT(98 프레임)
| config | ICC | ActLvl Acc | QW-Kappa | MAE | meanPred(GT=3.35) |
|---|---|---|---|---|---|
| **Earlier baseline**: coupling+1 issue, original elbow rule | 0.752 | 0.367 | 0.522 | 1.888 | 5.05 |
| **Released measurement module**: coupling=0 partial score, corrected elbow range, neck/trunk adjustment | **0.757** | **0.459** | **0.601** | **1.561** | 4.67 |
| Δ | +0.005 | **+0.092 (+25%)** | **+0.079 (+15%)** | **−0.327 (−17%)** | 과대예측 완화 |

- **해석**: 버그수정(coupling 상수+1 제거 = partial REBA, elbow REBA 정범위)으로 **action-level 정확도·QWK 대폭↑, MAE↓**. ICC는 상수편향에 둔감해 ~보합(원래 일치도 자체는 높았음).
- ablation 근거: `01_pose/evaluation/_reba_ablation.py` → coupling=0 단독이 Acc/MAE 개선의 주동인; elbow는 ICC 소폭↑.

## 2. DURATION — 정적구간 검출 정확도
- **원본**: 정적검출에 대한 **독립 평가 부재**(검출 segment를 그대로 보고만).
- **Released measurement module**: **frame-level F1 / IoU / MoF vs frame-level GT**(시간 segmentation 표준 평가) 신설 — `eval_vs_GT.py`(+ `make_silver_GT.py` 자동 silver GT, `tune_params.py`·`_param_sensitivity.py` 파라미터 민감도).
- 현재 = **silver GT(미검수) 데모값**(예 baseline 설정 F1≈0.75). **검수(verified) GT 투입 시 최종 산출**. → *원본 대비 "평가 가능성" 자체가 신규 기여*.

## 3. REPETITION — 검출 정확도 (interval-GT 방식으로 전환)
- **원본**: 평가 부재(count만).
- **방식 전환**: 구 consistency(검출 peak를 사람이 Likert로 평가 = precision-only·검출 종속) → **사람이 한 반복의 시작/끝 구간을 표시한 독립 GT**에 대해 평가(duration GT와 동일 철학; `make_silver_GT_rep.py`→`eval_vs_GT_rep.py`).
- **Released measurement metrics**(silver GT, 미검수 데모): overall **count MAE≈5.6 / localization F1≈0.77 / frame IoU≈0.34**. → 영상 검수(verified=YES) 후 최종.
- 보조(유지): consistency pooled **4.167/5**(132 peaks, `_rep_eval.py`) = 검출 peak 타당성(precision-side).

## 4. 종합
- **측정 정확도(원본 대비 정량 개선 확인)** = REBA(Acc +25%·QWK +15%·MAE −17%).
- **평가 가능성(신규)** = duration frame-level F1/IoU·repetition consistency = 원본엔 없던 *측정 검증 체계*.
- **위험 해석 성능**(risk_summary/key_factor 정확도)은 Module 2 범위 밖 → Module 3(LLM) structured GT로 별도 평가.
