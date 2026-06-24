<!--
파일명: silver_GT/README.md
역할: duration 평가용 silver(자동 1차) GT의 정의·검수 절차·다음단계 안내.
-->

# duration silver GT — 검수 안내

## 이게 뭔가
duration 검출의 **파라미터 민감도(정확도)** 를 재려면 *파라미터-독립* GT가 필요하다(분석기 검출을 GT로 쓰면 순환). 본 silver GT는 그 1차 초안 — **사람이 검수·수정할 출발점**이다.

- **독립성**: 분석기(SD-속도 적응임계)와 *다른* 방법으로 생성 = **range 기반**(각도가 4s 창에서 **δ=15° 이내 유지=static**). 그래서 순환이 아님.
- **2단계 구조**(사용자 정의): (1) **전신 static 구간** = 몸통 또는 다리(postural core)가 유지(`WB_MODE='core'`) → (2) 각 구간에서 **부위별 static**(Neck/Trunk/Legs/ArmsL/ArmsR).
- **현 초안 결과**: RebarTying≈88%, Wall≈58%, Masonry/RebarPlacement 10–44%(분석기와 근사하나 독립 산출).

## 파일 형식
`<clip>_silverGT_intervals.csv` — 행 1개 = 전신 static 구간 1개:
`interval_id, start_frame, end_frame, duration_s, Neck_static, Trunk_static, Legs_static, ArmsL_static, ArmsR_static, verified`
- `*_static` = 1/0 (그 구간에서 해당 부위가 static인가).
- `verified` = NO(초안). 검수 후 YES.

## 검수 방법 (사람)
1. 클립 영상(`model/00_video/raw/Images/<clip>/` 또는 `SKELTON OVERLAY/<clip>/vis/`)과 CSV를 나란히 본다.
2. 각 구간의 start/end가 실제 "자세 유지" 구간과 맞는지 확인 → **start_frame/end_frame 수정**.
3. 부위 static 플래그(1/0)가 맞는지 확인 → 수정(예: 팔이 실제로 멈췄으면 ArmsL_static=1).
4. 누락 구간 **행 추가**, 잘못된 구간 **행 삭제**.
5. 맞으면 `verified=YES`.
6. (선택) 정의 자체를 바꾸려면 `make_silver_GT.py`의 `DELTA`(°), `WB_MODE`(core/quorum/any), `T_MIN`를 조정 후 재생성.

## 다음 단계 (검수 후)
`python ../eval_vs_GT.py` 실행 → 검수 GT 대비 분석기 검출의 **frame-level F1 / IoU / MoF**를 파라미터 설정별로 산출 = **정확도 기반 파라미터 민감도**.
- ⚠️ scope: 본 GT = *순수* static(위험필터 없음). 분석기 검출 = safe-zone(REBA 중립대역) gating 적용 → 중립자세 유지는 분석기가 의도적으로 제외하므로 recall<1이 정상. **파라미터 간 F1 변화(Δ)** 가 민감도의 핵심 출력이다.
