# =====================================================
# 파일명: _rep_eval.py
# 역할: Module 2 Part④(Repetition) 재현가능 평가 — peak-count 모델의 검출 품질 정량화
# 입력: rep_peak_eval/Repetition frames_kk.xlsx (clip별 sheet: peak_frame/rep_frame + Likert Score 1–5)
#       [(b) 설계용·미존재] true per-rep count GT, true per-rep timestamp GT
# 출력: _rep_consistency_eval.csv (part a: clip별 + pooled consistency)
#       part (b)는 GT 부재로 *설계만* 제공(실행 시 SKIP 안내). 수치 날조 없음.
# 의존: pandas, numpy, openpyxl
# =====================================================
#
# 설계 의도(Module 2 CONTRACT §0 정합):
#   본 모델은 COUNT_MODE="peaks" — fused periodicity의 peak 검출 = 반복 이벤트(pipeline_reps.py).
#   각 peak는 시점특정(time-localized)·사람이 영상에서 검증 가능한 이벤트 → explainability/reproducibility.
#   따라서 "검출된 각 peak가 실제 반복인가"를 사람이 Likert(1–5)로 라벨 → consistency가
#   true-count GT 없이도 산출 가능한 1차 surrogate 지표(positive predictive quality).
#
#   ⚠️ consistency는 *precision-류*(검출 peak의 타당성)만 측정한다.
#      놓친 반복(recall), 절대 개수정확도(count MAE/OBO), 시점정확도(localization)는
#      true per-rep GT가 있어야 측정 가능 → part (b)에 *설계만* 두고 실행하지 않는다(날조 금지).

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- 0. 경로 설정 -----------------------------------
HERE = Path(__file__).parent  # 03_3REPETITION/
GT_XLSX = HERE / "rep_peak_eval" / "Repetition frames_kk.xlsx"  # clip별 peak + Likert
OUT_CSV = HERE / "_rep_consistency_eval.csv"  # part (a) 산출물


# --- 1. GT 로드 유틸 --------------------------------
def _find_score_col(df: pd.DataFrame):
    """
    의도: sheet마다 컬럼명이 흔들려도 Likert Score 컬럼을 견고하게 찾는다.
    입력: 한 clip sheet(DataFrame, header=0).
    출력: Score 컬럼명(str). 없으면 ValueError.
    """
    for c in df.columns:  # 'Score' 정확매칭 우선(대소문자·공백 무시)
        if str(c).strip().lower() == "score":
            return c
    raise ValueError("Score 컬럼을 찾지 못함")  # 스키마 위반은 조용히 넘기지 않음


def load_clip_likert(xlsx_path: Path) -> pd.DataFrame:
    """
    의도: clip별 Likert Score를 long-format으로 모은다(peak 1개=행 1개).
    입력: Repetition frames_kk.xlsx 경로(sheet=clip).
    출력: columns=[clip, score] DataFrame (NaN score 제외).
    """
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")  # sheet 목록 = clip 목록
    rows = []
    for sn in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sn, header=0, engine="openpyxl")
        score_col = _find_score_col(df)  # peak_frame/rep_frame 컬럼명 차이는 무시
        s = pd.to_numeric(
            df[score_col], errors="coerce"
        ).dropna()  # 빈칸·문자행은 자동 제외
        for v in s.values:
            rows.append({"clip": sn, "score": float(v)})
    return pd.DataFrame(rows)


# --- 2. (a) consistency 산출 [실행됨] ---------------
def compute_consistency(long_df: pd.DataFrame):
    """
    의도: clip별 + pooled repetition consistency 계산.
          consistency = 검출 peak에 대한 human Likert의 분포 요약(검출 타당성).
    입력: long_df(columns=[clip, score]).
    출력: (per_clip DataFrame, pooled dict).
    """

    def _agg(s: pd.Series) -> dict:
        s = s.astype(float)
        return dict(
            n_peaks=int(s.size),  # 검출 peak 수 = Likert 라벨 수
            mean_Likert=round(float(s.mean()), 4),  # 평균 신뢰도(주 지표)
            median=float(s.median()),
            pct_ge4=round(100.0 * float((s >= 4).mean()), 2),  # "확실한 반복" 비율
            pct_ge3=round(100.0 * float((s >= 3).mean()), 2),  # "수용 가능" 비율
        )

    # clip별 집계 — groupby로 sheet 단위 요약
    per = (
        long_df.groupby("clip")["score"]
        .apply(lambda s: pd.Series(_agg(s)))
        .unstack()
        .reset_index()
    )
    per["n_peaks"] = per["n_peaks"].astype(int)  # groupby 후 float화 방지

    # pooled = 모든 peak를 한 풀로(=peak-가중; clip 평균이 아님, 표본 수 반영)
    pooled = _agg(long_df["score"])
    return per, pooled


# --- 3. (b) 설계만: count-MAE/OBO + localization-F1 [미실행: GT 부재] ---
#
#   아래는 *true per-rep count/timestamp GT가 생기면* 채워질 평가의 설계다.
#   현재 GT는 consistency(Likert)뿐이라 실행하지 않는다 — 추정·날조 금지(CONTRACT §3).
#
#   (b-1) count accuracy  [needs: clip별 true 반복수 GT]
#       MAE = mean_clip |n_pred - n_true|
#       OBO(off-by-one) = mean_clip 1[ |n_pred - n_true| <= 1 ]
#       n_pred = repetitions_centers.csv의 peak 수(또는 summary repetitions_total_peaks).
#
#   (b-2) localization F1@tol  [needs: clip별 true per-rep timestamp GT (초/frame)]
#       pred peak ↔ GT rep을 |t_pred - t_gt| <= tol(예: ±0.5·period 또는 ±1s)로 1:1 그리디 매칭.
#       TP=매칭쌍, FP=미매칭 pred, FN=미매칭 GT.
#       precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2PR/(P+R).
#       → consistency가 못 보는 recall(놓친 반복)·시점정확도를 정량화.
#
#   (b-3) period agreement  [needs: GT period/cadence]
#       MAE(refined_period_sec_pred, period_true), Bland–Altman.

_PART_B_DESIGN = """\
[part (b) — DESIGNED, NOT RUN]  사유: true per-rep count/timestamp GT 부재(있는 GT는 Likert consistency뿐).
 (b-1) count MAE / OBO         needs: clip별 true 반복수 GT
 (b-2) localization F1@tol     needs: clip별 true per-rep timestamp GT  (그리디 1:1, tol=±0.5·period 또는 ±1s)
 (b-3) period agreement (MAE)  needs: GT period/cadence
 → true-count GT 수집 시 본 함수에 구현·실행하여 peak-count의 *해석가능성*을 완결 정량화한다(I5 한계와 정합).
"""


def run_part_b_or_skip():
    """
    의도: GT 유무를 확인하고, 없으면 *수치 산출 없이* 설계만 출력(날조 차단).
    입력: 없음(향후 GT 경로를 인자로 받도록 확장).
    출력: None(콘솔에 SKIP 사유 + 설계 출력).
    """
    print(_PART_B_DESIGN)  # GT 생기면 위 (b-1~3) 구현 후 교체


# --- 4. 메인 ----------------------------------------
def main():
    if not GT_XLSX.exists():  # 입력 무결성 우선 확인
        print(f"[ERROR] GT 없음: {GT_XLSX}", file=sys.stderr)
        sys.exit(1)

    long_df = load_clip_likert(GT_XLSX)  # (a) GT 로드
    per, pooled = compute_consistency(long_df)  # (a) 산출

    # CSV 저장 — 기존 산출물과 동일 스키마 유지(clip,n_peaks,mean_Likert,median,pct_ge4,pct_ge3)
    per_out = per[["clip", "n_peaks", "mean_Likert", "median", "pct_ge4", "pct_ge3"]]
    per_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 콘솔 리포트
    print("=== (a) Repetition consistency [RUN] ===")
    print(per_out.to_string(index=False))
    print(
        f"\nPOOLED (peak-weighted, n={pooled['n_peaks']}): "
        f"mean_Likert={pooled['mean_Likert']:.4f} / 5, "
        f"pct>=4={pooled['pct_ge4']:.2f}%, pct>=3={pooled['pct_ge3']:.2f}%"
    )
    print(f"[saved] {OUT_CSV}")
    print()
    run_part_b_or_skip()  # (b) 설계만


if __name__ == "__main__":
    main()
