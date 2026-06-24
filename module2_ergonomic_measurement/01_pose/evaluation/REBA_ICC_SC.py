import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy import stats
import tkinter as tk
from tkinter import filedialog
import os

# ==============================================================================
# 1. GROUND TRUTH (GT) 데이터베이스
# ==============================================================================
gt_database = {
    "MansoryBrickLaying_00": {
        "Frames": [212, 652, 1053, 1448],
        "Final_GT": [4, 4, 3, 1]
    },
    "MansoryBrickLaying_01": {
        "Frames": [226, 693, 1078, 1520, 1840, 2410, 2814, 3145, 3591],
        "Final_GT": [1, 4, 1, 3, 1, 1, 6, 4, 2]
    },
    "MansoryBrickLaying_02": {
        "Frames": [318, 584, 947, 1443, 1709, 2156, 2581, 3010, 3509, 3742, 4236, 4662, 4907, 5473, 5749, 6121, 6781, 7010],
        "Final_GT": [1, 1, 1, 4, 1, 3, 2, 2, 3, 2, 4, 1, 1, 1, 1, 3, 1, 1 ]
    },
    "MansoryCement_02": {
        "Frames": [231, 676, 1052, 1413, 1915, 2247, 2852, 3145, 3653, 4007, 4476, 4798],
        "Final_GT": [1, 1, 4, 1, 4, 3 ,3, 1, 1, 6 ,1 ,1]
    },
    "RebarPlacement_00": {
        "Frames": [203, 757, 1147, 1451, 1966, 2087, 2529, 2883, 3383, 3810, 4175, 4770, 5014, 5394, 5888, 6156, 6786, 6944, 7659, 7991, 8165],
        "Final_GT": [6, 6, 6, 1, 3, 1, 4 ,3, 4, 2 ,7 ,4 , 4, 4, 6, 4, 3 ,4, 4, 3, 3]
    },
    "RebarTying_00": {
        "Frames": [251, 728, 1185, 1664],
        "Final_GT": [6, 8, 6, 8]
    },
    "RebarTying_01": {
        "Frames": [251, 728, 1185, 1664],
        "Final_GT": [7 ,7, 4, 10]
    },
    "WallPlacement_00": {
        "Frames": [397, 663, 1172, 1402, 1658, 2312, 2736, 3056, 3548, 3836, 4252, 4616, 4851, 5290, 5662, 6320, 6589, 7074, 7646, 7774, 8245, 8795, 8977, 9318, 9811, 10134],
        "Final_GT": [10, 7, 1, 1, 1 ,6 ,6, 6, 6, 4, 4, 4, 6, 4, 4, 4, 3, 1, 1, 1, 4, 1, 7, 1, 1, 1]
    }
}

# ==============================================================================
# 2. 통계 계산 함수
# ==============================================================================
def get_action_level(score):
    """ REBA Action Levels 변환 함수 """
    if score == 1: return 0 
    elif 2 <= score <= 3: return 1
    elif 4 <= score <= 7: return 2
    elif 8 <= score <= 10: return 3
    elif score >= 11: return 4
    return 0

def calculate_icc_consistency(targets, raters):
    try:
        data = np.array([targets, raters]).T
        n, k = data.shape
        grand_mean = np.mean(data)
        SST = np.sum((data - grand_mean)**2)
        row_means = np.mean(data, axis=1)
        SSR = k * np.sum((row_means - grand_mean)**2)
        BMS = SSR / (n - 1)
        col_means = np.mean(data, axis=0)
        SSC = n * np.sum((col_means - grand_mean)**2)
        SSE = SST - SSR - SSC
        EMS = SSE / ((n - 1) * (k - 1))
        icc = (BMS - EMS) / (BMS + (k - 1) * EMS)
        return icc
    except:
        return 0.0

def calculate_spearman(targets, raters):
    try:
        if len(targets) < 2: return 1.0
        coef, _ = stats.spearmanr(targets, raters)
        if np.isnan(coef): return 0.0
        return coef
    except:
        return 0.0

def calculate_weighted_kappa(targets, predictions):
    """
    Quadratic Weighted Kappa 계산
    - Action Level처럼 순서가 있는 등급(Ordinal) 평가에 최적화됨.
    - 정답과의 거리가 멀수록 더 큰 페널티 부여.
    """
    try:
        # weights='quadratic'이 순서형 데이터에 가장 적합
        return cohen_kappa_score(targets, predictions, weights='quadratic')
    except:
        return 0.0

# ==============================================================================
# 3. 평가 로직
# ==============================================================================
pooled_gt_scores = []
pooled_pred_scores = []

# Action Level용 Pooled Data
pooled_gt_levels = []
pooled_pred_levels = []

def evaluate_file(video_name, filepath):
    gt_info = gt_database.get(video_name)
    if not gt_info: return None

    try:
        pred_df = pd.read_csv(filepath)
    except: return None

    target_frames = gt_info["Frames"]
    gt_scores = gt_info["Final_GT"]
    pred_scores = []
    
    for frame in target_frames:
        val = pred_df.loc[pred_df['frame'] == frame, 'Final']
        if not val.empty:
            pred_scores.append(int(val.values[0])) # Ensure int for levels
        else:
            pred_scores.append(0)

    # REBA Score -> Action Level 변환
    gt_levels = [get_action_level(s) for s in gt_scores]
    pred_levels = [get_action_level(s) for s in pred_scores]

    # Pooling (점수 & 레벨 각각 저장)
    pooled_gt_scores.extend(gt_scores)
    pooled_pred_scores.extend(pred_scores)
    
    pooled_gt_levels.extend(gt_levels)
    pooled_pred_levels.extend(pred_levels)
    
    # [지표 1] Raw Score 평가
    icc_val = calculate_icc_consistency(gt_scores, pred_scores)
    spearman_val = calculate_spearman(gt_scores, pred_scores)
    
    # [지표 2] Action Level 평가 (Accuracy & Weighted Kappa)
    acc_val = accuracy_score(gt_levels, pred_levels)
    kappa_val = calculate_weighted_kappa(gt_levels, pred_levels)
    
    return {
        "Video Name": video_name,
        "Consistency (ICC)": f"{icc_val:.3f}",
        "Rank Corr (Spearman)": f"{spearman_val:.3f}",
        "Act.Lvl Acc": f"{acc_val:.2f}",       # 단순 정확도
        "Act.Lvl Kappa": f"{kappa_val:.3f}"     # 가중 카파
    }

# ==============================================================================
# 4. Main GUI
# ==============================================================================
def main():
    root = tk.Tk()
    root.withdraw()

    print(">>> CSV 파일들을 선택하세요 (다중 선택 가능)...")
    file_paths = filedialog.askopenfilenames(title="Time Series CSV 파일 선택", filetypes=[("CSV Files", "*.csv")])

    if not file_paths:
        print("선택된 파일 없음")
        return

    results = []
    pooled_gt_scores.clear()
    pooled_pred_scores.clear()
    pooled_gt_levels.clear()
    pooled_pred_levels.clear()

    print("\n[ 처리 중... ]")
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        matched_video = None
        for gt_key in gt_database.keys():
            if gt_key in filename:
                matched_video = gt_key
                break
        
        if matched_video:
            print(f"- {matched_video} 처리")
            res = evaluate_file(matched_video, filepath)
            if res: results.append(res)
        else:
            print(f"- [Skip] GT 매칭 안됨: {filename}")

    if results:
        df_final = pd.DataFrame(results)
        
        # Overall 계산 (점수)
        overall_icc = calculate_icc_consistency(pooled_gt_scores, pooled_pred_scores)
        overall_spearman = calculate_spearman(pooled_gt_scores, pooled_pred_scores)
        
        # Overall 계산 (레벨)
        overall_acc = accuracy_score(pooled_gt_levels, pooled_pred_levels)
        overall_kappa = calculate_weighted_kappa(pooled_gt_levels, pooled_pred_levels)
        
        overall_row = {
            "Video Name": "Overall (Pooled)",
            "Consistency (ICC)": f"{overall_icc:.3f}",
            "Rank Corr (Spearman)": f"{overall_spearman:.3f}",
            "Act.Lvl Acc": f"{overall_acc:.2f}",
            "Act.Lvl Kappa": f"{overall_kappa:.3f}"
        }
        
        df_final = pd.concat([df_final, pd.DataFrame([overall_row])], ignore_index=True)
        
        print("\n" + "="*90)
        print("★★★ 최종 결과 (Score Consistency & Action Level Accuracy) ★★★")
        print(" [Act.Lvl Kappa] : Quadratic Weighted Kappa (등급간 거리 반영, 0.75 이상이면 매우 우수)")
        print("="*90)
        print(df_final.to_string(index=False))
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, "Final_REBA_Level_Result.xlsx")
        df_final.to_excel(save_path, index=False)
        print(f"\n[완료] 결과 저장됨: {save_path}")
    else:
        print("\n결과 없음")

if __name__ == "__main__":
    main()