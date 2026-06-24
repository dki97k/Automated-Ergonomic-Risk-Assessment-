import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog
import os

# ==============================================================================
# 1. GROUND TRUTH (GT) 데이터베이스
# ==============================================================================
gt_database = {
    "MansoryBrickLaying_00": {
        "Frames": [212, 652, 1053, 1448],
        "Data": {
            "Neck": [1, 1, 2, 2], 
            "Trunk": [1, 4, 1, 1], 
            "Leg": [2, 2, 1, 1],
            "UpperArm": [3, 2, 3, 1], 
            "LowerArm": [2, 2, 2, 2], 
            "Wrist": [2, 1, 2, 3]
        }
    },
    "MansoryBrickLaying_01": {
        "Frames": [226, 693, 1078, 1520, 1840, 2410, 2814, 3145, 3591],
        "Data": {
            "Neck": [2, 2, 2, 2, 2, 2, 2, 1, 2], 
            "Trunk": [1, 3, 1, 1, 1, 1, 3, 2, 1],
            "Leg": [1, 2, 1, 1, 1, 1, 2, 1, 1], 
            "UpperArm": [1, 2, 2, 3, 2, 2, 3, 3, 3],
            "LowerArm": [1, 2, 2, 2, 2, 2, 2, 2, 2], 
            "Wrist": [2, 2, 2, 2, 2, 1, 2, 2, 1]
        }
    },
    "MansoryBrickLaying_02": {
        "Frames": [318, 584, 947, 1443, 1709, 2156, 2581, 3010, 3509, 3742, 4236, 4662, 4907, 5473, 5749, 6121, 6781, 7010],
        "Data": {
            "Neck": [2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2],
            "Trunk": [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1, 2, 1, 1],
            "Leg": [1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
            "UpperArm": [1, 1, 1, 4, 1, 3, 1, 2, 2, 3, 2, 2, 2, 1, 1, 3, 1, 2],
            "LowerArm": [1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1],
            "Wrist": [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1]
        }
    },
    "MansoryCement_02": {
        "Frames": [231, 676, 1052, 1413, 1915, 2247, 2852, 3145, 3653, 4007, 4476, 4798],
        "Data": {
            "Neck": [1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2],
            "Trunk": [1, 1, 3, 1, 3, 2, 1, 1, 1, 3, 1, 1],
            "Leg": [1, 2, 1, 1, 1, 1, 3, 1, 2, 3, 1, 1],
            "UpperArm": [1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1],
            "LowerArm": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2],
            "Wrist": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
    },
    "RebarPlacement_00": {
        "Frames": [203, 757, 1147, 1451, 1966, 2087, 2529, 2883, 3383, 3810, 4175, 4770, 5014, 5394, 5888, 6156, 6786, 6944, 7659, 7991, 8165],
        "Data": {
            "Neck": [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2],
            "Trunk": [3, 3, 3, 1, 2, 1, 1, 1, 2, 1, 3, 3, 4, 3, 4, 3, 2, 2, 2, 3, 2],
            "Leg": [3, 3, 3, 2, 1, 1, 3, 3, 3, 2, 3, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1],
            "UpperArm": [3, 3, 3, 1, 2, 2, 3, 2, 2, 1, 3, 3, 2, 2, 3, 1, 1, 1, 1, 2, 1],
            "LowerArm": [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2],
            "Wrist": [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2]
        }
    },
    "RebarTying_00": {
        "Frames": [224, 652, 1079, 1481],
        "Data": {
            "Neck": [2, 2, 2, 2], 
            "Trunk": [3, 4, 3, 3], 
            "Leg": [3, 3, 3, 3],
            "UpperArm": [2, 3, 2, 3], 
            "LowerArm": [2, 2, 2, 2], 
            "Wrist": [2, 1, 2, 2]
        }
    },
    "RebarTying_01": {
        "Frames": [251, 728, 1185, 1664],
        "Data": {
            "Neck": [2, 2, 1, 2], 
            "Trunk": [3, 3, 1, 4], 
            "Leg": [4, 4, 4, 4],
            "UpperArm": [2, 2, 2, 4], 
            "LowerArm": [2, 2, 1, 2], 
            "Wrist": [1, 1, 2, 2]
        }
    },
    "WallPlacement_00": {
        "Frames": [397, 663, 1172, 1402, 1658, 2312, 2736, 3056, 3548, 3836, 4252, 4616, 4851, 5290, 5662, 6320, 6589, 7074, 7646, 7774, 8245, 8795, 8977, 9318, 9811, 10134],
        "Data": {
            "Neck": [2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2],
            "Trunk": [4, 3, 1, 1, 1, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1],
            "Leg": [4, 4, 1, 1, 1, 2, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, 4, 1, 1, 1],
            "UpperArm": [3, 2, 1, 1, 2, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 3, 1, 2, 1, 1, 2, 2, 1, 2, 1],
            "LowerArm": [2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2],
            "Wrist": [2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1]
        }
    }
}


# ==============================================================================
# 2. 평가 로직 함수 (Format: "RMSE (±SD)") + 데이터 수집
# ==============================================================================
# 전역 변수에 전체 데이터 모으기 (Pooled Average 계산용)
global_pooled_data = {
    "Neck": {"gt": [], "pred": []},
    "Trunk": {"gt": [], "pred": []},
    "Leg": {"gt": [], "pred": []},
    "UpperArm": {"gt": [], "pred": []},
    "LowerArm": {"gt": [], "pred": []},
    "Wrist": {"gt": [], "pred": []},
    "ALL": {"gt": [], "pred": []} # 전체 통합
}

def evaluate_file(video_name, filepath):
    gt_info = gt_database.get(video_name)
    if not gt_info:
        return None

    try:
        pred_df = pd.read_csv(filepath)
    except Exception as e:
        print(f"파일 읽기 오류 ({video_name}): {e}")
        return None

    target_frames = gt_info["Frames"]
    body_parts = ["Neck", "Trunk", "Leg", "UpperArm", "LowerArm", "Wrist"]
    
    result_row = {"VIDEO_NM": video_name}
    
    video_all_gt = []
    video_all_pred = []

    for part in body_parts:
        gt_scores = gt_info["Data"][part]
        pred_scores = []
        
        for frame in target_frames:
            val = pred_df.loc[pred_df['frame'] == frame, part]
            if not val.empty:
                pred_val = val.values[0]
            else:
                pred_val = 0 # 데이터 없음
            pred_scores.append(pred_val)

        # -----------------------------------------------------
        # [Pooled Data 수집] - 단순 평균이 아님을 증명하기 위해
        # -----------------------------------------------------
        global_pooled_data[part]["gt"].extend(gt_scores)
        global_pooled_data[part]["pred"].extend(pred_scores)
        global_pooled_data["ALL"]["gt"].extend(gt_scores)
        global_pooled_data["ALL"]["pred"].extend(pred_scores)
        
        video_all_gt.extend(gt_scores)
        video_all_pred.extend(pred_scores)

        # -----------------------------------------------------
        # [개별 비디오 계산] RMSE & SD
        # -----------------------------------------------------
        mse = mean_squared_error(gt_scores, pred_scores)
        rmse = np.sqrt(mse)
        abs_errors = np.abs(np.array(pred_scores) - np.array(gt_scores))
        sd = np.std(abs_errors)
        
        result_row[part] = f"{rmse:.2f} (±{sd:.2f})"

    # 3. AVE (해당 비디오 내 전체 부위 통합 RMSE)
    # 이것도 부위별 평균을 내는게 아니라, 해당 비디오의 모든 샘플을 모아서 계산
    vid_mse = mean_squared_error(video_all_gt, video_all_pred)
    vid_rmse = np.sqrt(vid_mse)
    vid_abs_err = np.abs(np.array(video_all_pred) - np.array(video_all_gt))
    vid_sd = np.std(vid_abs_err)
    
    result_row["AVE"] = f"{vid_rmse:.2f} (±{vid_sd:.2f})"

    # 4. p-value (Paired t-test)
    if np.array_equal(video_all_gt, video_all_pred):
        result_row["p"] = 1.0
    else:
        try:
            _, p_val = stats.ttest_rel(video_all_gt, video_all_pred)
            result_row["p"] = round(p_val, 3)
        except:
            result_row["p"] = "N/A"
            
    return result_row

# ==============================================================================
# 3. 데이터 후처리: Pooled Average 행 계산 (단순 평균 X)
# ==============================================================================
def calculate_pooled_average_row():
    avg_row = {"VIDEO_NM": "Average (Pooled)"}
    body_parts = ["Neck", "Trunk", "Leg", "UpperArm", "LowerArm", "Wrist"]
    
    # 각 부위별로 전체 데이터를 모아서(Pooling) 한 번에 계산
    for part in body_parts:
        gts = global_pooled_data[part]["gt"]
        preds = global_pooled_data[part]["pred"]
        
        if gts:
            mse = mean_squared_error(gts, preds)
            rmse = np.sqrt(mse)
            abs_err = np.abs(np.array(preds) - np.array(gts))
            sd = np.std(abs_err)
            avg_row[part] = f"{rmse:.2f} (±{sd:.2f})"
        else:
            avg_row[part] = "-"

    # AVE 열: 전체 데이터셋(모든 비디오, 모든 부위) 통합 계산
    all_gts = global_pooled_data["ALL"]["gt"]
    all_preds = global_pooled_data["ALL"]["pred"]
    
    if all_gts:
        total_mse = mean_squared_error(all_gts, all_preds)
        total_rmse = np.sqrt(total_mse)
        total_abs_err = np.abs(np.array(all_preds) - np.array(all_gts))
        total_sd = np.std(total_abs_err)
        avg_row["AVE"] = f"{total_rmse:.2f} (±{total_sd:.2f})"
        
        # [추가됨] 전체 데이터(Pooled)에 대한 p-value 계산
        if np.array_equal(all_gts, all_preds):
             avg_row["p"] = 1.0
        else:
            try:
                _, p_val = stats.ttest_rel(all_gts, all_preds)
                avg_row["p"] = round(p_val, 3)
            except:
                 avg_row["p"] = "N/A"
    else:
        avg_row["AVE"] = "-"
        avg_row["p"] = "-"
        
    return avg_row

# ==============================================================================
# 4. GUI 및 실행
# ==============================================================================
def main():
    root = tk.Tk()
    root.withdraw()

    print(">>> 1. 분석할 CSV 파일들을 선택하세요...")
    file_paths = filedialog.askopenfilenames(
        title="Time Series CSV 파일 선택",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not file_paths:
        print("파일이 선택되지 않았습니다.")
        return

    results = []
    
    # 전역 데이터 초기화
    for key in global_pooled_data:
        global_pooled_data[key]["gt"] = []
        global_pooled_data[key]["pred"] = []

    print("\n[ 처리 중... ]")
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        matched_video = None
        for gt_key in gt_database.keys():
            if gt_key in filename:
                matched_video = gt_key
                break
        
        if matched_video:
            print(f"- {matched_video} 처리 완료")
            res = evaluate_file(matched_video, filepath)
            if res:
                results.append(res)
        else:
            print(f"- [Skip] GT 매칭 안됨: {filename}")

    if results:
        df_final = pd.DataFrame(results)
        
        cols = ["VIDEO_NM", "Neck", "Trunk", "Leg", "UpperArm", "LowerArm", "Wrist", "AVE", "p"]
        df_final = df_final[cols]
        
        # Pooled Average
        avg_row_data = calculate_pooled_average_row()
        df_final = pd.concat([df_final, pd.DataFrame([avg_row_data])], ignore_index=True)
        
        print("\n" + "="*100)
        print("최종 평가 결과 [Format: RMSE (±SD)]")
        print("="*100)
        print(df_final.to_string(index=False))
        
        # [저장 위치 선택]
        print("\n>>> 2. 결과를 저장할 위치를 선택하세요...")
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            initialfile="Final_REBA_RMSE_Pooled_Result.xlsx",
            title="결과 저장 위치 선택"
        )
        
        if save_path:
            df_final.to_excel(save_path, index=False)
            print(f"\n[완료] 결과 저장됨: {save_path}")
        else:
            print("\n[취소] 저장이 취소되었습니다.")
            
    else:
        print("\n결과가 없습니다.")

if __name__ == "__main__":
    main()