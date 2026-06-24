import pandas as pd
import numpy as np
from scipy import stats
import tkinter as tk
from tkinter import filedialog
import os
import sys

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
# 2. 전역 변수 (Pooled Data & Max Diff Tracker)
# ==============================================================================
global_pooled_data = {
    "Neck": {"gt": [], "pred": []},
    "Trunk": {"gt": [], "pred": []},
    "Leg": {"gt": [], "pred": []},
    "UpperArm": {"gt": [], "pred": []},
    "LowerArm": {"gt": [], "pred": []},
    "Wrist": {"gt": [], "pred": []},
    "ALL": {"gt": [], "pred": []}
}

# [NEW] 최대 오차 추적용 딕셔너리
max_diff_tracker = {
    "Neck": {"max_diff": -1, "cases": []},
    "Trunk": {"max_diff": -1, "cases": []},
    "Leg": {"max_diff": -1, "cases": []},
    "UpperArm": {"max_diff": -1, "cases": []},
    "LowerArm": {"max_diff": -1, "cases": []},
    "Wrist": {"max_diff": -1, "cases": []}
}

# ==============================================================================
# 3. 평가 로직 함수 (Bias, SD, Max Diff Check)
# ==============================================================================
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
        
        # [Safety Check] GT 데이터 개수 검증
        if len(gt_scores) != len(target_frames):
            print(f"\n[CRITICAL ERROR] {video_name} - {part}")
            print(f" -> Frame 개수({len(target_frames)})와 GT 데이터 개수({len(gt_scores)})가 다릅니다!")
            print(" -> GT DB를 수정해야 합니다. 건너뜁니다.")
            return None

        # 프레임별 값 추출 및 오차 추적
        for i, frame in enumerate(target_frames):
            val = pred_df.loc[pred_df['frame'] == frame, part]
            if not val.empty:
                pred_val = float(val.values[0])
            else:
                pred_val = 0.0
            pred_scores.append(pred_val)
            
            # --- [NEW] 최대 오차 추적 로직 ---
            gt_val = float(gt_scores[i])
            diff = abs(pred_val - gt_val)
            tracker = max_diff_tracker[part]
            
            # 현재 최대값보다 크면 -> 리셋하고 새로 기록
            if diff > tracker["max_diff"]:
                tracker["max_diff"] = diff
                tracker["cases"] = [{
                    "video": video_name, "frame": frame, 
                    "gt": gt_val, "pred": pred_val
                }]
            # 현재 최대값과 같으면 -> 리스트에 추가 (복수 정답)
            elif diff == tracker["max_diff"]:
                tracker["cases"].append({
                    "video": video_name, "frame": frame, 
                    "gt": gt_val, "pred": pred_val
                })
            # ---------------------------------

        # Pooled Data 수집
        global_pooled_data[part]["gt"].extend(gt_scores)
        global_pooled_data[part]["pred"].extend(pred_scores)
        global_pooled_data["ALL"]["gt"].extend(gt_scores)
        global_pooled_data["ALL"]["pred"].extend(pred_scores)
        
        video_all_gt.extend(gt_scores)
        video_all_pred.extend(pred_scores)

        # 개별 비디오 계산 (Bias, SD)
        errors = np.array(pred_scores) - np.array(gt_scores)
        bias = np.mean(errors)
        sd = np.std(errors)
        
        result_row[part] = f"{bias:.2f} (±{sd:.2f})"

    # AVE (해당 비디오 통합)
    vid_errors = np.array(video_all_pred) - np.array(video_all_gt)
    vid_bias = np.mean(vid_errors)
    vid_sd = np.std(vid_errors)
    result_row["AVE"] = f"{vid_bias:.2f} (±{vid_sd:.2f})"

    # p-value
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
# 4. 데이터 후처리
# ==============================================================================
def calculate_pooled_average_row():
    avg_row = {"VIDEO_NM": "Average (Pooled)"}
    body_parts = ["Neck", "Trunk", "Leg", "UpperArm", "LowerArm", "Wrist"]
    
    for part in body_parts:
        gts = global_pooled_data[part]["gt"]
        preds = global_pooled_data[part]["pred"]
        if gts:
            errors = np.array(preds) - np.array(gts)
            bias = np.mean(errors)
            sd = np.std(errors)
            avg_row[part] = f"{bias:.2f} (±{sd:.2f})"
        else:
            avg_row[part] = "-"

    all_gts = global_pooled_data["ALL"]["gt"]
    all_preds = global_pooled_data["ALL"]["pred"]
    if all_gts:
        all_errors = np.array(all_preds) - np.array(all_gts)
        total_bias = np.mean(all_errors)
        total_sd = np.std(all_errors)
        avg_row["AVE"] = f"{total_bias:.2f} (±{total_sd:.2f})"
    else:
        avg_row["AVE"] = "-"
        
    avg_row["p"] = "-" 
    return avg_row

# ==============================================================================
# 5. GUI 및 메인 실행
# ==============================================================================
def main():
    root = tk.Tk()
    root.withdraw()

    # 1. 입력 파일 선택
    print("Step 1. 평가할 CSV 파일들을 선택하세요 (다중 선택 가능)")
    file_paths = filedialog.askopenfilenames(
        title="Time Series CSV 파일 선택",
        filetypes=[("CSV Files", "*.csv")]
    )

    if not file_paths:
        print("파일이 선택되지 않았습니다. 종료합니다.")
        return

    # 2. 저장 폴더 선택
    print("Step 2. 결과 엑셀 파일을 저장할 폴더를 선택하세요")
    save_dir = filedialog.askdirectory(title="결과 저장 폴더 선택")

    if not save_dir:
        print("저장 폴더가 선택되지 않았습니다. 종료합니다.")
        return

    # 3. 처리 시작
    results = []
    print("\n[ 처리 중... ]")
    
    # 전역 변수 초기화 (중복 실행 대비)
    for key in global_pooled_data:
        global_pooled_data[key]["gt"] = []
        global_pooled_data[key]["pred"] = []
    
    for key in max_diff_tracker:
        max_diff_tracker[key] = {"max_diff": -1, "cases": []}

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        matched_video = None
        for gt_key in gt_database.keys():
            if gt_key in filename:
                matched_video = gt_key
                break
        
        if matched_video:
            print(f"- {matched_video} 처리 중...", end=" ")
            res = evaluate_file(matched_video, filepath)
            if res:
                results.append(res)
                print("OK")
            else:
                print("Skip (Error)")
        else:
            print(f"- [Skip] GT 매칭 안됨: {filename}")

    # 4. 저장 및 종료
    if results:
        df_final = pd.DataFrame(results)
        cols = ["VIDEO_NM", "Neck", "Trunk", "Leg", "UpperArm", "LowerArm", "Wrist", "AVE", "p"]
        df_final = df_final[cols]
        
        avg_row_data = calculate_pooled_average_row()
        df_final = pd.concat([df_final, pd.DataFrame([avg_row_data])], ignore_index=True)
        
        print("\n" + "="*100)
        print("최종 평가 결과 [Format: Bias (±SD)]")
        print("="*100)
        print(df_final.to_string(index=False))
        
        # ----------------------------------------------------------------------
        # [NEW] 터미널에 Worst Cases 출력
        # ----------------------------------------------------------------------
        print("\n" + "="*100)
        print("🚨 [WORST CASES] 각 부위별 GT와 가장 차이가 큰 프레임 (복수 정답 포함)")
        print("="*100)
        
        body_parts_order = ["Neck", "Trunk", "Leg", "UpperArm", "LowerArm", "Wrist"]
        
        for part in body_parts_order:
            tracker = max_diff_tracker[part]
            max_d = tracker["max_diff"]
            cases = tracker["cases"]
            
            if max_d <= 0:
                print(f"[{part}] 오차 없음 (Perfect Match)")
            else:
                print(f"[{part}] 최대 오차: {max_d:.2f}")
                for i, case in enumerate(cases):
                    print(f"  {i+1}. {case['video']} (Frame: {case['frame']}) | GT: {case['gt']} vs Pred: {case['pred']}")
            print("-" * 60)
        # ----------------------------------------------------------------------

        # 저장 경로 생성
        save_filename = "Final_REBA_Bias_Pooled_Result.xlsx"
        save_path = os.path.join(save_dir, save_filename)
        
        try:
            df_final.to_excel(save_path, index=False)
            print(f"\n✅ [완료] 결과 파일 저장됨: {save_path}")
            try:
                os.startfile(save_dir)
            except:
                pass
        except PermissionError:
            print(f"\n❌ [오류] 엑셀 파일이 열려있습니다: {save_path}")
            
    else:
        print("\n결과가 없습니다.")

if __name__ == "__main__":
    main()