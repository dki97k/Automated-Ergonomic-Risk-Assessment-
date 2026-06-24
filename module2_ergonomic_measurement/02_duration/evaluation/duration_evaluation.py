import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# ==============================================================================
# 1. 데이터 로드 및 파싱
# ==============================================================================
def load_and_parse_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.xlsx', '.xls']:
            df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
        else:
            try:
                df_raw = pd.read_csv(file_path, header=None, encoding='utf-8')
            except UnicodeDecodeError:
                df_raw = pd.read_csv(file_path, header=None, encoding='cp949')
    except Exception as e:
        print(f"[Critical Error] {e}")
        return pd.DataFrame()

    data_list = []
    current_video = None

    for index, row in df_raw.iterrows():
        if pd.notna(row[0]) and pd.isna(row[1]):
            current_video = row[0]
            continue
        
        if row[0] in ['Total', 'Part'] and pd.notna(row[2]) and pd.notna(row[7]):
            try:
                sys_start = float(row[2])
                sys_end = float(row[3])
                gt_start = float(row[7])
                gt_end = float(row[8])
                
                if sys_end <= sys_start or gt_end <= gt_start: continue

                data_list.append({
                    'VIDEO_NM': current_video,
                    'Category': row[0],
                    'Part': row[1],
                    'Sys_Frame_S': sys_start,
                    'Sys_Frame_E': sys_end,
                    'GT_Frame_S': gt_start,
                    'GT_Frame_E': gt_end
                })
            except ValueError:
                continue 
    return pd.DataFrame(data_list)

# ==============================================================================
# 2. 통계 및 IoU 계산 함수
# ==============================================================================
def calculate_icc(gt, sys):
    if len(gt) < 2: return 0.0
    n = len(gt)
    data_mat = np.stack([gt, sys], axis=1)
    grand_mean = np.mean(data_mat)
    SST = ((data_mat - grand_mean) ** 2).sum()
    k = 2
    subject_means = np.mean(data_mat, axis=1)
    SS_b = k * ((subject_means - grand_mean) ** 2).sum()
    MS_b = SS_b / (n - 1)
    rater_means = np.mean(data_mat, axis=0)
    SS_j = n * ((rater_means - grand_mean) ** 2).sum()
    MS_j = SS_j / (k - 1)
    SS_e = SST - SS_b - SS_j
    MS_e = SS_e / ((n - 1) * (k - 1))
    numerator = MS_b - MS_e
    denominator = MS_b + (k - 1) * MS_e + (k / n) * (MS_j - MS_e)
    return numerator / denominator if denominator != 0 else 0.0

def calculate_metrics_with_iou(sub_df, fps=30.0):
    if sub_df.empty: return pd.Series()
    n_count = len(sub_df)

    gt_s = sub_df['GT_Frame_S'].values / fps
    gt_e = sub_df['GT_Frame_E'].values / fps
    sys_s = sub_df['Sys_Frame_S'].values / fps
    sys_e = sub_df['Sys_Frame_E'].values / fps

    inter_s = np.maximum(sys_s, gt_s)
    inter_e = np.minimum(sys_e, gt_e)
    intersection = np.maximum(0, inter_e - inter_s)
    
    sys_dur = sys_e - sys_s
    gt_dur = gt_e - gt_s
    union = sys_dur + gt_dur - intersection
    
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union!=0)

    diff_s = sys_s - gt_s
    diff_e = sys_e - gt_e
    
    combined_diff = np.concatenate([diff_s, diff_e])
    comb_bias = np.mean(combined_diff)
    comb_sd = np.std(combined_diff, ddof=1) if len(combined_diff) > 1 else 0
    comb_loa_low = comb_bias - 1.96 * comb_sd
    comb_loa_high = comb_bias + 1.96 * comb_sd
    
    combined_gt = np.concatenate([gt_s, gt_e])
    combined_sys = np.concatenate([sys_s, sys_e])
    comb_icc = calculate_icc(combined_gt, combined_sys)
    
    mabe = (np.mean(np.abs(diff_s)) + np.mean(np.abs(diff_e))) / 2

    mean_bias_s = np.mean(diff_s)
    sd_bias_s = np.std(diff_s, ddof=1) if n_count > 1 else 0
    icc_s = calculate_icc(gt_s, sys_s)
    
    mean_bias_e = np.mean(diff_e)
    sd_bias_e = np.std(diff_e, ddof=1) if n_count > 1 else 0
    icc_e = calculate_icc(gt_e, sys_e)

    success_05 = np.mean((np.abs(diff_s) < 0.5) & (np.abs(diff_e) < 0.5)) * 100
    success_10 = np.mean((np.abs(diff_s) < 1.0) & (np.abs(diff_e) < 1.0)) * 100
    
    miou = np.mean(iou)
    sr_iou_30 = np.mean(iou >= 0.3) * 100
    sr_iou_50 = np.mean(iou >= 0.5) * 100
    sr_iou_70 = np.mean(iou >= 0.7) * 100

    return pd.Series({
        'n': n_count,
        'Combined Bias (SD)': f"{comb_bias:.2f} ({comb_sd:.2f})",
        'Combined 95% LoA': f"{comb_loa_low:.2f} ~ {comb_loa_high:.2f}",
        'Combined ICC': f"{comb_icc:.3f}",
        'MABE (s)': f"{mabe:.2f}",
        'mIoU': f"{miou:.3f}",
        'IoU > 0.3 (%)': f"{sr_iou_30:.1f}",
        'IoU > 0.5 (%)': f"{sr_iou_50:.1f}",
        'IoU > 0.7 (%)': f"{sr_iou_70:.1f}",
        'Start Bias (SD)': f"{mean_bias_s:.2f} ({sd_bias_s:.2f})",
        'ICC (Start)': f"{icc_s:.3f}",
        'End Bias (SD)': f"{mean_bias_e:.2f} ({sd_bias_e:.2f})",
        'ICC (End)': f"{icc_e:.3f}",
        'Success (Strict < 0.5s)': f"{success_05:.1f}%",
        'Success (Strict < 1.0s)': f"{success_10:.1f}%"
    })

# ==============================================================================
# 3. 메인 실행 루틴 (GUI 적용)
# ==============================================================================
if __name__ == "__main__":
    # GUI 설정 (창을 띄우지 않고 대화상자만 사용)
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # 대화상자를 맨 앞으로 가져옴

    print("[알림] 분석할 GT.xlsx 파일을 선택해주세요...")
    
    # 파일 선택창 호출
    file_path = filedialog.askopenfilename(
        title="GT 데이터 파일 선택",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not file_path:
        print("[경고] 파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
        root.destroy()
    else:
        try:
            print(f"[Info] Loading data from {file_path}...")
            df = load_and_parse_data(file_path)
            
            if df.empty:
                print("[Warning] No data loaded.")
            else:
                print(f"[Info] Loaded {len(df)} segments.")
                
                overall_res = calculate_metrics_with_iou(df)
                overall_res.name = 'Overall'
                
                video_res = df.groupby('VIDEO_NM').apply(calculate_metrics_with_iou)
                video_final = pd.concat([video_res, pd.DataFrame([overall_res])])
                
                part_res = df.groupby('Part').apply(calculate_metrics_with_iou)
                part_final = pd.concat([part_res, pd.DataFrame([overall_res])])
                
                main_cols = ['n', 'mIoU', 'IoU > 0.5 (%)', 
                             'Combined Bias (SD)', 'Combined ICC', 'Combined 95% LoA', 
                             'Success (Strict < 0.5s)', 'Success (Strict < 1.0s)']
                
                print("\n" + "="*80)
                print(" [Final Evaluation] Including Combined ICC")
                print("="*80)
                print("\n--- By Video (with Overall) ---")
                print(video_final[main_cols])
                print("\n--- By Body Part (with Overall) ---")
                print(part_final[main_cols])
                
                output_dir = os.path.dirname(file_path)
                output_file = os.path.join(output_dir, 'Evaluation_Result_Final.xlsx')
                
                with pd.ExcelWriter(output_file) as writer:
                    video_final.to_excel(writer, sheet_name='By_Video')
                    part_final.to_excel(writer, sheet_name='By_Part')
                print(f"\n[Info] Saved to '{output_file}'")

        except Exception as e:
            print(f"[Error] {e}")
        finally:
            root.destroy()