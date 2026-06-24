import sys
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# [CORE] 설정 (Configuration)
# ==============================================================================


@dataclass
class AnalysisConfig:
    fps: float = 30.0
    t_min: int = 120  # 4초

    hysteresis_frames: int = 15
    trunk_hysteresis: int = 20
    knee_hysteresis: int = 20
    support_hysteresis: int = 20

    smoothing_window: int = 3
    sd_window: int = 15

    # SD 임계값
    sd_min: float = 0.0
    sd_a: float = 0.1
    sd_b: float = 0.15
    sd_leg: float = 0.0015

    # 부위별 계수 (Trunk만 1.5 -> 1.3으로 하향 조정)
    n_coef: float = 2.9  # Neck
    t_coef: float = 1.8  # Trunk
    u_coef: float = 6.1  # Upper arm
    l_coef: float = 3.0  # Lower arm
    w_coef: float = 2.2  # Wrist
    k_coef: float = 2.5  # Knee
    s_coef: float = 1.0  # Support

    key_joints: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("neck", "flexion"),
            ("neck", "bending"),
            ("trunk", "flexion"),
            ("trunk", "bending"),
            ("trunk", "twisting"),
            ("upperarm", "left_flexion"),
            ("upperarm", "right_flexion"),
            ("upperarm", "left_abduction"),
            ("upperarm", "right_abduction"),
            ("lower arm", "left_flexion"),
            ("lower arm", "right_flexion"),
            ("wrist", "left_flexion"),
            ("wrist", "right_flexion"),
            ("wrist", "left_twisting"),
            ("wrist", "right_twisting"),
            ("knee", "left_flexion"),
            ("knee", "right_flexion"),
            ("leg_support", "ratio"),
        ]
    )


# ==============================================================================
# [CORE] 분석 엔진 (Analysis Engine)
# ==============================================================================


class StaticPostureAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.cfg = config

    def _get_sd_threshold(self, joint_name: str, angle: float) -> float:
        # 1. 계수 선택
        coef = 1.0
        if "neck" in joint_name:
            coef = self.cfg.n_coef
        elif "trunk" in joint_name:
            coef = self.cfg.t_coef
        elif "upperarm" in joint_name:
            coef = self.cfg.u_coef
        elif "lower arm" in joint_name:
            coef = self.cfg.l_coef
        elif "wrist" in joint_name:
            coef = self.cfg.w_coef
        elif "knee" in joint_name:
            coef = self.cfg.k_coef
        elif "leg_support" in joint_name:
            coef = self.cfg.s_coef

        adj_a = self.cfg.sd_a * coef
        adj_b = self.cfg.sd_b * coef
        adj_leg = self.cfg.sd_leg * coef

        # 2. 관절별 로직 (Safe Zone에서는 0 반환 -> OR 로직에서 걸러짐)
        if "neck" in joint_name:
            if "flexion" in joint_name:
                return (
                    0 if -5 <= angle <= 20 else adj_a
                )  # [FIX] REBA neck score-1(-5..20) 일치(기존 15)
            if "bending" in joint_name:
                return 0 if abs(angle) <= 20 else adj_a
            if "twisting" in joint_name:
                return 0 if abs(angle) <= 20 else adj_a

        if "trunk" in joint_name:
            if "flexion" in joint_name:
                if -5 <= angle <= 5:  # [FIX] REBA trunk score-1(-5..5) 일치(기존 10)
                    return 0
                elif angle <= 20:
                    return adj_a
                elif angle <= 60:
                    return adj_b
                else:
                    return adj_b
            if "bending" in joint_name:
                return 0 if abs(angle) <= 20 else adj_a
            if "twisting" in joint_name:
                return 0 if abs(angle) <= 20 else adj_a

        if "upperarm" in joint_name:
            if "flexion" in joint_name:
                if -20 <= angle <= 20:
                    return 0
                elif angle <= 45:
                    return adj_a
                elif angle <= 90:
                    return adj_b
                else:
                    return adj_b
            if "abduction" in joint_name:
                return adj_a if angle > 45 else 0

        if "lower arm" in joint_name:
            # [T4-FIX] REBA elbow: phi=90-flex; safe=phi[60,100]→flex[-10,30]. 기존 ≤20은 flex20-30 누락.
            return 0 if -10 <= angle <= 30 else adj_a

        if "wrist" in joint_name:
            if "flexion" in joint_name:
                return 0 if abs(angle) <= 15 else adj_a
            if "twisting" in joint_name:
                return 0 if abs(angle) <= 45 else adj_a

        if "knee" in joint_name:
            if angle < 30:
                return 0
            elif angle < 60:
                return adj_a
            else:
                return adj_b

        if "leg_support" in joint_name:
            return adj_leg if (angle < 0.48 or angle > 0.52) else 0

        return adj_a

    def _core_detection_loop(self, joint, frames, ang, v_sd):
        joint_str = f"{joint[0]}_{joint[1]}"
        hys = self.cfg.hysteresis_frames
        if "trunk" in joint_str:
            hys = self.cfg.trunk_hysteresis
        if "knee" in joint_str:
            hys = self.cfg.knee_hysteresis
        if "leg_support" in joint_str:
            hys = self.cfg.support_hysteresis

        segments = []
        is_static = False
        start_idx = 0
        unstable_cnt = 0

        def _add_seg(s, e, reason, th):
            dur = e - s + 1
            if dur >= self.cfg.t_min:
                segments.append(
                    {
                        "joint": joint,
                        "start_frame_idx": s,
                        "end_frame_idx": e,
                        "start_frame": int(frames[s]),
                        "end_frame": int(frames[e]),
                        "length_frames": dur,
                        "avg_angle": ang.iloc[s : e + 1].mean(),
                        "stop_sign": reason,
                        "threshold_used": th,
                    }
                )

        for t in range(len(ang)):
            val_sd = v_sd.iat[t]
            if pd.isna(val_sd):
                if is_static:
                    is_static = False
                    _add_seg(start_idx, t - 1, "NaN", 0)
                continue

            target_th = self._get_sd_threshold(joint_str, ang.iat[t])
            # target_th가 0이면 감지 대상 아님 -> real_th도 0 -> 아래 로직에서 static 진입 불가
            real_th = self.cfg.sd_min if target_th == 0 else target_th

            if not is_static:
                # real_th가 0보다 클 때만 진입 가능 (Safe Zone 필터링)
                if real_th > 0 and val_sd < real_th:
                    is_static = True
                    start_idx = t
                    unstable_cnt = 0
            else:
                if val_sd >= real_th:
                    unstable_cnt += 1
                    if unstable_cnt >= hys:
                        is_static = False
                        end_idx = max(start_idx, t - hys)
                        reason = (
                            f"sd{(val_sd/real_th):.1f}" if real_th > 0 else "sd_inf"
                        )
                        _add_seg(start_idx, end_idx, reason, real_th)
                else:
                    unstable_cnt = 0

        if is_static:
            last_th = self._get_sd_threshold(joint_str, ang.iat[-1])
            _add_seg(start_idx, len(ang) - 1, "EndOfFile", last_th)

        return segments, None

    def generate_integrated_report(
        self, all_segments, total_frames, frames_idx, available_joints
    ) -> pd.DataFrame:

        # [OR Logic] Safe Zone(0) 때문에, 하나라도 감지되면 유효한 정적 부하로 판단해야 함.
        def _make_mask(j_list):
            m = np.zeros(total_frames, dtype=int)
            valid_joints = [j for j in j_list if j in available_joints]
            if not valid_joints:
                return None

            # 리스트에 있는 관절 중 '하나라도' 세그먼트가 있다면 마킹 (OR)
            for s in [x for x in all_segments if x["joint"] in valid_joints]:
                m[s["start_frame_idx"] : s["end_frame_idx"] + 1] = 1
            return m

        # 1. 그룹 정의 (원래대로 다리 통합, OR 로직 사용)
        groups_def = {
            "Neck": [("neck", "flexion"), ("neck", "bending"), ("neck", "twisting")],
            "Trunk": [
                ("trunk", "flexion"),
                ("trunk", "bending"),
                ("trunk", "twisting"),
            ],
            "Legs": [
                ("knee", "left_flexion"),
                ("knee", "right_flexion"),
                ("leg_support", "ratio"),
            ],
            "Arms (L)": [
                ("upperarm", "left_flexion"),
                ("upperarm", "left_abduction"),
                ("lower arm", "left_flexion"),
                ("wrist", "left_flexion"),
                ("wrist", "left_twisting"),
            ],
            "Arms (R)": [
                ("upperarm", "right_flexion"),
                ("upperarm", "right_abduction"),
                ("lower arm", "right_flexion"),
                ("wrist", "right_flexion"),
                ("wrist", "right_twisting"),
            ],
        }

        # 2. 전신 로직 (OR 연산)
        wb_components = ["Neck", "Trunk", "Legs", "Arms (L)", "Arms (R)"]
        valid_masks = []
        for name in wb_components:
            m = _make_mask(groups_def[name])
            if m is not None:
                valid_masks.append(m)

        integrated_data = []

        # (A) Whole Body Segments
        if valid_masks:
            wb_mask = np.zeros(total_frames, dtype=int)
            for m in valid_masks:
                wb_mask |= m  # OR 연산: 어느 부위라도 부하가 걸리면 전신 부하로 간주

            diff = np.diff(np.concatenate(([0], wb_mask, [0])))
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1

            for s, e in zip(starts, ends):
                dur = (e - s + 1) / self.cfg.fps
                if dur >= (self.cfg.t_min / self.cfg.fps):
                    integrated_data.append(
                        {
                            "Category": "Total",
                            "Part": "Whole Body",
                            "start_frame": int(frames_idx[s]),
                            "end_frame": int(frames_idx[e]),
                            "duration_sec": round(dur, 2),
                        }
                    )

        # (B) Body Parts Segments
        for name, j_list in groups_def.items():
            m = _make_mask(j_list)
            if m is None:
                continue

            diff = np.diff(np.concatenate(([0], m, [0])))
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1

            for s, e in zip(starts, ends):
                dur = (e - s + 1) / self.cfg.fps
                if dur >= (self.cfg.t_min / self.cfg.fps):
                    integrated_data.append(
                        {
                            "Category": "Part",
                            "Part": name,
                            "start_frame": int(frames_idx[s]),
                            "end_frame": int(frames_idx[e]),
                            "duration_sec": round(dur, 2),
                        }
                    )

        if not integrated_data:
            return pd.DataFrame(
                columns=["Category", "Part", "start_frame", "end_frame", "duration_sec"]
            )

        return pd.DataFrame(integrated_data)

    def _save_wide_segments(
        self, segments: List[Dict], target_joints: List[Tuple], output_path: Path
    ) -> None:
        sub_cols = ["start_frame", "end_frame", "angle", "duration", "stop_sign"]
        header = []
        for j in target_joints:
            for sub_col in sub_cols:
                header.append((j[0], j[1], sub_col))

        mi_cols = pd.MultiIndex.from_tuples(header, names=["Joint", "Angle", "Metric"])

        if not segments:
            pd.DataFrame(columns=mi_cols).to_csv(output_path, index=False)
            return

        df_seg = pd.DataFrame(segments)
        if "length_frames" in df_seg.columns:
            df_seg["duration"] = df_seg["length_frames"] / self.cfg.fps
        if "avg_angle" in df_seg.columns:
            df_seg.rename(columns={"avg_angle": "angle"}, inplace=True)

        joint_dfs = []
        for j in target_joints:
            mask = df_seg["joint"] == j
            if mask.any():
                df_j = (
                    df_seg[mask][sub_cols]
                    .sort_values("start_frame")
                    .reset_index(drop=True)
                )
            else:
                df_j = pd.DataFrame(columns=sub_cols)

            df_j.columns = pd.MultiIndex.from_product(
                [[j[0]], [j[1]], sub_cols], names=["Joint", "Angle", "Metric"]
            )
            joint_dfs.append(df_j)

        if joint_dfs:
            final_df = pd.concat(joint_dfs, axis=1)
            final_df = final_df.reindex(columns=mi_cols)
            final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    def process_file(self, csv_path: Path, output_dir: Path) -> Dict[str, Any]:
        try:
            df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        except Exception as e:
            raise ValueError(f"CSV Read Error: {e}")

        avail_cols = df.columns.tolist()
        target_joints = [k for k in self.cfg.key_joints if k in avail_cols]
        if not target_joints:
            raise ValueError("No matching key joints found.")

        # Smoothing & SD
        df_smooth = (
            df[target_joints]
            .rolling(window=self.cfg.smoothing_window, center=True, min_periods=1)
            .mean()
        )

        df_v = df_smooth.diff().abs()
        df_v_sd = (
            df_v.rolling(window=self.cfg.sd_window, center=True, min_periods=1)
            .std()
            .fillna(0)
        )

        all_segs = []
        frames_idx = df.index.to_numpy(dtype=int)

        for joint in target_joints:
            segs, _ = self._core_detection_loop(
                joint, frames_idx, df_smooth[joint], df_v_sd[joint]
            )
            all_segs.extend(segs)

        df_integrated = self.generate_integrated_report(
            all_segs, len(df), frames_idx, target_joints
        )

        base_name = csv_path.stem.replace("_ANGLES", "")

        # 1. 개별 관절 세부 데이터 저장
        seg_dir = output_dir / "segments"
        seg_dir.mkdir(exist_ok=True)
        seg_out_path = seg_dir / f"{base_name}_segments_wide.csv"
        self._save_wide_segments(all_segs, target_joints, seg_out_path)

        # 2. 통합 분석 결과 저장
        dur_dir = output_dir / "duration"
        dur_dir.mkdir(exist_ok=True)
        vis_out_path = dur_dir / f"{base_name}_integrated_analysis.csv"
        df_integrated.to_csv(vis_out_path, index=False, encoding="utf-8-sig")

        # 3. 요약 통계 계산
        wb_data = df_integrated[df_integrated["Part"] == "Whole Body"]
        total_static = wb_data["duration_sec"].sum() if not wb_data.empty else 0
        total_frames = len(df)
        total_time_sec = total_frames / self.cfg.fps

        real_event_count = len(wb_data)

        if total_time_sec > 0:
            freq_per_min = (real_event_count / total_time_sec) * 60
            static_ratio = (total_static / total_time_sec) * 100
        else:
            freq_per_min = 0.0
            static_ratio = 0.0

        mean_holding = (
            (total_static / real_event_count) if real_event_count > 0 else 0.0
        )

        return {
            "VIDEO_NM": base_name,
            "Total frame": total_frames,
            "Total duration (s)": round(total_time_sec, 2),
            "Frequency (events/min)": round(freq_per_min, 2),
            "Mean holding time (s)": round(mean_holding, 2),
            "Total static duration (s)": round(total_static, 2),
            "Static posture ratio (%)": round(static_ratio, 2),
        }


def main():
    root = tk.Tk()
    root.withdraw()

    print("▶ 분석할 CSV 파일들을 선택하세요...")
    file_paths = filedialog.askopenfilenames(
        title="Select CSV Files", filetypes=[("CSV", "*.csv"), ("All", "*.*")]
    )
    if not file_paths:
        return

    print("▶ 결과물을 저장할 폴더를 선택하세요...")
    out_dir = filedialog.askdirectory(title="Select Output Folder")
    if not out_dir:
        return

    config = AnalysisConfig()
    analyzer = StaticPostureAnalyzer(config)

    out_path = Path(out_dir)
    results = []

    print(f"\nProcessing {len(file_paths)} files...")
    for f in file_paths:
        try:
            res = analyzer.process_file(Path(f), out_path)
            results.append(res)
            print(f"[OK] {res['VIDEO_NM']}")
        except Exception as e:
            print(f"[ERR] {Path(f).name}: {e}")
            import traceback

            traceback.print_exc()

    if results:
        df_results = pd.DataFrame(results)

        target_cols = [
            "VIDEO_NM",
            "Frequency (events/min)",
            "Static posture ratio (%)",
            "Mean holding time (s)",
        ]

        avail_cols = [c for c in target_cols if c in df_results.columns]
        summary = df_results[avail_cols]

        numeric_cols = summary.select_dtypes(include=[np.number]).columns
        avg_data = summary[numeric_cols].mean().round(2).to_dict()
        avg_data["VIDEO_NM"] = "Average"

        df_avg = pd.DataFrame([avg_data])
        summary = pd.concat([summary, df_avg], ignore_index=True)

        save_path = out_path / "TOTAL_SUMMARY_REPORT.csv"
        summary.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"\n[Done] All files saved to: {out_path}")


if __name__ == "__main__":
    main()
