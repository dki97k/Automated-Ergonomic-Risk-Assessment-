# Field Plausibility by Occlusion Severity

This analysis compares SAM-3DBody and AlphaPose-MotionBERT on a balanced field-frame subset sampled by occlusion label.

Important caveat: these are plausibility and operational robustness metrics, not 3D ground-truth correctness metrics.

## Outputs

- `field_plausibility_per_pose_metrics.csv`
- `field_plausibility_by_severity_summary.csv`

## Summary

            pipeline                  field_occlusion_label  n  ok  failure_rate_percent  left_right_bone_symmetry_error_mean  joint_angle_feasibility_mean  bone_length_cv_mean
AlphaPose-MotionBERT                                   none 50  50                   0.0                             0.082843                           1.0             0.410171
AlphaPose-MotionBERT                                partial 50  50                   0.0                             0.126013                           1.0             0.488985
AlphaPose-MotionBERT                        severe_detected 50  50                   0.0                             0.132050                           1.0             0.383185
AlphaPose-MotionBERT severe_detection_failure_temporal_bbox 50  50                   0.0                             0.110057                           1.0             0.540947
          SAM-3DBody                                   none 50  50                   0.0                             0.010226                           1.0             0.043113
          SAM-3DBody                                partial 50  50                   0.0                             0.012815                           1.0             0.057008
          SAM-3DBody                        severe_detected 50  50                   0.0                             0.011058                           1.0             0.054091
          SAM-3DBody severe_detection_failure_temporal_bbox 50  50                   0.0                             0.012396                           1.0             0.052594
