# Field Plausibility by Three Occlusion Levels

The two severe categories (`severe_detected` and `severe_detection_failure_temporal_bbox`) were combined into a single `severe` group. BLC-CV and relaxed JAF invalid percentage are plausibility metrics, not ground-truth correctness metrics.

            pipeline field_occlusion_label   n   BLC_CV  relaxed_JAF_invalid_percent
AlphaPose-MotionBERT                  none  50 0.410463                     1.333333
AlphaPose-MotionBERT               partial  50 0.495351                    10.333333
AlphaPose-MotionBERT                severe 100 0.510387                     4.333333
          SAM-3DBody                  none  50 0.146116                     0.666667
          SAM-3DBody               partial  50 0.192786                     0.666667
          SAM-3DBody                severe 100 0.177999                     0.833333
