# Field Plausibility Recomputed with m1_eval.py Definitions

BLC and JAF were recomputed using the original `m1_eval.py` definitions: COCO-17 index space, 12 bones for BLC-CV, and six joint-angle checks for JAF invalid percentage. Lower is better for both metrics.

            pipeline                  field_occlusion_label  n  BLC_CV_m1eval  JAF_invalid_percent_m1eval
AlphaPose-MotionBERT                                   none 50       0.410463                   24.666667
AlphaPose-MotionBERT                                partial 50       0.495351                   11.666667
AlphaPose-MotionBERT                        severe_detected 50       0.380043                   21.000000
AlphaPose-MotionBERT severe_detection_failure_temporal_bbox 50       0.545371                   16.666667
          SAM-3DBody                                   none 50       0.146116                    0.000000
          SAM-3DBody                                partial 50       0.192786                    0.000000
          SAM-3DBody                        severe_detected 50       0.135977                    0.000000
          SAM-3DBody severe_detection_failure_temporal_bbox 50       0.192420                    0.000000
