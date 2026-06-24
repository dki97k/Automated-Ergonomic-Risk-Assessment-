# Field Plausibility by Occlusion Level, All Frames

This analysis compares SAM-3DBody and AlphaPose-MotionBERT on all field frames with temporal bounding-box fallback. `severe_detected` and `severe_detection_failure_temporal_bbox` are combined as `severe`.

BLC-CV and JAF are computed using the original `m1_eval.py` definitions: COCO-17 index space, 12 bones for BLC-CV, and six joint-angle checks for JAF invalid percentage.

## Label counts

occlusion_level  n_frames
           none     29247
        partial      7314
         severe      3930

## Summary

            pipeline occlusion_level     n    ok  failure_rate_percent   BLC_CV  JAF_invalid_percent
AlphaPose-MotionBERT            none 29247 29247              0.000000 0.388363            23.708984
          SAM-3DBody            none 29247 29247              0.000000 0.157035             0.021085
AlphaPose-MotionBERT         partial  7314  7314              0.000000 0.484750             8.768572
          SAM-3DBody         partial  7314  7313              0.013672 0.181750             0.002279
AlphaPose-MotionBERT          severe  3930  3930              0.000000 0.524286            16.344360
          SAM-3DBody          severe  3930  3930              0.000000 0.196926             0.000000
