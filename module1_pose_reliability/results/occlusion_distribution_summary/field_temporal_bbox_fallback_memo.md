# Temporal Bbox Fallback Ablation

This ablation evaluates the Module 1 temporal bbox fallback on field frames
where YOLO did not detect a person. The fallback borrows the nearest valid
person bbox from the same sequence and runs SAM-3DB body inference.

## Result

- YOLO-only failure frames: 13
- Recovered with temporal bbox fallback: 13/13
- Recovery rate among YOLO failure frames: 100.0%

## Donor Frame Distance

- Mean distance: 196.0 frames
- Median distance: 127.0 frames
- Range: 10-761 frames

## Interpretation

These cases support describing Module 1 as more than a direct YOLO + SAM-3DB
assembly. YOLO-only failed under substantial object/self occlusion or unusual
posture, while the temporal bbox fallback allowed SAM-3DB inference to continue
for all reviewed failure frames.

The fallback does not prove pose correctness in these occluded frames. It should
be reported as a detection-continuity/recovery mechanism and paired with
occlusion severity labels and downstream caution/reliability reporting.

## Outputs

- `field_temporal_bbox_fallback_summary.csv`
- `field_temporal_bbox_fallback_ablation_table.csv`
- `field_temporal_bbox_fallback_distance_summary.csv`
- `field_temporal_bbox_fallback_montage.jpg`
