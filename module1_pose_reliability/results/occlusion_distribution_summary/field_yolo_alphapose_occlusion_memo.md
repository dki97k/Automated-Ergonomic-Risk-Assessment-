# YOLO + AlphaPose Field Occlusion Severity Draft

This is a first-pass automatic severity estimate for 120 stratified field
frames. YOLO provides the primary person bounding box, and AlphaPose COCO-17
keypoint confidence is used to compute visible joint ratio.

## Primary Automatic Measure

- Primary ratio: `visible_joint_ratio_body12`
- Body joints: shoulders, elbows, wrists, hips, knees, ankles
- Visibility threshold: keypoint confidence >= 0.40

## Severity Rule

| Severity | Visible body-joint ratio |
| --- | ---: |
| none | >= 0.90 |
| mild | 0.70-0.89 |
| moderate | 0.50-0.69 |
| severe | 0.20-0.49 |
| full | < 0.20 |

## Draft Results

All sampled frames:

auto_severity
none    86
mild    21
full    13

Detection status:

status
ok                    107
no_person_detected     13

Frames with successful YOLO detection only:

auto_severity
none    86
mild    21

## Interpretation Caveat

`no_person_detected` frames are not automatically treated as confirmed full
occlusion in the manuscript. They should be manually reviewed because they may
reflect small scale, unusual posture, low detector confidence, or true severe
occlusion.

Object/self/mixed source labels are not inferred automatically from confidence
alone and should be manually reviewed using the overlay montage.

## Outputs

- `field_yolo_alphapose_occlusion_severity.csv`
- `field_yolo_alphapose_severity_summary.csv`
- `field_yolo_alphapose_sequence_by_severity.csv`
- `field_yolo_alphapose_severity_overlay_montage.jpg`
