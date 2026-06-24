# E1: 3DPW Correctness

Purpose: address the reviewer request for correctness-based Module 1 validation
using public 3D ground truth.

## Dataset

- Dataset: 3DPW test split
- Ground truth: SMPL-24 3D joints from `jointPositions`
- Images: local 3DPW `imageFiles`
- Use: correctness only, not construction-site occlusion robustness

## Metrics

- MPJPE, mm
- Root-aligned MPJPE, mm
- PA-MPJPE, mm

## Joint Subset

SAM 3D Body predictions are MHR-70 keypoints. 3DPW ground truth is SMPL-24.
Evaluation uses a conservative shared body subset:

`pelvis`, `neck`, shoulders, elbows, wrists, hips, knees, and ankles.

Pelvis for MHR-70 is computed as the midpoint of left and right hips.

## Coordinate Frame

The evaluation uses camera-frame 3D coordinates.

- 3DPW `jointPositions` are transformed with the per-frame `cam_poses`
  extrinsic matrix following the official 3DPW demo convention:
  `X_cam = R @ X + t`.
- SAM 3D Body `pred_keypoints_3d` are body/model-frame keypoints. The evaluator
  adds `pred_cam_t` to obtain camera-frame keypoints before computing metrics.
- If a saved prediction file already contains camera-frame MHR-70 keypoints,
  run the evaluator with `--prediction-coordinates camera`.

## Current Local Status

The evaluation side of the experiment is implemented and has passed an oracle
sanity run.

The oracle sanity result must not be reported as a model result. It only checks
that data loading, joint mapping, and metric computation are wired correctly.

A local Mac CPU smoke test has also succeeded with detector/segmentor disabled:

- input: one 3DPW test frame
- bbox source: 3DPW 2D keypoints
- SAM mode: body-only inference
- output: MHR-70 `pred_keypoints_3d` and `pred_cam_t`
- evaluated samples: 1
- MPJPE: 364.892 mm
- root-aligned MPJPE: 76.228 mm
- PA-MPJPE: 33.654 mm

This single-frame value is only a pipeline smoke test and must not be reported
as an experimental result.

The stride-50 subset experiment has now been completed:

- sampling rule: every 50th frame for each visible subject in the 3DPW test split
- planned samples: 818 person-frame samples
- evaluable samples: 728
- excluded samples: 90
- exclusion reason: invalid 3DPW camera pose annotation
- mean CPU inference time: 25.05 s per successful sample

Aggregate SAM 3D Body correctness:

| Metric | Mean (mm) | SD (mm) | Median (mm) |
| --- | ---: | ---: | ---: |
| Camera-frame MPJPE | 374.726 | 173.294 | 348.296 |
| Root-aligned MPJPE | 90.877 | 18.207 | 87.520 |
| PA-MPJPE | 43.634 | 12.406 | 40.526 |

Root-aligned MPJPE and PA-MPJPE should be interpreted as the primary
correctness metrics because downstream ergonomic analysis depends mainly on
relative body configuration. Camera-frame MPJPE is retained as a diagnostic
measure of global localization/depth error.

## Failure Handling

The 90 excluded samples failed before inference because 3DPW marked the
corresponding camera pose as invalid. These are not detector failures. Replacing
the 3DPW keypoint-derived bounding box with YOLO would not resolve the missing
ground-truth camera transform needed for camera-frame correctness evaluation.

## Baseline Comparison

AlphaPose-MotionBERT should be discussed as a standard baseline, not as a
reliable reference. The response to reviewers should avoid describing it as
ground truth or as a reference under occlusion. The comparison should be framed
as:

- SAM 3D Body versus 2D-to-3D baseline pipelines on the same 3DPW stride-50 subset;
- identical exclusion rule for invalid 3DPW camera poses;
- the same joint subset and metrics where the baseline output permits it;
- reporting failure rate separately from metric averages.

A first MotionBERT baseline has been executed using the 3DPW-provided
COCO/OpenPose-format 2D detections as input. This is not yet a fresh AlphaPose
detection run, so it should be described as `MotionBERT with 3DPW 2D detections`
or as a controlled 2D-to-3D diagnostic, not as `AlphaPose-MotionBERT`.

A second baseline has now been executed with pixel-based AlphaPose inference.
Because the official Halpe-26 checkpoint was temporarily unavailable due to
Google Drive quota limits, this run uses the official AlphaPose COCO-17
FastPose checkpoint. The 3DPW 2D annotations are used only to derive a person
bounding box; AlphaPose predicts the 2D keypoints from the image crop. This is
therefore a bbox-conditioned `AlphaPose-COCO17 -> MotionBERT` baseline rather
than a detector-complete YOLO + AlphaPose pipeline.

| Method | Evaluable samples | Root-aligned MPJPE (mm) | Scale-aligned MPJPE (mm) | PA-MPJPE (mm) |
| --- | ---: | ---: | ---: | ---: |
| SAM 3D Body | 728 | 90.877 ± 18.207 | n/a | 43.634 ± 12.406 |
| MotionBERT with 3DPW 2D detections | 728 | 197.018 ± 85.467 | 163.195 ± 79.267 | 115.861 ± 81.950 |
| Bbox-conditioned AlphaPose-COCO17 -> MotionBERT | 728 | 202.343 ± 55.791 | 178.798 ± 46.563 | 115.133 ± 35.196 |

The MotionBERT output is not a metric camera-frame prediction in the same sense
as SAM 3D Body. Therefore, the fairest comparison is PA-MPJPE, with
scale-aligned MPJPE reported as an additional diagnostic. Camera-frame MPJPE
should not be directly compared for MotionBERT unless a validated metric-scale
camera-frame reconstruction is available.

The bbox-conditioned AlphaPose baseline is more defensible than the
3DPW-provided-2D diagnostic when responding to fairness concerns, because the
2D keypoints are estimated from pixels by AlphaPose. It still should not be
described as the full deployed `YOLO + AlphaPose + MotionBERT` pipeline because
the person detector is controlled by the 3DPW-derived bounding box.
