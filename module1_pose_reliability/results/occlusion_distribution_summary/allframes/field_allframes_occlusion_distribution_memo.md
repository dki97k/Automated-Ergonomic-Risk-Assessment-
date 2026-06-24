# Field Occlusion Distribution: All Frames

Input CSV: `<private_workspace>/m1/results/occlusion_distribution/field_yolo_alphapose_occlusion_severity_allframes.csv`

## Scope

- Total field frames evaluated: 40,491
- Sequences: 8
- YOLO + AlphaPose success frames: 36,689
- YOLO no-person detection frames: 3,802

## Severity Rule

Severity is estimated from `visible_joint_ratio_body12`, using shoulders, elbows, wrists, hips, knees, and ankles with AlphaPose keypoint confidence >= 0.40.

| Severity | Visible body-joint ratio |
| --- | ---: |
| none | >= 0.90 |
| mild | 0.70-0.89 |
| moderate | 0.50-0.69 |
| severe | 0.20-0.49 |
| full | < 0.20 or YOLO no-person detection |

## Overall Automatic Severity Counts

auto_severity     n  percent
         none 29247    72.23
         mild  5814    14.36
         full  3804     9.39
     moderate  1500     3.70
       severe   126     0.31

## Detection Status Counts

            status     n  percent
                ok 36689    90.61
no_person_detected  3802     9.39

## Successful Detection Only

auto_severity     n  percent
         none 29247    79.72
         mild  5814    15.85
     moderate  1500     4.09
       severe   126     0.34
         full     2     0.01

## Interpretation

The all-frame field analysis shows that most detectable workers fall into none or mild occlusion. Moderate and severe cases are present but less frequent, while full candidates primarily appear as YOLO no-person detection cases. These no-person frames should be manually reviewed before being described as confirmed full occlusion because detector failure can reflect true occlusion, unusual posture, small scale, or detector limitations.

## Outputs

- `field_allframes_status_summary.csv`
- `field_allframes_severity_summary.csv`
- `field_allframes_ok_only_severity_summary.csv`
- `field_allframes_sequence_by_severity.csv`
- `field_allframes_sequence_by_status.csv`
- `field_allframes_no_person_detection_failure_cases.csv`
- `field_allframes_representative_severity_montage.jpg`
- `field_allframes_no_person_failure_review_montage.jpg`
