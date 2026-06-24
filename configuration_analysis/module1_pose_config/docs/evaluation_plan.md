# Module 1 Configuration Analysis Plan

## Purpose

The analysis addresses the configuration contribution question raised by Reviewer
#6 and supports the novelty responses for Reviewer 1-7 and Reviewer 2-1. The
goal is not to treat one pose estimator as ground truth, but to quantify how
changing Module 1 propagates into Module 2 ergonomic features and Module 3 report
content.

## Compared Conditions

Two pose-reconstruction configurations are compared:

1. AlphaPose-MotionBERT
2. SAM-3DB

The same set of field cases should be used in both conditions. Case IDs are
standardized to:

- MansoryBrickLaying_00
- MansoryBrickLaying_01
- MansoryBrickLaying_02
- MansoryCement_02
- RebarPlacement_00
- RebarTying_01
- RebarTying_02
- WallPlacement_00

## Module 2 Downstream Metrics

Only a small representative set of metrics is used to keep the configuration
analysis readable.

1. Shared-region posture risk-bin transition rate
   - Compute the proportion of shared body-region case pairs whose posture risk
     bin changes between the two Module 1 configurations.
   - Report direction as higher-risk shift, lower-risk shift, or unchanged.
   - Use trunk, upper arm, lower arm, and leg only, because AlphaPose-MotionBERT
     lacks the head/hand landmarks needed for full neck and wrist REBA scoring.

2. Static exposure change rate
   - Compare static exposure estimated from the shared trunk/arm/leg angle
     signals.
   - Report the proportion of cases whose static posture presence changes.
   - Also report median absolute difference in static exposure seconds.

3. Repetition change rate
   - Compare repetition count, mean cycle time, frequency band, and quality flag
     when repetition has been re-estimated from configuration-specific coordinate
     sequences.
   - Report the proportion of cases whose frequency band changes.
   - If repetition is not re-estimated separately for both pose configurations,
     report it as unavailable for the Module 1 configuration comparison rather
     than reusing frozen repetition values.

## Module 3 Structured Report Metrics

The same evidence-aware prompt is used for both pose configurations. Missing
evidence is treated as "Insufficient evidence", not as low risk.

1. Risk-summary label change rate
   - Compare posture, duration, and repetition labels generated from the two
     Module 2 outputs.

2. Key-factor Jaccard overlap
   - Compare the set of structured key-risk factors selected under each pose
     configuration.

3. Insufficient evidence rate
   - Report how often each pose configuration cannot support a structured
     judgment because the required anatomical or temporal evidence is absent.

## Module 3 Natural Report Metrics

1. Key-factor overlap
   - Jaccard overlap of key-factor categories extracted from repeated natural
     reports.

2. Recommendation overlap
   - Jaccard overlap of recommendation categories extracted from repeated natural
     reports.

3. Representative changed cases
   - Report a small number of cases where the pose configuration materially
     changes Module 2 evidence and Module 3 interpretation.

The prepared prompt files are:

- `prompts/structured_pose_neutral_contribution.txt`
- `prompts/natural_pose_neutral_contribution.txt`

## Interpretation Rule

The analysis should use neutral comparative language. SAM-3DB should not be
described as ground truth in this configuration analysis. Terms such as
`higher-risk shift` and `lower-risk shift` are preferable to `overestimate` and
`underestimate` unless an external reference label is introduced.

## Current Compatibility Note

The current Module 2 posture runner expects 70-keypoint JSONL-style input for raw
angle extraction and uses 70-keypoint indices for wrist, hand, elbow, ear, and
neck landmarks. The available SAM-3DB field output has 70 keypoints and is
compatible with this design. The available AlphaPose-MotionBERT field output has
`pred_common14`; therefore, a shared-joint adapter or reduced-angle protocol is
required before full posture execution can be considered methodologically fair.

The coordinate-based repetition runner is available under
`<private_workspace>/m2/REP++`. Its input contract is less strict than
the posture runner and can be satisfied with major body joints after JSONL
normalization. A conversion adapter and per-case runner are included in
`scripts/prepare_repetition_jsonl.py` and `scripts/run_repetition_all_cases.py`.
