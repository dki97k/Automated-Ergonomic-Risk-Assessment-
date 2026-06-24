# Module 1 Configuration Analysis Summary

This summary reports the pose-configuration contribution analysis used in the manuscript.
The comparison evaluates how changing the pose-estimation configuration from AlphaPose-MotionBERT to SAM-3DB affects downstream ergonomic measurements and Module 3 reports.

## Module 2 Downstream Measurements

- Frame-level shared-region risk-bin disagreement rate: 29.0%.
- Upper-arm disagreement rate: 39.2%.
- Trunk disagreement rate: 31.1%.
- Lower-arm disagreement rate: 23.1%.
- Leg disagreement rate: 22.5%.
- Higher-risk shift rate: 18.6%.
- Lower-risk shift rate: 10.4%.
- Case-region posture-risk transition rate: 9.4%.
- Static posture presence change rate: 12.5%.
- Repetition frequency-band change rate: 37.5%.

## Module 3 Report Effects

- Structured risk-summary label change rate: 0.1917.
- Posture-risk label change rate: 0.0000.
- Duration-risk label change rate: 0.1250.
- Repetition-risk label change rate: 0.4500.
- Structured key-factor label change rate: 0.4143.
- Positive key-factor Jaccard mean: 0.5632.
- Natural-language key-factor Jaccard mean: 0.7188.
- Natural-language recommendation Jaccard mean: 0.4512.

## Interpretation

SAM-3DB provides complete REBA-compatible body-component coverage, whereas AlphaPose-MotionBERT lacks neck and wrist evidence in this workflow. Pose configuration has a stronger effect on frame-level ergonomic evidence, detailed key factors, repetition-related judgments, and narrative recommendations than on aggregated posture-risk labels.
