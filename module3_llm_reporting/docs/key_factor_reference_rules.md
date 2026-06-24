# Curated Reference for Structured Key Risk Factors

This table defines the reference labels for the seven structured key risk
factors. The current reference is imported from the curated origin files added
for the structured-reporting GT update:

- `<private_workspace>/m3_origin/GT_key_factors_llm.xlsx`
- `<private_workspace>/m3_origin/results_structured_key.csv`

The Excel file provides the source criteria, and the CSV `GT` column provides
the final case-level labels used for validation.

This is not an external ergonomic expert ground truth.

## General Rule Logic

The imported GT labels are treated as the reference labels for the current
structured validation pass. The criteria below document how the corresponding
items were defined in the origin workbook.

## Key Factor Reference Rules

| Key risk factor | Origin item | Reference criterion |
|---|---|---|
| Trunk overflexion | Excessive trunk flexion | `trunk flexion >= 60 deg` |
| Neck overflexion or extension | Neck flexion / extension, bending, twisting | `max(neck flexion, abs(neck bending), abs(neck twisting)) >= 20 deg` |
| Upper arm elevation | Elevated upper arm; extreme shoulder abduction | `upper arm flexion >= 90 deg` or `upper arm abduction >= 90 deg` |
| Wrist deviation | Wrist deviation | `abs(wrist flexion) >= 15 deg` or `abs(wrist twisting) >= 45 deg` |
| Knee overflexion | Deep knee flexion / squatting | `knee angle >= 60 deg` |
| Prolonged static posture | Prolonged static posture | `Final REBA >= 8` maintained for at least `120` consecutive frames |
| Repetitive work | High repetition motion | Detected repetitive activity based on Module 2 repetition results |

## Risk Summary

No rule-based reference is defined for the three risk-summary labels at this
stage:

- posture risk
- duration risk
- repetition risk

These labels will be evaluated using repeated-run agreement, consistent with
the manuscript's structured-reporting validation logic.

## Reference Note

The current key-factor labels are curated reference labels imported from the
origin CSV. They are used to evaluate whether Module 3 reproduces the
structured key-factor labeling standard. They are not a new human expert
ergonomic assessment.
