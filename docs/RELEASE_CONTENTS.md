# Release Contents

This folder was created from the internal research workspace at:

```text
<private_workspace>
```

The original workspace was left unchanged. The release folder keeps final code, prompts, derived tabular outputs, and summary reports while excluding raw construction-site media and exploratory clutter.

## Module 1: `module1_pose_reliability`

Purpose:

- Public 3DPW correctness evaluation.
- Construction-site occlusion and anatomical plausibility analysis.
- Synthetic occlusion summary analysis.

Included:

- `src/`, `scripts/`, `tests/`, `docs/`
- Public correctness CSV summaries.
- Field occlusion/plausibility CSV and Markdown summaries.
- Synthetic occlusion CSV and Markdown summaries.

Excluded:

- Private construction-site images/videos.
- Model checkpoints.
- Large intermediate pose predictions.
- Visual montage images.

## Module 2: `module2_ergonomic_measurement`

Purpose:

- Convert joint-angle trajectories into ergonomic measurement features.
- Produce posture, static-duration, and repetition measurement outputs.
- Generate risk-free LLM input contracts for Module 3.

Included:

- Measurement scripts and one-command runner.
- Released angle CSV inputs.
- Pose, duration, and repetition measurement outputs.
- Measurement evaluation scripts and tabular GT/evaluation files.
- Reviewer reproduction guide.

## Module 3: `module3_llm_reporting`

Purpose:

- Convert Module 2 quantitative summaries into structured and natural-language ergonomic reports.
- Evaluate report reproducibility, key-factor classification, and claim-level grounding.

Included:

- Prompt files.
- Structured and natural-language validation scripts.
- Output schemas and reference-rule documentation.
- Final structured validation summary CSVs.

Excluded:

- API-generated report batches when they are large or exploratory.
- Private image context.

## Configuration Analysis: `configuration_analysis`

Purpose:

- Compare downstream effects of pose configuration, evidence composition, and Module 3 input/prompt choices.

Included:

- Scripts, prompts, docs, payload summaries where compact.
- Final evaluation summaries and metrics.

Excluded:

- Large intermediate input dumps.
- Generated report batches.
- Pairwise judge prompt dumps.
