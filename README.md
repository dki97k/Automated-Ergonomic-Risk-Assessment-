# Automated Ergonomic Risk Assessment from Construction-Site Video

This repository contains the public release package for the study:

**From Construction Site Video to Ergonomic Risk Reports: An Automated Framework for Pose-Based Risk Assessment and Interpretation**

The project implements a three-module pipeline for automated work-related musculoskeletal disorder risk screening from single-view RGB construction-site videos:

1. **Module 1: Pose reliability and occlusion analysis**
   - 3D pose inference and plausibility analysis under construction-site visibility constraints.
   - Public 3DPW correctness evaluation and field occlusion/plausibility summary outputs.

2. **Module 2: Ergonomic measurement**
   - Pose-derived joint angles, partial REBA-compatible posture scores, static loading duration, and repetition measurements.
   - This module outputs measurement and symbolic ergonomic features only. Risk interpretation is handled by Module 3.

3. **Module 3: LLM-based reporting**
   - Evidence-grounded structured reports and natural-language ergonomic risk reports generated from Module 2 quantitative summaries.

The release excludes private raw construction-site videos and image frames. Public-data scripts, anonymized/derived tabular outputs, prompts, evaluation scripts, and final result summaries are included.

## Repository Layout

```text
.
├── docs/
│   ├── DATA_AVAILABILITY.md
│   └── RELEASE_CONTENTS.md
├── examples/
│   ├── public_inputs/
│   └── anonymized_outputs/
├── module1_pose_reliability/
├── module2_ergonomic_measurement/
├── module3_llm_reporting/
├── configuration_analysis/
└── tests/
```

## Quick Start

Module 2 is the most self-contained reproduction target because it uses released angle CSVs and derived measurement inputs:

```bash
cd module2_ergonomic_measurement
python -m pip install -r requirements.txt
python verify_measurement_release.py
```

Expected high-level checks include:

- REBA measurement agreement against consensus GT: ICC approximately 0.757.
- Repetition consistency: pooled score approximately 4.167/5.
- Measurement-contract JSONs contain no risk interpretation fields.

Module 1 scripts require external public datasets and model dependencies. See `module1_pose_reliability/docs/data_and_model_access.md`.

Module 3 scripts require LLM API access for report generation, but the prompts, schemas, validation scripts, and final evaluation summaries are included.

## Data Policy

Private construction-site raw videos, frame images, face-visible examples, model checkpoints, and large intermediate pose dumps are not included. Derived CSV/JSON summaries are included when they do not expose raw identifiable imagery.

See `docs/DATA_AVAILABILITY.md` for details.

## Main Results Included

- Public 3DPW pose correctness summaries.
- Field occlusion and anatomical plausibility summaries.
- Module 2 posture, duration, and repetition measurement outputs.
- Module 2 consensus/silver GT evaluation tables.
- Module 3 structured and natural-language validation scripts and summaries.
- Configuration-based contribution analysis summaries for Modules 1, 2, and 3.

## Notes for Reviewers

- Start with `module2_ergonomic_measurement/REVIEWER_README.md` for the one-command measurement reproduction workflow.
- The release is intentionally smaller and cleaner than the internal research workspace.
- Paths in older scripts may reflect the original research workspace and may need adjustment if running outside this repository.
