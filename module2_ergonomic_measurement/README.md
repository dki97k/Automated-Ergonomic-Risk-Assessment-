# Module 2: Ergonomic Measurement

This module converts 3D joint-coordinate outputs into ergonomic measurement
features for posture, static-duration, and repetition analysis. It is designed
as a measurement layer for the downstream LLM reporting module.

## Quick Start

```bash
pip install -r requirements.txt
python run_measurement.py
python verify_measurement_release.py
```

`run_measurement.py` regenerates the posture, duration, repetition, and
LLM-input measurement outputs. `verify_measurement_release.py` reruns the
release checks and compares the outputs against the expected values reported in
the manuscript.

## Scope

Included:

- joint-angle summaries
- REBA posture scores and body-part score summaries
- static-posture duration segments and summary statistics
- repetition count, period, and rate summaries
- measurement-only JSON inputs for Module 3 reporting
- self-contained evaluation scripts for released measurement outputs

Excluded:

- raw construction-site images or videos
- model checkpoint files and raw pose arrays
- final risk interpretation or integrated risk labels, which are handled by
  Module 3 structured reporting

## Directory Overview

```text
00_joint_angle/    Joint-angle input tables and calculation script
01_pose/           REBA scoring outputs and posture-measurement evaluation
02_duration/       Static-posture detection outputs and evaluation
03_repetition/     Repetition measurement outputs and evaluation
llm_input/         Measurement-only JSON inputs for Module 3
docs/              Measurement scope and performance notes
run_measurement.py One-command measurement runner
```

## Release Verification Targets

The bundled verification script checks the following released targets:

- REBA system-vs-GT: ICC = 0.757, accuracy = 0.459, QWK = 0.601, MAE = 1.561
- repetition consistency: pooled score = 4.167 / 5
- knee high-flexion diagnostic: flip rate = 0.0%
- Module 3 input contracts: 8 measurement-only JSON files with no risk fields

See `docs/MEASUREMENT_SCOPE.md` for the measurement/reporting boundary and
`docs/PERFORMANCE_VS_BASELINE.md` for the performance summary.
