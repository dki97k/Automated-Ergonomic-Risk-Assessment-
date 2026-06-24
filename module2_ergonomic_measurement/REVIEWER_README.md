# Module 2 Reviewer Guide

This guide provides a cold-start reproduction path for the released ergonomic
measurement module.

## Environment

```bash
python 3.10+
pip install -r requirements.txt
```

The released path uses provided tabular measurement inputs. GPU, PyTorch, raw
images, and raw pose arrays are not required.

## One-Command Reproduction

```bash
python verify_measurement_release.py
```

The command regenerates the released measurement outputs and validates the
expected posture, repetition, knee-diagnostic, and JSON-contract checks. A
successful run prints `PASS` for all checks.

For regeneration without verification:

```bash
python run_measurement.py
```

## Expected Values

- REBA system-vs-GT: ICC = 0.757, accuracy = 0.459, QWK = 0.601, MAE = 1.561
- repetition consistency: pooled score = 4.167 / 5 across 132 peaks
- knee high-flexion diagnostic: flip rate = 0.0%
- Module 3 input contracts: 8 measurement-only JSON files with no risk fields

## Output Structure

```text
00_joint_angle/  angle measurement inputs
01_pose/         posture scoring outputs and evaluation
02_duration/     static-duration outputs and evaluation
03_repetition/   repetition outputs and evaluation
llm_input/       measurement-only JSON files for Module 3
docs/            scope and performance notes
```

## Declared Limitations

- Force/load is not observed, so dose-response analysis is outside the release
  scope.
- Module 2 reports measurement features only. Final risk interpretation is
  evaluated in Module 3 structured reporting.
- Duration and repetition evaluation files are included for transparent
  reproduction of the released measurement workflow.
