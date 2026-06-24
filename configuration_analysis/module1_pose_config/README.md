# Module 1 Configuration Analysis

This workspace prepares the Module 1 contribution analysis for the revised manuscript.
The analysis tests how the pose-reconstruction configuration changes downstream
ergonomic features from Module 2 and report outputs from Module 3.

## Configurations

- AlphaPose-MotionBERT: field all-frame `pred_common14` output.
- SAM-3DB: field all-frame MHR-70 keypoint output, stored by sequence.

SAM-3DB provides the 70-keypoint structure expected by the current Module 2
`joint_angle.py` script. AlphaPose-MotionBERT currently provides 14 common joints,
so a shared-joint conversion or a reduced-angle adapter is required before it can
be passed through the same Module 2 runner without unsupported wrist/hand fields.

## Prepared Folders

- `inputs/m1_pose/alphapose_motionbert/`: AlphaPose-MotionBERT pose predictions.
- `inputs/m1_pose/sam3db_mhr70_by_sequence/`: SAM-3DB sequence-level predictions.
- `inputs/m1_pose/manifest/`: field frame/case manifest.
- `inputs/shared_angle_csv/{alphapose_motionbert,sam3db}/`: shared trunk/arm/leg angle CSVs for fair Module 2 comparison.
- `inputs/repetition_jsonl/{alphapose_motionbert,sam3db}/`: pose-condition-specific REP++ JSONL inputs.
- `inputs/module2_outputs/{alphapose_motionbert,sam3db}/`: reserved location for condition-specific Module 2 JSON outputs used by Module 3.
- `module2_runner/ergonomic-risk-module2-main/`: copied Module 2 GitHub runner.
- `scripts/`: helper scripts for compatibility checks, Module 2 execution, and case-ID normalization.
- `prompts/`: evidence-aware neutral prompts for Module 3 downstream report generation.
- `docs/evaluation_plan.md`: metrics and analysis design.

## Current Execution Status

The Module 2 runner was tested with the bundled Python environment. REBA, duration,
ISO-duration, repetition-risk, and schema generation completed successfully. The
final reproduction-evaluation step failed because `sklearn` is not installed in
the available Python environment. This does not block Module 1 configuration
analysis because the required downstream artifacts are the Module 2 schema JSON
files and `schema_summary.csv`, which are generated before that evaluation step.

Use `scripts/run_module2_without_eval.py` to run the downstream generation steps
without the final sklearn-dependent reproduction check.

## Main Commands

Prepare shared-angle inputs, compute shared-duration exposure, and write Module 2
configuration metrics:

```bash
python3 scripts/run_m1_configuration_analysis.py
```

Include repetition re-estimation for both pose configurations before metric
evaluation:

```bash
python3 scripts/run_m1_configuration_analysis.py --include-repetition --cpu
```

Evaluate Module 3 downstream metrics after structured and natural reports have
been generated under `results/generated_reports/`:

```bash
python3 scripts/run_m1_configuration_analysis.py --include-module3
```

The current Module 2-side metric summary is written to
`results/metrics/module1_configuration_metric_summary.json`. Module 3 downstream
metrics are written to
`results/m3_downstream_metrics/module1_module3_downstream_metric_summary.json`.

## Repetition Runner

The coordinate-based repetition pipeline was found under
`<private_workspace>/m2/REP++`. It includes:

- `pipeline_reps.py`
- `pytorch_weights.pth`
- `Newkeypoints/*.jsonl`

The provided `Newkeypoints` JSONL files use snake_case joint names and list
coordinates. `scripts/prepare_repetition_jsonl.py` converts them to the
dictionary-based joint format expected by `pipeline_reps.py`. The converted files
are stored in `inputs/repetition_jsonl/normalized/`.

Use `scripts/run_repetition_all_cases.py --cpu` to run REP++ case by case while
preserving per-case outputs under `results/repetition/`. A smoke test completed
successfully for `RebarTying_01`.
