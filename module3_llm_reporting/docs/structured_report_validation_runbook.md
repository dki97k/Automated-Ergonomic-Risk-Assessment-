# M3 Structured Report Validation Runbook

This runbook records the finalized structured-report validation workflow for
Module 3.

## Final Validation Setting

- Task: structured report generation from Module 2 quantitative summaries.
- Prompt: `prompts/structured/evidence_grounded_structured_report_prompt.txt`.
- Input condition: current remapped Module 2 numerical summaries only
  (`data/structured_validation/inputs/numerical_only_m2_current_remapped`).
- RGB input: not used.
- Repeated runs: five generations per case.
- Human comparison: included in the final validation table.

The prompt is fixed for validation. Prompt/input comparisons belong to
configuration analysis, not validation.

## Reviewer-Safe Prompt Constraints

The finalized prompt is designed to address reviewer concerns by:

- using only Module 2 quantitative evidence;
- excluding RGB and visual-context interpretation;
- prohibiting task stereotypes, external assumptions, and latent-risk
  inference;
- providing field mappings only, not decision thresholds or the GT criterion
  table;
- separating key-risk-factor presence from overall High/Low risk-summary
  labels.

## Data Inputs

When revised Module 2 outputs are available, pass the processed Module 2 JSON
folder to `prepare_structured_validation.py` using `--processed-dir`.

```text
/path/to/module2/processed
```

Human evaluation and rule-based key-factor GT are read from the original Module
3 folder:

```text
<private_workspace>/m3_origin/results_structured_key.csv
<private_workspace>/m3_origin/results_structured_level.csv
```

The key-factor GT is imported into the clean Module 3 workspace by
`import_structured_gt_from_origin.py`. It is used for evaluation only and is not
inserted into the prompt.

## Re-Run Commands

Run these commands from the project root:

```bash
cd /Users/danna/Desktop/MSD

$PY m3/src/prepare_structured_validation.py

$PY m3/src/import_structured_gt_from_origin.py

$PY m3/src/run_structured_validation.py --runs 5 --force

$PY m3/src/evaluate_structured_validation.py

$PY m3/src/build_structured_validation_table.py
```

If the revised Module 2 outputs are stored outside the default processed folder,
replace the first command with:

```bash
$PY m3/src/prepare_structured_validation.py \
  --processed-dir /path/to/revised/module2/processed
```

## Outputs

LLM structured reports:

```text
m3/results/structured_validation/p6_user_prompt_knee_definition_m2_current_remapped/reports/
```

LLM-only validation metrics:

```text
m3/results/structured_validation/p6_user_prompt_knee_definition_m2_current_remapped/evaluation_gt_results_structured_key/metrics.json
```

Final validation table including human evaluators:

```text
m3/results/structured_validation/p6_user_prompt_knee_definition_m2_current_remapped/validation_table_gt_results_structured_key/validation_table.csv
```

The final table contains:

- risk-summary Fleiss' kappa;
- key-factor accuracy;
- key-factor precision;
- key-factor recall;
- key-factor F1-score;
- key-factor sample count.

## Current Final Result

The finalized reviewer-safe prompt produced the following validation result on
the released Module 2 trial outputs:

```text
evaluator,risk_summary_fleiss_kappa,key_factor_accuracy,key_factor_precision,key_factor_recall,key_factor_f1,key_factor_n
Human evaluators,0.6168,0.5821,0.9638,0.5429,0.6945,280
LLM Module 3,0.9641,0.7893,0.9266,0.8245,0.8726,280
```

These values should be regenerated after new Module 2 outputs are inserted.
