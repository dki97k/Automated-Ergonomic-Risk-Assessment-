# M3 Natural-Language Report Validation

This runbook records the clean Module 3 natural-language validation workflow.

## Validation Condition

- Prompt: `prompts/natural/p1_evidence_grounded_advanced.txt`.
- Input: current remapped Module 2 numerical summaries only
  (`data/structured_validation/inputs/numerical_only_m2_current_remapped`).
- RGB image input is excluded from validation.
- Sample identifiers are omitted from the prompt payload so task labels are not
  used as evidence.
- The canonical natural-language report prompt is stored directly in
  `m3/prompts/natural/`; generated `prompt_payloads/` under the results
  directory are audit artifacts, not separate validation prompts.
- Generated reports are saved directly from the model response. No automatic
  content filtering, retry prompt, or repair step is applied; report quality is
  controlled through the prompt itself.
- Prompt/input comparisons are configuration analyses, not validation.

## Metrics

Natural-language validation uses:

- supported claim rate
- unsupported claim rate
- contradiction rate
- key-factor overlap Jaccard
- recommendation overlap Jaccard

Claim-support metrics require claim-level labels in
`claim_annotation_sheet.csv`. Overlap metrics are computed as a deterministic
preview from report text, with extracted tags saved for audit and correction.

## Commands

From `/Users/danna/Desktop/MSD`, set the bundled Python path:

```bash
PY=/path/to/python
```

If revised Module 2 processed JSON files are available, first regenerate the
numerical validation inputs:

```bash
$PY m3/src/prepare_structured_validation.py --processed-dir /path/to/revised/module2/processed
```

Then generate reports and score them:

```bash
$PY m3/src/run_natural_validation.py --runs 5 --force
$PY m3/src/run_natural_scoring_workflow.py
```

The scoring workflow performs these steps:

1. Rebuild `evaluation/claim_annotation_sheet.csv` from generated reports.
2. Back up the unlabeled sheet as
   `evaluation/claim_annotation_sheet_unlabeled_template.csv`.
3. Draft-label claims with `draft_label_natural_claims.py`.
4. Apply the draft labels to `claim_annotation_sheet.csv`.
5. Compute claim-support metrics with `evaluate_natural_claim_labels.py`.
6. Compute key-factor and recommendation overlap with
   `evaluate_natural_overlap.py`.
7. Build `validation_table/validation_table.csv`.

If claim labels are manually reviewed or edited afterward, recompute metrics:

```bash
$PY m3/src/evaluate_natural_claim_labels.py
$PY m3/src/build_natural_validation_table.py
```

Allowed `support_label` values are:

- `supported`
- `unsupported`
- `contradiction`

## Current Validation Run

Final run on the current remapped Module 2 outputs using the bounded
evidence-based three-bullet prompt and no automatic report postprocessing:

- reports: 40 reports (8 samples x 5 runs)
- claim annotation rows: 443
- labeled claims: 443 manually reviewed claim labels
- supported claim rate: 0.8194
- unsupported claim rate: 0.1016
- contradiction rate: 0.0790
- key-factor overlap Jaccard: 0.8625
- recommendation overlap Jaccard: 0.7425

The claim-support metrics reflect the manually reviewed claim labels in
`claim_annotation_sheet.csv`.
