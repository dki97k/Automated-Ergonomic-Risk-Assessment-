# Module #2 Configuration Contribution Analysis

This workspace prepares the Module #2 information-composition comparison for
the manuscript's configuration-based contribution analysis.

## Purpose

The analysis responds mainly to Reviewer #6's novelty concern, and also helps
address the broader Reviewer 1-7 and Reviewer 2-1 concerns about whether the
framework is more than an assembly of existing components.

The central question is:

> How does Module #3 reporting change when it receives RGB-only evidence,
> REBA-only posture evidence, or the full Module #2 posture-duration-repetition
> evidence?

## Folder Layout

```text
m2/
  inputs/
    rgb/            # source RGB frames
    reba/           # source REBA-only JSON files
    full_module2/   # source full Module #2 JSON files
  payloads/
    rgb_only/       # anonymized prompt payloads + image assets
    reba_only/      # anonymized prompt payloads
    full_module2/   # anonymized prompt payloads
    manifest.json   # case mapping, not for prompts
  prompts/
    structured_neutral_contribution.txt
    natural_neutral_contribution.txt
    pairwise_judge_evidence_aware.txt
  scripts/
    build_condition_payloads.py
    run_configuration_reports.py
    evaluate_configuration_metrics.py
    run_pairwise_judge.py
    analyze_pairwise_judge.py
    label_configuration_claims.py
    run_full_configuration_analysis.py
  results/
    generated_reports/
    evaluation/
    pairwise_judge/
    pairwise_judge_analysis/
  docs/
    evaluation_plan.md
```

## Run Order

After replacing Module #2 outputs, put the updated files in:

```text
inputs/full_module2/   # updated full Module #2 JSON files
inputs/reba/           # updated REBA-only JSON files
inputs/rgb/            # matching representative RGB frames
```

Then run the full workflow:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/run_full_configuration_analysis.py --force-reports
```

The workflow rebuilds payloads, generates Module #3 reports, computes
structured/natural metrics, applies initial evidence-audit claim labels, and
runs the pairwise judge analysis. If reports already exist and only failed or
dry-run outputs should be retried, omit `--force-reports`.

1. Build anonymized payloads:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/build_condition_payloads.py --force
```

The generated payloads use `case_01`, `case_02`, ... so task names do not leak
into the prompt. Source sample names remain available only in
`payloads/manifest.json`.

2. Generate structured and natural Module #3 reports:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/run_configuration_reports.py --report-type both --runs 5 --force
```

Use `--dry-run` to verify prompts and output paths without calling the API.
For `rgb_only`, the script attaches the peak-risk RGB image as a vision input.

3. Compute deterministic configuration metrics:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/evaluate_configuration_metrics.py
```

This writes condition-level tables for structured key-factor agreement,
insufficient-evidence rates, natural key-factor/recommendation overlap, and a
claim annotation sheet for supported/unsupported/contradiction labels.

4. Apply initial evidence-audit labels:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/label_configuration_claims.py
python3 <private_workspace>/configuration_analysis/m2/scripts/evaluate_configuration_metrics.py
```

The generated labels are initial labels. The user should review
`results/evaluation/claim_annotation_sheet.csv`, revise `support_label`,
`evidence_note`, and `reviewer_note` where needed, and then rerun
`evaluate_configuration_metrics.py` without `--force-annotation` to finalize
claim-level metrics.

5. Run the secondary LLM-as-Judge preference analysis:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/run_pairwise_judge.py --pairing same-run
python3 <private_workspace>/configuration_analysis/m2/scripts/analyze_pairwise_judge.py
```

Add `--include-reverse` if you want a stricter order-bias check.

## Prompt Strategy

The structured and natural prompts are neutral evidence-availability prompts.
They are shared across all three conditions and allow `Insufficient evidence`
when evidence required for a risk dimension is absent.

This prevents the comparison from being biased toward the full Module #2
condition by prompt design alone. If full Module #2 performs better, the result
should reflect wider evidence coverage rather than condition-specific prompt
advantages.

## Reviewer Mapping

- What changes: structured label transitions, natural key-factor and
  recommendation overlap against `full_module2`, and pairwise preference rows.
- How much improves: strict/answered key-factor accuracy, F1, supported-claim
  rate after annotation, overlap stability, and judge win rate.
- Where fails: insufficient-evidence rates, missed full-module key factors,
  unsupported/contradictory claims, and low evidence-grounding/usefulness
  judge scores.
