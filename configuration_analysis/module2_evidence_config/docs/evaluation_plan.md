# Module #2 Configuration Contribution Analysis

This folder supports the Reviewer #6 contribution analysis for Module #2:
whether the ergonomic evidence supplied to Module #3 changes downstream
structured judgments and natural-language reports.

## Conditions

The same neutral prompts are used for every condition.

1. `rgb_only`: one representative peak-risk RGB frame only.
2. `reba_only`: REBA-derived posture summaries only.
3. `full_module2`: REBA posture summaries plus joint-angle, duration, and
   repetition summaries from Module #2.

Prompt-visible payloads use anonymized case IDs (`case_01`, `case_02`, ...).
Task names are retained only in `payloads/manifest.json` for auditability.

## Core Evaluation Metrics

Use one common metric set across all configurations, then interpret the
condition-wise changes through Reviewer #6's three questions: what changes, how
much it improves, and where it fails.

### Structured Reports

- `Fleiss' kappa`: repeated-generation reproducibility of risk-summary labels.
- `Accuracy / Precision / Recall / F1`: key-risk-factor agreement with the
  rule-based reference used in Module #3 validation.
- `Insufficient evidence rate`: proportion of risk-summary and key-factor
  items labeled `Insufficient evidence`.

For configuration analysis, `Insufficient evidence` is not automatically a
wrong answer. It is a meaningful outcome showing that a limited evidence
configuration cannot support a risk dimension.

### Natural-Language Reports

- `Supported claim rate`: proportion of claims supported by the evidence
  available under the current condition.
- `Unsupported claim rate`: proportion of claims that introduce unsupported
  case-specific information.
- `Contradiction rate`: proportion of claims contradicting available evidence.
- `Key-factor overlap Jaccard`: repeated-generation consistency of identified
  key-factor categories.
- `Recommendation category overlap Jaccard`: repeated-generation consistency of
  recommendation categories.

### LLM-as-Judge Pairwise Preference

LLM-as-Judge is a secondary report-quality assessment, not factual validation.
Use it after the grounding metrics above.

Pairwise comparisons are made within each case:

- `rgb_only` vs `reba_only`
- `rgb_only` vs `full_module2`
- `reba_only` vs `full_module2`

Criteria:

- Clarity
- Coherence
- Relevance
- Usefulness
- Professionalism
- Evidence Grounding
- Overall

Judges should use the existing Module #3 pairwise preference scale:

```text
+2: Report A clearly and substantially better
+1: Report A slightly better
 0: No meaningful difference
-1: Report B slightly better
-2: Report B clearly and substantially better
```

Report:

- criterion-level win rate;
- overall win rate;
- mean preference strength;
- GPT/Claude agreement rate if multiple judges are used.

## Reviewer #6 Mapping

`What changes` is shown by label distributions, insufficient-evidence rates,
claim support patterns, and LLM-as-Judge criterion-level preferences.

`How much it improves` is shown by condition-wise improvements in F1, supported
claim rate, unsupported claim reduction, overlap stability, and overall
preference win rate.

`Where it fails` is shown by high insufficient-evidence rates, missed risk
signals, unsupported claims, contradictions, and weak LLM-as-Judge grounding or
usefulness scores.

Expected interpretation:

- `rgb_only` may describe visible posture but should not infer duration or
  repetition.
- `reba_only` should improve posture-risk reporting but should remain limited
  for prolonged static posture and repetitive work.
- `full_module2` should expand assessable evidence to posture, duration, and
  repetition, improving coverage and cross-factor interpretation while still
  being checked for unsupported claims.

## Implementation Outputs

For a complete rerun after updating Module #2 results:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/run_full_configuration_analysis.py --force-reports
```

This command rebuilds payloads, generates structured and natural reports,
computes deterministic metrics, applies initial evidence-audit claim labels,
recomputes claim-level metrics, and runs the LLM-as-Judge analysis. The initial
claim labels should be reviewed manually in
`results/evaluation/claim_annotation_sheet.csv`. After review, rerun:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/evaluate_configuration_metrics.py
```

Do not use `--force-annotation` after manual review, because that option
regenerates the annotation sheet.

Run generation:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/run_configuration_reports.py --report-type both --runs 5 --force
```

Primary deterministic evaluation:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/evaluate_configuration_metrics.py
```

Main output files:

- `results/evaluation/configuration_summary.csv`: condition-level summary for
  the manuscript table.
- `results/evaluation/structured_key_factor_rows.csv`: factor-level structured
  predictions against the full Module #2-derived reference.
- `results/evaluation/structured_label_changes_vs_full.csv`: what changed from
  limited evidence to full Module #2 evidence.
- `results/evaluation/structured_positive_overlap_vs_full.csv`: positive
  key-factor Jaccard overlap against `full_module2`.
- `results/evaluation/natural_within_condition_overlap.csv`: within-condition
  reproducibility of key-factor and recommendation categories.
- `results/evaluation/natural_overlap_vs_full.csv`: natural-report category
  changes against `full_module2`.
- `results/evaluation/claim_annotation_sheet.csv`: claim-level sheet for
  supported/unsupported/contradiction labeling.

Secondary LLM-as-Judge evaluation:

```bash
python3 <private_workspace>/configuration_analysis/m2/scripts/run_pairwise_judge.py --pairing same-run
python3 <private_workspace>/configuration_analysis/m2/scripts/analyze_pairwise_judge.py
```

Main judge outputs:

- `results/pairwise_judge/pairwise_results_*.jsonl`: raw criterion-level judge
  decisions.
- `results/pairwise_judge_analysis/pairwise_overall_win_rate.csv`: overall
  win/tie/loss rate by condition.
- `results/pairwise_judge_analysis/pairwise_condition_stats.csv`: criterion
  win/tie/loss and mean signed preference strength.
- `results/pairwise_judge_analysis/pairwise_by_condition_pair.csv`: pair-level
  preference strength for `rgb_only` vs `reba_only`, `rgb_only` vs
  `full_module2`, and `reba_only` vs `full_module2`.
