# Module 2 Configuration Analysis: Final Structured Rerun

Date: 2026-06-24

Structured reports were regenerated using the final Module 2 configuration
prompt, while existing natural-language reports and reviewed claim annotations
were retained.

- Prompt: `<private_workspace>/configuration_analysis/m2_config/prompts/structured_neutral_final.txt`
- Generated reports: `<private_workspace>/configuration_analysis/m2_config/results/generated_reports_final`
- Evaluation: `<private_workspace>/configuration_analysis/m2_config/results/evaluation_final`

## Updated Structured Results

| Condition | Risk-summary kappa | Risk-summary insufficient rate | Key-factor insufficient rate | Key-factor strict accuracy | Key-factor recall | Key-factor F1 |
|---|---:|---:|---:|---:|---:|---:|
| RGB-only | n/a | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| REBA-only | 0.9475 | 0.6667 | 0.3571 | 0.2893 | 0.3191 | 0.4601 |
| Full Module #2 | 0.8611 | 0.0000 | 0.0000 | 0.8036 | 0.9149 | 0.8866 |

## Updated Natural-Language Results

Natural-language reports were not regenerated in this rerun.

| Condition | Supported claim rate | Unsupported claim rate | Contradiction rate | Key-factor overlap | Recommendation overlap |
|---|---:|---:|---:|---:|---:|
| RGB-only | 0.6465 | 0.3535 | 0.0000 | 0.8010 | 0.6165 |
| REBA-only | 0.6257 | 0.3743 | 0.0000 | 0.6975 | 0.6167 |
| Full Module #2 | 0.7957 | 0.2043 | 0.0000 | 0.6833 | 0.5692 |

## Changes Against Full Module #2

| Comparison | Key-factor label change rate | Positive key-factor Jaccard overlap |
|---|---:|---:|
| RGB-only vs Full Module #2 | 1.0000 | 0.0000 |
| REBA-only vs Full Module #2 | 0.6429 | 0.3179 |

## Interpretation

The updated structured prompt made limited-evidence conditions more conservative.
Under RGB-only input, all structured risk-summary and key-factor items were
labeled as Insufficient evidence. Under REBA-only input, posture-related items
remained partly assessable, but duration- and repetition-dependent items were
mostly treated as unavailable.

Full Module #2 remained the strongest structured condition. Compared with the
previous run, its key-factor F1 stayed essentially unchanged, while risk-summary
reproducibility increased. The updated results therefore emphasize the
contribution of full Module #2 evidence more strongly: duration and repetition
summaries are required to support temporal risk judgments, and joint-angle plus
temporal summaries improve key-factor coverage beyond REBA-only posture scores.
