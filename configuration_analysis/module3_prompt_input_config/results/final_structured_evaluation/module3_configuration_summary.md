# Module 3 Configuration Analysis: Final Structured Rerun

Date: 2026-06-24

Structured Module #3 configuration reports were regenerated using the
final RGB-compatible structured prompt. Payloads were rebuilt before the
rerun. Natural-language configuration reports were not regenerated in this
rerun and remain available in the existing Module #3 configuration results.

- Prompt: `<private_workspace>/configuration_analysis/m3_config/prompts/structured_rgb_compatible_final.txt`
- Generated structured reports: `<private_workspace>/configuration_analysis/m3_config/results/generated_reports_final/structured`
- Structured evaluation: `<private_workspace>/configuration_analysis/m3_config/results/structured_evaluation_final`

## Structured Classification Results

| Input condition | Risk-summary kappa | Key-factor accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Module 2 only | 1.0000 | 0.7000 | 0.9215 | 0.7184 | 0.8073 |
| Module 2 + RGB | 0.9590 | 0.7071 | 0.9223 | 0.7265 | 0.8128 |

## Structured Input-Effect Results

| Comparison | Risk-summary label change rate | Key-factor label change rate | Positive key-factor Jaccard overlap |
|---|---:|---:|---:|
| Module 2 only vs Module 2 + RGB | 0.0333 | 0.0857 | 0.7934 |

Risk-summary changes were limited to repetition risk. No label changes were
observed for posture risk or duration risk. Key-factor changes were limited and
mainly appeared in neck, upper-arm, wrist, knee, and repetitive-work labels.

Compared with the previous structured configuration run, the final prompt
reduced the apparent effect of adding RGB context. The risk-summary label change
rate decreased from 0.1583 to 0.0333, the key-factor label change rate decreased
from 0.3286 to 0.0857, and positive key-factor Jaccard overlap increased from
0.5991 to 0.7934.

## Interpretation

The updated structured prompt made the report generation more strongly anchored
to Module #2 quantitative summaries. Adding a representative RGB frame produced
only minor changes in structured report labels, indicating that RGB context did
not substantially override the quantitative ergonomic evidence. The small
improvement in key-factor F1 under Module 2 + RGB suggests that directly visible
context may provide limited supplementary information, but the broad structured
judgments remained primarily driven by Module #2 evidence.
