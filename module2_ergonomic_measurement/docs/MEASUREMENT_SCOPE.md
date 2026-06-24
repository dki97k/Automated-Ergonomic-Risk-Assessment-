# Measurement Scope

Module 2 is the ergonomic measurement layer. It produces deterministic posture,
duration, and repetition features that are passed to Module 3 for structured LLM
reporting.

## Included in Module 2

| Component | Released Output |
|---|---|
| Joint angles | Mean, p90, and maximum summaries for the released angle fields |
| Posture | REBA final and body-part score summaries |
| Duration | Static-posture segments, total duration, maximum segment, and event count |
| Repetition | Repetition count, mean period, period variation, and rate |
| Evaluation | Posture agreement, repetition consistency, knee diagnostic, and contract checks |
| LLM input | Measurement-only JSON contracts in `llm_input/` |

## Handled by Module 3

Module 3 performs the interpretive reporting step. The following are therefore
not produced as Module 2 outputs:

- integrated risk labels
- High/Low risk summaries
- key-factor Yes/No conclusions
- co-occurrence risk labels
- final natural-language recommendations

This separation keeps Module 2 reproducible as a measurement module and keeps
the final interpretation aligned with the structured-reporting evaluation.

## Public Release Boundary

The release includes scripts, tabular intermediate outputs, prompts, evaluation
tables, and documentation needed to reproduce the reported results. It excludes
raw construction-site images/videos, model checkpoints, raw pose arrays, and
private workspace paths.
