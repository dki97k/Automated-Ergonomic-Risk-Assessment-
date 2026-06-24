# Module 3 Pipeline Plan

## Purpose

Module 3 will be rebuilt as an interpretation and reporting module. It will not
calculate ergonomic risks independently. It will transform Module 2 quantitative
outputs into structured reports and natural language reports.

## Validation

Validation asks whether the final Module 3 setting works reliably as an
evidence-grounded reporting module.

### Structured Reports

Validation setting:

- prompt: evidence-based structured prompt only
- input: Module 2 quantitative evidence summary only
- repeated runs: five independent LLM generations per case

Target outputs:

- posture risk
- duration risk
- repetition risk
- key risk factor flags

Planned metrics:

- agreement across repeated runs
- accuracy
- precision
- recall
- F1-score

For the first structured validation pass, the rule-based reference is defined
only for key risk factors. Risk-summary labels are evaluated using repeated-run
agreement only.

See:

- `structured_output_schema.md`
- `module2_input_summary.md`
- `key_factor_reference_rules.md`
- `structured_prompt_plan.md`

### Natural Language Reports

Target outputs:

- risk interpretation
- key contributing factors
- recommendations

Planned metrics:

- supported claim rate
- unsupported claim rate
- contradiction rate
- reproducibility of key factors using Jaccard similarity
- reproducibility of recommendations using Jaccard similarity

Recommendation alignment may be added after we finalize how recommendations
should be linked to risk factors.

## Configuration Analysis

Configuration analysis asks how design choices change the outputs.

Planned axes:

- structured reports: prompt fixed to the evidence-based structured prompt;
  compare input-condition effects only
- natural language reports: compare input conditions and three prompt
  strategies

The configuration section will reuse many validation metrics, but the purpose
will be comparative: to show how prompt and input choices affect consistency,
grounding, and report quality.
