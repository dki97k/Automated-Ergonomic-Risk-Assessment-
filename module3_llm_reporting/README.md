# Module 3: LLM Reporting

This module evaluates structured and natural-language ergonomic reports generated
from Module 2 measurement summaries.

## Included Workflows

- structured-report prompt and validation tables
- natural-language report validation and scoring workflow
- final public prompts for evidence-grounded reporting
- documentation for the Module 2 measurement input contract

## Key Files

```text
prompts/structured/evidence_grounded_structured_report_prompt.txt
prompts/natural/evidence_grounded_natural_report_prompt.txt
structured_key_factor_reference.csv
structured_risk_summary_reference.csv
src/run_natural_scoring_workflow.py
```

## Validation Outputs

- Structured key-factor reference: `structured_key_factor_reference.csv`
- Structured risk-summary reference: `structured_risk_summary_reference.csv`

Module 3 receives compact measurement summaries rather than raw frame-level
image or video data.
