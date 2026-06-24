# Structured Prompt Plan

This document defines the structured-report prompt for the clean Module 3
rebuild.

## Validation vs Configuration

Validation and configuration are separated.

- Validation uses the final selected structured prompt and evaluates the final
  module setting.
- Configuration analysis keeps the structured prompt fixed and compares input
  conditions to explain how the available evidence changes the
  structured-reporting outputs.

Input-condition comparisons belong to configuration analysis, not validation.

## Prompt Set

### Final Structured Report Prompt

File: `prompts/structured/structured_report_prompt.txt`

Purpose:

- Restricts interpretation to explicitly provided evidence.
- Avoids task stereotypes and external assumptions.
- Follows the original evidence-based structured-prompt logic while excluding
  RGB input.
- Uses conservative labels when evidence is weak, missing, or ambiguous.

## Planned Use

Validation:

- Use the final structured report prompt.
- Use Module 2 quantitative input only.
- Run five repeated generations under the validation condition.
- Evaluate risk-summary consistency using Fleiss' kappa.
- Evaluate key-factor labels against the rule-based reference using accuracy,
  precision, recall, and F1-score.
- Build the final validation table with human evaluator results included.

The curated key-factor reference is used for evaluation only. The prompt does
not reproduce the full GT criterion table; it asks the LLM to make an
evidence-grounded structured report from the provided Module 2 summaries.

Configuration:

- Keep the final structured report prompt fixed.
- Compare input-condition effects only.
- Use the same structured metrics for comparison, but report them as
  configuration effects.

For the finalized re-run procedure, see
`structured_report_validation_runbook.md`.
