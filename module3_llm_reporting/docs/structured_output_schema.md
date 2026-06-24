# Structured Report Output Schema

This schema follows the manuscript-level structured report definition.

## Output Object

The structured report contains two sections:

1. `risk_summary`
2. `key_risk_factors`

The output should be generated as valid JSON in the following structure.

```json
{
  "risk_summary": {
    "posture_risk": {
      "label": "High | Low",
      "rationale": "short evidence-grounded rationale"
    },
    "duration_risk": {
      "label": "High | Low",
      "rationale": "short evidence-grounded rationale"
    },
    "repetition_risk": {
      "label": "High | Low",
      "rationale": "short evidence-grounded rationale"
    }
  },
  "key_risk_factors": {
    "trunk_overflexion": {
      "label": "Yes | No",
      "rationale": "short evidence-grounded rationale"
    },
    "neck_overflexion_or_extension": {
      "label": "Yes | No",
      "rationale": "short evidence-grounded rationale"
    },
    "upper_arm_elevation": {
      "label": "Yes | No",
      "rationale": "short evidence-grounded rationale"
    },
    "wrist_deviation": {
      "label": "Yes | No",
      "rationale": "short evidence-grounded rationale"
    },
    "knee_overflexion": {
      "label": "Yes | No",
      "rationale": "short evidence-grounded rationale"
    },
    "prolonged_static_posture": {
      "label": "Yes | No",
      "rationale": "short evidence-grounded rationale"
    },
    "repetitive_work": {
      "label": "Yes | No",
      "rationale": "short evidence-grounded rationale"
    }
  }
}
```

## Validation Use

- `risk_summary` labels are evaluated using inter-run agreement only.
- `key_risk_factors` labels are evaluated against the rule-based reference in
  `key_factor_reference_rules.md`.
- Rationales are retained for grounding checks, but are not used for the first
  structured-report validation pass.

