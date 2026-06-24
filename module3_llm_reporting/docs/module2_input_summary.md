# Module 2 Input Summary for Structured Reporting

Module 3 receives compact measurement summaries rather than full frame-level
time series. The summary object is designed to preserve the evidence needed for
structured ergonomic reporting while keeping the prompt input auditable.

## Source Data

Each processed case contains:

- `meta`: video ID, FPS, total frames, and source metadata
- `angles`: joint-angle estimates summarized into compact statistics
- `reba`: REBA-derived body-part and final score summaries
- `duration`: static-posture segment summaries
- `repetition`: repetition count and period summaries

## Input Summary Object

Each case should be converted into the following compact input object.

```json
{
  "sample_id": "case_id",
  "meta": {
    "fps": 30,
    "total_frames": 0,
    "total_duration_sec": 0.0
  },
  "posture_summary": {
    "final_reba": {
      "mean": 0.0,
      "p90": 0.0,
      "max": 0.0,
      "risk_bin_distribution": {
        "low": 0.0,
        "medium": 0.0,
        "high": 0.0
      },
      "longest_high_risk_run_frames": 0,
      "longest_high_risk_run_sec": 0.0
    },
    "body_part_reba": {
      "neck": {"mean": 0.0, "p90": 0.0, "max": 0.0},
      "trunk": {"mean": 0.0, "p90": 0.0, "max": 0.0},
      "leg": {"mean": 0.0, "p90": 0.0, "max": 0.0},
      "upper_arm": {"mean": 0.0, "p90": 0.0, "max": 0.0},
      "lower_arm": {"mean": 0.0, "p90": 0.0, "max": 0.0},
      "wrist": {"mean": 0.0, "p90": 0.0, "max": 0.0}
    }
  },
  "joint_angle_summary": {
    "trunk_flexion_deg": {"mean": 0.0, "p90": 0.0, "max": 0.0},
    "neck_flexion_deg": {"mean": 0.0, "p90": 0.0, "max": 0.0},
    "neck_bending_deg": {"mean": 0.0, "p90": 0.0, "max": 0.0},
    "neck_twisting_deg": {"mean": 0.0, "p90": 0.0, "max": 0.0},
    "upper_arm_flexion_deg": {"mean": 0.0, "p90": 0.0, "max": 0.0},
    "upper_arm_abduction_deg": {"mean": 0.0, "p90": 0.0, "max": 0.0},
    "wrist_flexion_deg": {"mean": 0.0, "p90": 0.0, "max": 0.0},
    "wrist_twisting_deg": {"mean": 0.0, "p90": 0.0, "max": 0.0},
    "knee_flexion_deg": {"mean": 0.0, "p90": 0.0, "max": 0.0}
  },
  "duration_summary": {
    "static_posture_ratio": 0.0,
    "total_static_duration_sec": 0.0,
    "max_static_segment_sec": 0.0,
    "static_event_count": 0,
    "max_moderate_or_high_risk_exposure_sec": 0.0,
    "max_high_risk_exposure_sec": 0.0
  },
  "repetition_summary": {
    "total_repetitions": 0,
    "repetition_rate_cycle_per_min": 0.0,
    "mean_period_sec": 0.0,
    "std_period_sec": 0.0
  }
}
```

## Notes

- `p90` is included to reduce sensitivity to single-frame outliers.
- `max` is retained because the report examples use maximum values as explicit
  evidence.
- Raw RGB input is excluded from the public structured-reporting workflow.
