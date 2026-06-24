# Reviewed Representative No-Person Failure Cases

A representative set of 48 YOLO no-person detection cases was sampled from the all-frame field analysis using up to six cases per sequence. Each case was manually labeled by occlusion source.

## Source Summary

source_label  n  percent
        self 38    79.17
      object  6    12.50
       mixed  4     8.33

## Interpretation

The reviewed no-person detection cases were dominated by self-occlusion/posture-related failures, with additional object and mixed occlusion cases. These manually reviewed cases should be used as representative failure examples, while the full no-person count remains an automatic detector-failure candidate set.

## Outputs

- `field_allframes_no_person_failure_reviewed_representative_cases.csv`
- `field_allframes_no_person_failure_reviewed_source_summary.csv`
- `field_allframes_no_person_failure_reviewed_source_by_sequence.csv`
