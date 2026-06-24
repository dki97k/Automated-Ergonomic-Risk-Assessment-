# Data Availability

This public release does not include private raw construction-site videos or image frames.

## Included

- Source code for the released analysis modules.
- Public-dataset preparation and evaluation scripts where applicable.
- Derived, anonymized CSV/JSON measurement summaries.
- Consensus and silver-GT tabular annotations used for measurement validation.
- LLM prompts, schemas, validation scripts, and final evaluation summaries.
- Configuration-analysis summaries used to support the revised manuscript.

## Excluded

- Raw construction-site RGB videos.
- Extracted construction-site image frames or montage images.
- Face-visible or otherwise identifiable worker imagery.
- Model checkpoints and downloaded third-party model weights.
- Large intermediate pose prediction dumps such as `.npy` and `.npz` files.
- Local virtual environments, caches, and macOS metadata files.

## External Data and Models

Module 1 public correctness evaluation uses public 3DPW data and external pose-estimation dependencies. These are not redistributed here. Follow the upstream dataset/model licenses and access instructions documented under:

```text
module1_pose_reliability/docs/data_and_model_access.md
```

## Private Dataset Access

The private construction-site videos used in the study contain potentially identifiable people and are not publicly distributed. Access may be considered only through the corresponding author under appropriate ethical, privacy, and data-protection agreements.
