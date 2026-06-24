# Module 1: Occlusion-Aware Reliability Analysis

This repository contains the reproducible Module 1 experiments for:

1. evaluating 3D pose correctness on public ground-truth datasets;
2. measuring field-data degradation under controlled synthetic occlusion;
3. analyzing occlusion severity and field failure conditions; and
4. propagating occlusion-related reliability to downstream ergonomic interpretation.

Private construction-site videos are not included.

## Directory layout

- `configs/`: dataset, model, and experiment configurations
- `src/m1/`: reusable implementation
- `scripts/`: command-line experiment entry points
- `tests/`: unit and integration tests
- `notebooks/`: exploratory analysis only
- `docs/`: protocols and reproducibility documentation
- `results/`: aggregate metrics, publication figures, and anonymized failure examples
- `outputs/`: generated intermediate artifacts; not tracked by Git
- `checkpoints/`: model weights; not tracked by Git

Public and private datasets live outside this repository at:

```text
<private_workspace>/data
```

## Planned experiment stages

1. Public correctness: reproduce SAM-3DB inference on 3DPW and report 3D
   ground-truth metrics such as MPJPE and PA-MPJPE.
2. Field occlusion: apply controlled synthetic occlusion to the private
   construction-site dataset, because this is the target visual domain.
3. Severity analysis: define visible-joint-ratio bins and report failure rates
   by none, mild, moderate, and severe occlusion.
4. Downstream effect: compare ergonomic interpretation with and without
   occlusion-severity conditioning.
5. Reproducibility: publish code, configs, and public-data instructions while
   excluding private videos, faces, datasets, and model checkpoints.
