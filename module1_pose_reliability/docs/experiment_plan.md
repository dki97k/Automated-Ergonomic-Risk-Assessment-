# Module 1 Experiment Plan

## E1. Public-dataset correctness

- Evaluate SAM-3DB against public 3D joint ground truth using 3DPW.
- Report MPJPE, PA-MPJPE, and 3DPCK.
- Treat BLC and JAF only as plausibility metrics.
- This experiment supports correctness claims, but not construction-site
  occlusion robustness claims.

## E2. Field-data synthetic occlusion robustness

- Apply synthetic occlusion to the private construction-site images, not to
  3DPW, because the reviewer concern is about field occlusion in the target
  domain.
- Define severity using the visible joint ratio after synthetic occlusion.
- Evaluate none, mild, moderate, and severe conditions.
- Include body-part-targeted and temporally persistent occlusion that mimics
  materials, tools, equipment, and partial worker blocking.
- Since private field images do not provide 3D ground truth, report reliability,
  failure, and downstream ergonomic changes rather than absolute 3D accuracy.

## E3. Failure analysis

- Report detection or reconstruction failures.
- Analyze pelvis/trunk occlusion, long occlusion, and near-full-body occlusion.
- Preserve anonymized or public-data-only visual examples.

## E4. Downstream effect

- Standard: pass SAM-3DB outputs downstream without occlusion information.
- Proposed: condition downstream interpretation on occlusion severity or a validated reliability score.
- Compare erroneous definitive judgments, retained coverage, and failure detection.
- This is the novelty claim for Module 1: not "SAM-3DB itself handles
  occlusion," but "occlusion severity changes how the downstream ergonomic
  output is interpreted and reported."

## E5. Reproducibility

- Record checkpoints, thresholds, ROI rules, joint mappings, coordinate alignment,
  failed-frame handling, software versions, hardware, seeds, and mask generation.
- Publish code and public-data reproduction instructions without private site videos.
