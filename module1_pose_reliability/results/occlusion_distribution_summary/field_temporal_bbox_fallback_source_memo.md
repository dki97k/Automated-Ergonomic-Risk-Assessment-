# Temporal Bbox Fallback by Occlusion Source

Manual source labels were assigned to the 13 YOLO failure frames reviewed in the montage.

## Source Distribution

- object: 3/13 YOLO failure frames; fallback recovered 3/3 (100.0%).
- self: 10/13 YOLO failure frames; fallback recovered 10/10 (100.0%).

## Overall Recovery

- YOLO-only failure frames: 13
- Temporal bbox fallback recovered: 13/13 (100.0%).

## Manuscript Use

These results can be reported as a component-level ablation: the temporal bbox fallback recovers YOLO detection failures under object- and self-occlusion, allowing SAM-3DB inference to continue. This should be framed as detection-continuity evidence, not direct 3D pose correctness evidence.
