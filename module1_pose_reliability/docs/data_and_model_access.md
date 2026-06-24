# Data and Model Access

## 3DPW

3DPW is used as the primary public correctness benchmark because:

- it provides RGB frames, 2D keypoints, 24 SMPL 3D joints, and camera parameters;
- it contains in-the-wild video captured using a moving phone camera; and
- SAM 3D Body reports official 3DPW MPJPE results.

The dataset is licensed for non-commercial scientific research. Downloading the
dataset constitutes acceptance of its license, and redistribution is prohibited.
The dataset must therefore remain outside the public Git repository.

Official page:
https://virtualhumans.mpi-inf.mpg.de/3DPW/

Local expected layout:

```text
../data/3dpw/extracted/
├── imageFiles/
└── sequenceFiles/
    ├── train/
    ├── validation/
    └── test/
```

Current local status:

- `sequenceFiles.zip`, `imageFiles.zip`, and `readme_and_demo.zip` have been
  downloaded under `<private_workspace>/data/3dpw/downloads`.
- The extracted 3DPW data are available under
  `<private_workspace>/data/3dpw/extracted`.
- The extracted image folders use sequential image names such as
  `image_00000.jpg`. The `img_frame_ids` annotation field records original
  source-video frame ids and should not be used directly as the local image
  filename.
- macOS AppleDouble files such as `._*.pkl` and `._*.jpg` may appear on the
  external drive and are ignored by the project loader.

## SAM 3D Body

The official source repository is vendored locally for development but excluded
from this repository:

```text
vendor/sam-3d-body
```

Pinned upstream revision:

```text
b5c765a0d89d789985e186d396315e7590887b94
```

The DINOv3 checkpoint requires Hugging Face login, acceptance of the SAM
license, and consent to share contact information with Meta:

https://huggingface.co/facebook/sam-3d-body-dinov3

Expected checkpoint layout:

```text
checkpoints/sam-3d-body-dinov3/
├── model.ckpt
└── assets/
    └── mhr_model.pt
```

Current local status:

- Hugging Face authentication has been completed locally with a read token.
- `model.ckpt` and `assets/mhr_model.pt` are available under
  `<private_workspace>/m1/checkpoints/sam-3d-body-dinov3`.
- Checkpoints are intentionally ignored by Git and should not be uploaded to a
  public repository.

## Compute note

The official inference implementation primarily targets CUDA. Although parts of
the demo expose a CPU fallback, the estimator currently contains CUDA-specific
tensor movement. Apple Silicon execution therefore requires compatibility
patches and may be prohibitively slow for full 3DPW evaluation. Correctness
experiments should ultimately be run on a CUDA GPU and the exact hardware and
software versions recorded.

Local development will therefore separate:

1. dataset loading, occlusion generation, metric computation, and result
   aggregation, which can be developed on this Mac; and
2. SAM 3D Body inference, which should be run either on a CUDA machine or after
   a documented local compatibility patch.
