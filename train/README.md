# Path C: train our own face embedder (ArcFace loss)

Trains `IR-ResNet (r18/r34/r50) -> 512-d embedding` with an ArcFace head written from scratch,
on aligned 112x112 faces, and tracks verification accuracy (LFW etc.) every epoch.

## Setup (uv, CUDA build of PyTorch)

```bash
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install numpy opencv-python pillow
```
Check: `uv run python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

## Data

1. Download the InsightFace dataset-zoo pack for CASIA-WebFace (`faces_webface_112x112`:
   `train.rec`, `train.idx`, `lfw.bin`, `cfp_fp.bin`, `agedb_30.bin`).
2. Convert the `.rec` to an image folder (no mxnet needed; spot-check the output):
   `uv run python tools/rec_to_folder.py --rec <pack>/train.rec --out data/casia_112`

## Train

```bash
uv run python -m train.train --data data/casia_112 \
  --val-bins <pack>/lfw.bin <pack>/cfp_fp.bin <pack>/agedb_30.bin \
  --arch r34 --epochs 25 --batch 128 --out runs/r34_casia
```
Resume with `--resume`. Quick experiment: `--arch r18 --max-ids 2000 --epochs 8`.
8 GB GPU (RTX 4070 laptop): r18/r34 at batch 128 with AMP should fit; for r50 try `--batch 64`.

## Experiments for the report

**1. Train our embedder** (above), then compare against pretrained `buffalo_l` (ArcFace R50, WebFace600K).

**2. Ablations**: one change per run vs. base (r34, ArcFace m=0.5, 512-d): backbone r18/r50, loss
softmax/CosFace/ArcFace, margin 0.3/0.5/0.7, embedding size 128/512.
```bash
uv run python -m train.ablate run --data data/casia_112 --val-bins <pack>/lfw.bin <pack>/cfp_fp.bin \
    --epochs 15 --extra --max-ids 5000            # shorter schedule so the full grid fits in days, not weeks
uv run python -m train.ablate summarize           # -> runs/ablation/summary.md
```

**3. Hard cases**: small faces, blur, occlusion (mask, sunglasses), JPEG compression on LFW pairs
(probe image degraded), plus pose (CFP-FP) and age (AgeDB-30), ours vs. pretrained.
```bash
uv run python -m train.hard_cases --ckpt runs/ablation/base/best.pt --degrade-bin <pack>/lfw.bin \
    --extra-bins <pack>/cfp_fp.bin <pack>/agedb_30.bin \
    --onnx ~/.insightface/models/buffalo_l/w600k_r50.onnx    # pretrained baseline (untested path)
```

## Use in the app
`pipeline/embedder.py` -> `Embedder("runs/r34_casia/best.pt").embed(list_of_aligned_112_bgr_faces)`.
Faces must be aligned with the standard 5-point ArcFace template (e.g. `insightface.utils.face_align.norm_crop`).

## Design notes
- Margin warmup (0 -> 0.5 over 2 epochs): ArcFace with m=0.5 from step 0 often fails to converge from scratch.
- LR is scaled linearly from 0.1 @ batch 512, 1 epoch linear warmup, then cosine decay.
- Input convention: RGB, (x-127.5)/127.5. Training and inference must match.

## Status
Run end to end on real data (Colab T4): full r34 run 25 epochs (LFW 99.30, CFP-FP 94.29, AgeDB-30 93.23), 8-run
ablation grid, hard-case study and augmentation fine-tune. See the top-level `README.md` and `results/`.
Colab throughput was ~307 img/s (limited by 2 CPU cores for data loading).
