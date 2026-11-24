# search.py indexing profile: data/lfw_groups/test

500 synthetic group photos (320–940 px, 5.02 faces/photo on average), RTX 5090, onnxruntime-gpu 1.31.1,
`search.py --person data/lfw_groups/gallery/Hamid_Karzai --photos data/lfw_groups/test --no-cache`.
ms/photo is wall time per stage inside the indexing loop (stage timers, `--profile`); GPU calls are
synchronous, so `det_infer` / `embed` include host↔device copies.

## Summary: ms/photo by stage

| stage | baseline (CUDA, per-face) | (a) batched ArcFace | (b) + prefetch decode | (c) + TensorRT FP16 |
|---|---:|---:|---:|---:|
| decode | 1.64 | 1.63 | 0.10 | 0.09 |
| det_pre | 2.14 | 2.09 | 2.30 | 2.10 |
| det_infer | 4.41 | 4.59 | 4.24 | 1.74 |
| det_post | 0.65 | 0.62 | 0.58 | 0.53 |
| align | 1.30 | 1.21 | 1.19 | 1.06 |
| embed | 17.24 | 5.35 | 5.19 | 2.47 |
| overhead | 0.10 | 0.20 | 0.18 | 0.16 |
| **total** | **27.48** | **15.70** | **13.77** | **8.15** |

Throughput (median of 3 separate runs without `--profile`): 36.3 → 65.0 → 71.3 → 118.1 photos/s.
The unmodified original code measured 35.0–35.5 photos/s.

Notes
- ArcFace is padded to a fixed batch of 16. Without padding, batching was *slower* (27.9 ms/photo
  for embed): onnxruntime's CUDA EP re-plans cuDNN convolutions whenever the batch size changes.
- After (c), CPU work (det_pre + det_post + align = 3.7 ms) is ~45% of the time; moving the
  letterbox/normalize onto the prefetch thread or the GPU is the next step.

## Per-run tables

### GPU (CUDA), per-face embedding, no prefetch
`data/lfw_groups/test`, 500 photos

| stage | ms/photo | share | what |
|---|---:|---:|---|
| decode | 1.64 | 6.0% | read file + JPEG decode (with prefetch: time spent waiting for the next image) |
| det_pre | 2.14 | 7.8% | resize + pad to 640x640 + normalize to NCHW blob |
| det_infer | 4.41 | 16.0% | SCRFD-10G session.run |
| det_post | 0.65 | 2.4% | anchor decode + score filter + NMS |
| align | 1.30 | 4.7% | 5-point similarity warp to 112x112, per face |
| embed | 17.24 | 62.7% | ArcFace R50 preprocessing + session.run |
| overhead | 0.10 | 0.4% | Face objects, result bookkeeping, Python loop |
| **total** | **27.48** | 100% | 36.4 photos/s, 5.02 faces/photo |

### GPU (CUDA), batched embedding, no prefetch
`data/lfw_groups/test`, 500 photos

| stage | ms/photo | share | what |
|---|---:|---:|---|
| decode | 1.63 | 10.4% | read file + JPEG decode (with prefetch: time spent waiting for the next image) |
| det_pre | 2.09 | 13.3% | resize + pad to 640x640 + normalize to NCHW blob |
| det_infer | 4.59 | 29.3% | SCRFD-10G session.run |
| det_post | 0.62 | 4.0% | anchor decode + score filter + NMS |
| align | 1.21 | 7.7% | 5-point similarity warp to 112x112, per face |
| embed | 5.35 | 34.1% | ArcFace R50 preprocessing + session.run |
| overhead | 0.20 | 1.2% | Face objects, result bookkeeping, Python loop |
| **total** | **15.70** | 100% | 63.7 photos/s, 5.02 faces/photo |

### GPU (CUDA), batched embedding, prefetch
`data/lfw_groups/test`, 500 photos

| stage | ms/photo | share | what |
|---|---:|---:|---|
| decode | 0.10 | 0.7% | read file + JPEG decode (with prefetch: time spent waiting for the next image) |
| det_pre | 2.30 | 16.7% | resize + pad to 640x640 + normalize to NCHW blob |
| det_infer | 4.24 | 30.8% | SCRFD-10G session.run |
| det_post | 0.58 | 4.2% | anchor decode + score filter + NMS |
| align | 1.19 | 8.6% | 5-point similarity warp to 112x112, per face |
| embed | 5.19 | 37.7% | ArcFace R50 preprocessing + session.run |
| overhead | 0.18 | 1.3% | Face objects, result bookkeeping, Python loop |
| **total** | **13.77** | 100% | 72.6 photos/s, 5.02 faces/photo |

### GPU (TensorRT FP16), batched embedding, prefetch
`data/lfw_groups/test`, 500 photos

| stage | ms/photo | share | what |
|---|---:|---:|---|
| decode | 0.09 | 1.1% | read file + JPEG decode (with prefetch: time spent waiting for the next image) |
| det_pre | 2.10 | 25.7% | resize + pad to 640x640 + normalize to NCHW blob |
| det_infer | 1.74 | 21.3% | SCRFD-10G session.run |
| det_post | 0.53 | 6.5% | anchor decode + score filter + NMS |
| align | 1.06 | 13.1% | 5-point similarity warp to 112x112, per face |
| embed | 2.47 | 30.4% | ArcFace R50 preprocessing + session.run |
| overhead | 0.16 | 1.9% | Face objects, result bookkeeping, Python loop |
| **total** | **8.15** | 100% | 122.7 photos/s, 5.02 faces/photo |
