# Snap-AI: what we did, how, and why

Status as of 2026-11-24. Course: USC CSCI 566 Deep Learning.

---

## 1. The problem

Event photos end up scattered, duplicated and uneven in quality. The goal is to organize them like Google/Apple
Photos: find the faces, group photos by person, let people search "show me all photos of X", and (from the brief)
also handle duplicates, low-quality images and best-shot selection.

Early on we agreed the brief is wider than face clustering. Face recognition is the core deep-learning part;
dedup, quality scoring and a shared album are engineering around it.

## 2. How Google/Apple Photos works (the 3-stage pipeline)

1. **Detect + align:** find each face box and 5 landmarks (eyes, nose, mouth corners), then warp the face to a
   standard 112x112 crop. Alignment removes pose/rotation nuisance before the network sees it.
2. **Embed:** a CNN maps each aligned face to a 512-d vector. Same person -> vectors close in angle; different
   people -> far apart. This is the part that needs deep learning.
3. **Cluster / name:** group vectors (DBSCAN/HDBSCAN/agglomerative), then let the user name or merge clusters.
   For known people, match a new face to stored person vectors with a threshold.

## 3. Decisions along the way and why

| Decision | Why |
|---|---|
| PIPA dataset is not enough on its own | The repo has annotations, not the images; and it is for person recognition in social media, not an embedder training set. |
| Using pretrained ArcFace directly would be thin | It works, but then the project has no deep learning of our own. |
| **Path C: train our own embedder with ArcFace loss** | It is a real DL contribution: implement the loss, train, ablate, evaluate on hard cases. |
| Train on **CASIA-WebFace** | Standard, ~0.49M images / 10.5k identities, available as InsightFace's aligned 112x112 pack with the standard verification sets (LFW, CFP-FP, AgeDB-30). |
| Train on **Colab Pro**, not locally | The user did not want the dataset on their PC. Data lives on Colab's disk; checkpoints on Drive. |
| Keep the teammate's tagger as the baseline/app | Hooman-20/CSCI-566 is a working pipeline on pretrained InsightFace buffalo_l (SCRFD + ArcFace R50, Hungarian matching, 0.45 threshold, synthetic LFW group benchmark, `search.py`). Good baseline; we add what is missing (own training, ablations, hard cases). We did not modify his repo. |

## 4. What we built (all in this folder)

| File | What it does |
|---|---|
| `train/model.py` | IR-ResNet (r18/r34/r50): PReLU, no max-pool, fc over 512x7x7 -> BatchNorm1d 512-d embedding. |
| `train/arcface.py` | ArcFace head written from scratch: logit = s*cos(theta + m), s=64, m=0.5, with the standard fallback when theta+m > pi. Also CosFace (s*(cos - m)) and plain softmax heads, selectable by `--loss`. Computed in fp32 even under AMP. |
| `train/data.py` | Reads `<identity>/*.jpg`; RGB, scaled to [-1,1]; random flip, color jitter, grayscale augmentation. |
| `train/eval_verification.py` | Standard face-verification protocol: InsightFace `.bin` pairs, 10-fold cross-validation, threshold picked on 9 folds and tested on the 10th, flip test-time augmentation. |
| `train/train.py` | Training loop: AMP, SGD (momentum 0.9, wd 5e-4), LR 0.1*batch/512, 1-epoch warmup then cosine, margin warmup 0 -> 0.5 over 2 epochs, resumable (`last.pt`), best checkpoint by mean validation accuracy (`best.pt`), `log.csv`, and `done.txt` on completion. |
| `train/ablate.py` | Runs the ablation grid (one change per run vs a base config) and writes `summary.md`. Skips runs with `done.txt`, auto-resumes interrupted ones, `--force` restarts one. |
| `train/hard_cases.py` | Hard-case evaluation: degrades the probe image of LFW pairs (low-res, blur, mask, sunglasses, JPEG) and also tests pose (CFP-FP) and age (AgeDB-30); ours vs pretrained buffalo_l. **Not yet run.** The ONNX baseline path is untested. |
| `pipeline/embedder.py` | `Embedder(ckpt).embed(aligned_faces)` - drop-in for the app, with flip TTA. |
| `tools/rec_to_folder.py` | Pure-Python MXNet RecordIO -> image-folder converter (no mxnet needed). |
| `notebooks/colab_train.ipynb` | The Colab notebook (GPU check, Drive, data, conversion, trial, full run, ablations, summarize, hard cases). |
| `snap_ai_code.zip` | train/, pipeline/, tools/ packaged for upload to Colab. |
| `train/README.md`, `requirements-train.txt` | Setup and usage. |

### Why these design choices

- **Margin warmup:** ArcFace with m=0.5 from step 0 often does not converge from scratch. The ramp is also why
  the loss *rises* during epochs 1-2 while accuracy rises: the task gets harder each step.
- **Linear-scaled LR + cosine:** standard for large-batch SGD; keeps results comparable across batch sizes.
- **Flip TTA + 10-fold protocol:** matches the published way of reporting LFW-style accuracy, so our numbers
  are comparable to the literature.
- **Ablation runs use a 2,000-identity subset and 8 epochs:** the full 8-run grid fits in a few hours instead
  of days. Conclusions are therefore "at reduced scale".

## 5. Environment

- Local: RTX 4070 Laptop 8 GB, uv venv (Python 3.12, torch 2.14 + CUDA 12.6). Used only for smoke tests on
  synthetic data (loss falls, accuracy ~99%, verification eval and converter run end to end).
  (Early on I pip-installed opencv-python and pillow into the global Python before switching to uv; disclosed.)
- Colab Pro: T4 so far, switching to L4 for the full run. Dataset pack copied from Drive, converted to image
  folders on the VM's local disk; checkpoints written to `MyDrive/snap_ai/runs`.
- Colab facts that shaped the workflow: `/content` is wiped on disconnect; Drive is the only durable place;
  background execution is not guaranteed on Pro, so the laptop must stay awake or you resume later.

## 6. Problems we hit and how we fixed them

| Problem | Cause | Fix |
|---|---|---|
| Local torch was CPU-only | default install | uv venv with the cu126 build |
| Smoke test crashed | scratch dir had been cleaned | regenerated synthetic data |
| `rec_to_folder.py` not found on Colab | code zip was never uploaded; `/content/code` empty | upload the zip with `files.upload()` |
| Upload cell deleted the zip it just uploaded | `rm -rf /content/code` ran while the working dir was that folder | `%cd /content` before uploading |
| Ablation skipped unfinished runs | "finished" was detected by `best.pt`, which exists after epoch 1 | `done.txt` marker + automatic `--resume` + `--force` |
| Colab kept running stale code | Colab saved the new upload as `snap_ai_code (1).zip`; the unzip re-read the old file | unzip the renamed file / take the newest zip by mtime |
| Loss "increases" in early epochs | margin warmup, not a bug | explained; accuracy rises as expected |

## 7. Results so far

### Data sanity
Converted 490,623 images / 10,572 identities; a spot-check grid showed aligned crops with consistent identity.

### r18 trial (2,000 ids, 8 epochs): LFW 96.28%, train acc 91.7%, ~730 img/s.

### Ablation grid (2,000 ids, 8 epochs, batch 128, one seed; best-epoch accuracy in %)

| run | arch | loss | margin | emb | params (M) | LFW | CFP-FP |
|---|---|---|---|---|---|---|---|
| base | r34 | ArcFace | 0.5 | 512 | 34.1 | 96.48 | 81.83 |
| arch_r18 | r18 | ArcFace | 0.5 | 512 | 24.0 | 96.45 | 80.79 |
| arch_r50 | r50 | ArcFace | 0.5 | 512 | 43.6 | 96.50 | 82.20 |
| emb_128 | r34 | ArcFace | 0.5 | 128 | 24.5 | 96.52 | 82.00 |
| loss_cosface | r34 | CosFace | 0.35 | 512 | 34.1 | 96.10 | 80.41 |
| loss_softmax | r34 | softmax | - | 512 | 34.1 | 94.42 | 80.14 |
| margin_0.3 | r34 | ArcFace | 0.3 | 512 | 34.1 | 96.15 | 81.81 |
| margin_0.7 | r34 | ArcFace | 0.7 | 512 | 34.1 | 96.53 | 79.39 |

Noise: standard error about +-0.25 pts on LFW and about +-0.5 on CFP-FP, so gaps under ~0.5 are not meaningful.

### What it shows
1. **The angular margin matters:** softmax is ~2 pts below ArcFace on LFW. The clearest result.
2. **ArcFace vs CosFace:** ArcFace slightly ahead (-0.4 LFW, -1.4 CFP-FP for CosFace), but CosFace's margin was not tuned.
3. **Margin size:** LFW is flat for 0.3-0.7; CFP-FP (pose) drops at 0.7 (79.39). m=0.5 is a safe middle.
4. **Backbone depth:** r18/r34/r50 are indistinguishable on LFW at this data size; data is the bottleneck.
5. **Embedding size:** 128-d matches 512-d (most parameters sit in the final fc layer), so it is free compression.

Caveats: single seed; reduced scale; speed numbers for emb_128, margin_0.3, margin_0.7 (~320 img/s) come from a
slower session after a reconnect and should not be compared. Valid speeds: r18 ~718, r34 ~534, r50 ~450 img/s.

### Full-data run (done): r34, ArcFace m=0.5, 512-d, all 10,572 identities, 490,623 images, 25 epochs, batch 256, T4

| | LFW | CFP-FP | AgeDB-30 |
|---|---|---|---|
| epoch 22 (`best.pt`, best mean 95.63) | 99.27 | 94.16 | 93.45 |
| epoch 25 (final) | 99.30 | 94.29 | 93.23 |
| 2,000-id ablation base (for scale) | 96.48 | 81.83 | - |

- Going from 2k to 10.5k identities gave +2.8 pts LFW and +12 pts CFP-FP: data was the bottleneck.
- LFW saturated by epoch ~9 (99.1-99.3, within noise). CFP-FP and AgeDB kept improving until ~epoch 21, then plateaued.
- Train accuracy ends at 96.7%; validation is flat over the last ~4 epochs, so more epochs would not help much.
- `best.pt` is selected by mean validation accuracy, which picked epoch 22. Differences between epochs 21-25 are noise.
- Speed was ~307 img/s (Colab warned the VM has only 2 CPU cores for 4 workers; data loading was the limit).
- Checkpoints: `MyDrive/snap_ai/runs/r34_casia/{best.pt,last.pt,log.csv,done.txt}`.

## 8. What is left

1. ~~Full-data run~~ done (above).
2. ~~Hard-case evaluation~~ done (below). `calfw.bin`/`cplfw.bin` were not added.

   Ours (r34, `best.pt`) vs pretrained buffalo_l (w600k_r50), accuracy %, probe image degraded (synthetic, on aligned 112x112 crops):

   | condition | ours | pretrained | gap |
   |---|---|---|---|
   | LFW clean | 99.13 | 99.82 | -0.68 |
   | small 56px | 99.07 | 99.82 | -0.75 |
   | small 32px | 98.47 | 99.65 | -1.18 |
   | small 24px | 96.20 | 99.42 | -3.22 |
   | small 16px | 82.23 | 98.17 | -15.93 |
   | blur sigma=2 | 98.38 | 99.70 | -1.32 |
   | blur sigma=4 | 81.63 | 98.90 | -17.27 |
   | mask (lower face) | 94.75 | 99.28 | -4.53 |
   | sunglasses | 96.85 | 99.72 | -2.87 |
   | jpeg q=20 | 99.05 | 99.72 | -0.67 |
   | jpeg q=5 | 81.63 | 98.88 | -17.25 |
   | CFP-FP (pose) | 94.31 | 97.44 | -3.13 |
   | AgeDB-30 (age) | 93.43 | 98.12 | -4.68 |

   Reading: mild degradations cost about the same as the clean gap (~0.7-1.3). Severe low-res, heavy blur and heavy
   compression make our model collapse to ~82% while the pretrained one stays above 98%. Occlusion and pose/age
   are in between. Hypothesis: our training had no blur/low-res/JPEG augmentation and 57x fewer identities.

   **Fix tested:** 5-epoch fine-tune from the finished model with `--degrade-p 0.5` (random low-res 24-112px, blur
   sigma 0.5-3.5, JPEG q10-95; milder than the hardest test cases), LR 0.02, `last.pt` evaluated (not chosen by a clean metric).
   Same pairs, same seed:

   | condition | base | aug | pretrained |
   |---|---|---|---|
   | LFW clean | 99.13 | 99.17 | 99.82 |
   | small 32px | 98.47 | 98.93 | 99.65 |
   | small 24px | 96.20 | 98.17 | 99.42 |
   | small 16px | 82.23 | 91.67 | 98.17 |
   | blur sigma=2 | 98.38 | 99.13 | 99.70 |
   | blur sigma=4 | 81.63 | 98.30 | 98.90 |
   | jpeg q=5 | 81.63 | 95.82 | 98.88 |
   | mask | 94.75 | 94.03 | 99.28 |
   | sunglasses | 96.85 | 96.62 | 99.72 |
   | CFP-FP | 94.31 | 94.03 | 97.44 |
   | AgeDB-30 | 93.43 | 92.92 | 98.12 |

   Reading: blur sigma=4 gap -17.3 -> -0.6; JPEG q=5 -17.3 -> -3.1; 16px -15.9 -> -6.5. Training ranges were milder than
   these test cases, so this is generalisation, not memorisation. Mask, sunglasses, pose and age did not improve
   (differences within about 0.5, i.e. noise) and are the remaining gap; they likely need more identities or occlusion
   augmentation. Clean accuracy on the three benchmarks moved by -0.1 to -0.3, comparable to noise. Caveats: single seed,
   no no-augmentation control (extra epochs alone could contribute a little), synthetic degradations on aligned crops.
   Oddity: blur sigma=4 and JPEG q=5 gave identical 81.63 for base; not yet checked against `degrade_examples` images.
   Evaluation jitter: re-evaluating the same checkpoint gave LFW 99.13 vs 99.27 logged during training, so differences
   of ~0.15 are noise. Pretrained clean LFW 99.82 matches published numbers, which validates the evaluation path.
3. **Group-photo evaluation:** use our embedder inside the teammate's tagger (needs a detect/align/embed split
   of his `tagger.py`) and compare to buffalo_l.
4. **Not built yet:** HDBSCAN clustering for unknown faces, PIPA evaluation script, dedup and quality/best-shot modules.
5. **Midterm draft fixes:** ReLU vs PReLU wording, the max-pool claim, "no retraining" wording, k-means vs
   HDBSCAN, and the threshold-isolation sentence. The intro paragraph has already been rewritten.

## 9. How to describe this honestly in the report

- Detection and alignment use pretrained InsightFace (SCRFD); **our contribution is the embedder**: architecture
  choices, ArcFace/CosFace/softmax heads implemented from scratch, training recipe, ablations, hard-case study.
- Our model is trained on ~0.5M images, versus pretrained buffalo_l on WebFace600K (~600k identities). Expect it
  to be below buffalo_l on absolute accuracy; the value is in the controlled comparisons and the failure analysis.
- Report ablations as reduced-scale, single-seed results with the noise level stated.
