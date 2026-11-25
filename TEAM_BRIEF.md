# Snap-AI: Our Own Face Embedder (team brief)

## 1. What we did, in one paragraph

The pipeline is **detect face -> align to 112x112 -> embed (512 numbers) -> compare / cluster**. We kept the detector
(SCRFD) pretrained and **trained our own embedding network from scratch** with the ArcFace loss on CASIA-WebFace
(0.49M images, 10,572 people). We then ran ablations, compared against the pretrained InsightFace buffalo_l model under
hard conditions, fixed the main weakness, and plugged our model into Hooman's group-photo tagger. Hooman's repo was
not modified.

## 2. The embedder architecture

**Input:** one aligned 112x112 RGB face, pixels scaled to [-1, 1].
**Output:** a 512-d vector, L2-normalized, so similarity between two faces = cosine similarity.

```
face 112x112x3
  -> stem: 3x3 conv (stride 1, 64 ch) + BN + PReLU          (no max-pool, small faces need resolution)
  -> stage 1: 3 residual blocks, 64 ch,  56x56   (first block downsamples 112 -> 56)
  -> stage 2: 4 residual blocks, 128 ch, 28x28
  -> stage 3: 6 residual blocks, 256 ch, 14x14
  -> stage 4: 3 residual blocks, 512 ch, 7x7
  -> BN -> flatten (25,088) -> Dropout 0.4 -> Linear -> BatchNorm1d
  -> 512-d embedding
```

- **Backbone: IR-ResNet-34** ("improved residual" ResNet, the standard for ArcFace). Residual block order is
  BN -> 3x3 conv -> BN -> PReLU -> 3x3 conv -> BN, plus a skip connection. 34.1M parameters. We also built r18 (24.0M)
  and r50 (43.6M) to compare.
- **Training head (only used during training):** a linear layer with one weight vector per person (10,572 of them).
  **ArcFace** adds an angular margin: the logit for the true person is `s * cos(theta + m)` with s=64, m=0.5. This
  forces a gap between different people, so same-person faces cluster tightly and different people stay apart.
  At test time the head is thrown away; only the backbone is used.
- **Why a margin loss:** plain softmax only needs the right class to win. ArcFace needs it to win by a margin, which is
  what makes cosine similarity meaningful for faces never seen in training.

## 3. How we trained it

| Setting | Value |
|---|---|
| Data | CASIA-WebFace, 490,623 images, 10,572 identities (already aligned 112x112) |
| Optimizer | SGD, momentum 0.9, weight decay 5e-4 |
| Learning rate | 0.1 x batch/512 (0.05 at batch 256), 1 epoch warmup then cosine decay |
| Margin warmup | m ramps 0 -> 0.5 over 2 epochs (m=0.5 from step 0 often does not converge) |
| Epochs / batch | 25 epochs, batch 256, mixed precision, grad-clip 5 |
| Augmentation | horizontal flip, colour jitter, 5% grayscale |
| Hardware | Colab Pro T4, about 307 images/s, checkpoints saved to Drive (resumable) |

Evaluation follows the published protocol: LFW (6,000 pairs), CFP-FP (7,000, frontal vs profile) and AgeDB-30
(6,000, 30-year age gap), 10-fold cross-validation with the threshold chosen on 9 folds, plus flip test-time
augmentation.

## 4. Results

### 4.1 Ablations (2,000-identity subset, 8 epochs, one seed, so conclusions are "at reduced scale")

| run | LFW | CFP-FP |
|---|---|---|
| base (r34, ArcFace m=0.5, 512-d) | 96.48 | 81.83 |
| r18 | 96.45 | 80.79 |
| r50 | 96.50 | 82.20 |
| 128-d embedding | 96.52 | 82.00 |
| CosFace (m=0.35) | 96.10 | 80.41 |
| plain softmax | 94.42 | 80.14 |
| margin 0.3 | 96.15 | 81.81 |
| margin 0.7 | 96.53 | 79.39 |

Noise is about +-0.25 on LFW and +-0.5 on CFP-FP, so gaps under 0.5 do not mean anything.

1. **The angular margin matters:** softmax is about 2 points below ArcFace. This is the clearest result.
2. **Depth does not matter at this data size** (r18 = r34 = r50 on LFW): data is the bottleneck.
3. **Margin 0.7 hurts pose** (CFP-FP 79.4), 0.5 is a safe middle.
4. **128-d is as good as 512-d** and saves 9.6M parameters, so it is free compression.

### 4.2 Full run (all 10,572 identities, 25 epochs)

| | LFW | CFP-FP | AgeDB-30 |
|---|---|---|---|
| ours, final epoch | 99.30 | 94.29 | 93.23 |
| pretrained buffalo_l | 99.82 | 97.44 | 98.12 |

More identities (2k -> 10.5k) gave +2.8 on LFW and +12 on CFP-FP. The pretrained model is the same kind of network
trained on about 57x more identities, which is the likely source of the remaining gap.

### 4.3 Hard cases: the problem we found and fixed

We degraded the second photo of each LFW pair (the "bad photo in a group shot") and compared accuracy:

| condition | ours before | ours after fix | pretrained |
|---|---|---|---|
| clean | 99.13 | 99.17 | 99.82 |
| small face 24 px | 96.20 | 98.17 | 99.42 |
| small face 16 px | 82.23 | 91.67 | 98.17 |
| heavy blur (sigma 4) | 81.63 | 98.30 | 98.90 |
| heavy JPEG (q=5) | 81.63 | 95.82 | 98.88 |
| mask (lower face) | 94.75 | 94.03 | 99.28 |
| sunglasses | 96.85 | 96.62 | 99.72 |
| CFP-FP (pose) | 94.31 | 94.03 | 97.44 |
| AgeDB-30 (age) | 93.43 | 92.92 | 98.12 |

- **Finding:** our model collapsed to about 82% under heavy blur, heavy compression and tiny faces, while the
  pretrained model stayed above 98%.
- **Cause:** we trained with no quality augmentation, so the network never saw degraded faces.
- **Fix:** a 5-epoch fine-tune with random low-res, blur and JPEG applied to half the training images (ranges milder
  than the test cases). Blur sigma 4 went 81.6 -> 98.3, JPEG q=5 81.6 -> 95.8, 16 px 82.2 -> 91.7, with clean
  accuracy unchanged within noise.
- **Not fixed:** masks, sunglasses, pose and age gaps stayed the same. The augmentation did not target them, and they
  likely need more identities.

### 4.4 In the real pipeline: group photos (Hooman's tagger flow)

300 synthetic LFW group photos, 111 enrolled people, 1,525 faces. Same detector, alignment and Hungarian matching for
all three models; only the embedder changes.

| model | best threshold | precision | recall | F1 | exact-photo match |
|---|---|---|---|---|---|
| ours, before fix | 0.45 | 0.991 | 0.986 | 0.989 | 92.3% |
| ours, after fix | 0.45 | 0.990 | 0.991 | 0.990 | 93.3% |
| pretrained buffalo_l | 0.40 | 0.999 | 0.998 | 0.999 | 99.0% |

Our embedder reaches F1 0.99 in the real pipeline, about 1 point behind pretrained, and uses the same 0.45 threshold
Hooman's tagger already uses. "Exact-photo match" is stricter: one wrong face spoils the whole photo, which is why it
drops to about 93%.

## 5. Caveats to be honest about

- Single seed everywhere; ablations are on a 2,000-identity subset at 8 epochs.
- Hard cases are synthetic degradations on aligned crops, not real low-quality photos.
- No no-augmentation control run, so a small part of the fix could be from extra epochs (unlikely to explain 81 -> 98).
- Group-photo thresholds were picked on the same photos they are scored on (equally for all three models).
- The group-photo benchmark is synthetic LFW tiles, not real event photos.

## 6. Where things are

- Code: `train/` (model, ArcFace/CosFace heads, training, evaluation, ablations, hard cases), `pipeline/embedder.py`
  (drop-in embedder), `pipeline/group_eval.py` (our model inside Hooman's tagger flow).
- Weights: `models/base_slim.pt`, `models/aug_slim.pt` (about 140 MB each).
- Results: `results/` (tables, training logs, degradation pictures).

## 7. What is left

1. Cluster unknown faces with HDBSCAN (right now unknown faces are only labeled "unknown").
2. Test on real photos (PIPA or our own event photos), not just synthetic tiles.
3. Duplicate, low-quality and best-shot modules from the original brief.
4. Optional: no-augmentation control run, more seeds, occlusion augmentation (masks, sunglasses).
