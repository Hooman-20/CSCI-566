# Presenter's guide: everything to know to present Snap-AI and answer questions

Companion to `PROJECT_EXPLAINED.md` (which is the chronological log). This file is organised by *concept*, with the
"why" behind each choice, the numbers to quote, and likely questions with answers.

Items marked **[verify]** are things I could not confirm from our own runs; say them cautiously.

---

## 0. The 60-second pitch

> Event photos pile up scattered and duplicated. Apps like Google Photos solve this by detecting faces, converting each
> face into a vector with a deep network, and grouping photos by person. The vector-producing network is the heart of it.
> We **trained our own face-embedding network from scratch** with the ArcFace loss on CASIA-WebFace (0.49M images,
> 10.5k identities), ran **controlled ablations** to see which design choices matter, and compared our model against a
> strong pretrained model (InsightFace buffalo_l) under **hard conditions**: tiny faces, blur, compression, masks,
> sunglasses, pose and age. We reach 99.3% on LFW (pretrained: 99.8%), found that our model collapses under severe
> degradation while the pretrained one does not, traced that to missing quality augmentation, and fixed it by fine-tuning
> with degradation augmentation (5 epochs): blur sigma=4 went 81.6 -> 98.3, JPEG q=5 81.6 -> 95.8, 16px 82.2 -> 91.7,
with clean accuracy unchanged within noise. Masks, sunglasses, profile and age gaps did not change, because the
augmentation did not target them: those gaps come from limited data (57x fewer identities than the pretrained model).

Three sentences you should be able to say without notes:
1. **What we built:** the embedding network + loss + training + evaluation, not the detector.
2. **What we found:** margin loss matters (+2 pts), depth doesn't at this data size, data does (+12 pts CFP-FP), and robustness to degradation is the real gap vs pretrained.
3. **What we didn't do:** we did not beat the pretrained model; detection/alignment are pretrained.

---

## 1. The problem and the 3-stage pipeline

Goal (from the brief): organise scattered event photos, group by person, find duplicates and bad photos.
Face recognition is the deep-learning core; dedup, quality scoring and best-shot selection are additional (not built).

How Google/Apple Photos works:

| Stage | What happens | Who does it in our project |
|---|---|---|
| 1. Detect + align | Find face boxes and 5 landmarks (eyes, nose, mouth corners); warp each face to a standard 112x112 crop | Pretrained InsightFace SCRFD (in the teammate's tagger). Not trained by us. |
| 2. Embed | CNN maps the crop to a 512-d vector; same person -> close vectors | **Ours** (`train/`, `pipeline/embedder.py`) |
| 3. Cluster / match | Group vectors (DBSCAN/HDBSCAN) or match to enrolled people with a threshold | Teammate's tagger: Hungarian matching + 0.45 threshold **[as I read his repo]**. HDBSCAN for unknown faces not built. |

**Why alignment matters:** it removes rotation/scale so the network only has to learn identity, not geometry. The embedder
must be fed faces aligned with the same 5-point template as training (InsightFace's `norm_crop`), RGB, scaled to [-1, 1].

**Why we picked "train our own embedder" (Path C):** using a pretrained ArcFace directly is a fine engineering solution but
contains no deep learning of our own. Training the embedder lets us implement the loss, run ablations and analyse failures.

**Three tasks people confuse:**
- *Verification:* are these two faces the same person? (LFW/CFP-FP/AgeDB are verification benchmarks.)
- *Identification:* who is this, among enrolled people? (the app's "tag" mode)
- *Clustering:* group unlabeled faces by person. (the app's "discover" mode)
All three reuse the same embedding; only the decision rule on top differs. Good embedding = good everything.

---

## 2. Embeddings, angles and why we normalise

- The network outputs a 512-d vector per face. We **L2-normalise** it, so every face lies on a unit hypersphere.
- On the sphere, similarity is the **cosine** (equivalently, the angle). For unit vectors, squared L2 distance = 2 - 2cos, so
  thresholding on distance or cosine is the same thing. (Our evaluation uses squared L2 in [0, 4].)
- Goal of training: same identity -> small angle, different identity -> large angle.
- **Flip test-time augmentation (TTA):** at test time we embed the image and its mirror and add the two vectors. Cheap, small gain.

---

## 3. Loss functions (the most likely question area)

### 3.1 Why plain softmax is not enough
Softmax classification over identities teaches the network to *separate the training identities*, but nothing forces
embeddings of one person to be tight or different people to be far apart by a margin. At test time we need that for
**unseen** people (open set). Result in our ablation: softmax **94.42** vs ArcFace **96.48** on LFW.

### 3.2 ArcFace (additive angular margin), `train/arcface.py`
1. Normalise the embedding x and each class weight w_j. Then `cos(theta_j) = x . w_j` is the angle to class j's centre.
2. For the *true* class y, replace theta_y by **theta_y + m** (add a margin of m radians). Others unchanged.
3. Multiply all logits by a **scale s**, then ordinary cross-entropy.

   `logit_y = s * cos(theta_y + m)`,   `logit_j = s * cos(theta_j)`  (j != y)

Intuition: to get the same loss as before, the face must be **m radians closer** to its class centre than other classes. This
squeezes each identity into a tight cone and leaves a gap between identities. The margin lives in *angle* space, so it is a
constant geodesic gap on the sphere (that is the "arc" in ArcFace).

**Hyperparameters and why:**
- **s = 64:** cosines are in [-1, 1], too small for softmax to become confident with thousands of classes; scale stretches the
  logits. 64 is the standard value for large-class face training.
- **m = 0.5 rad (~29 degrees):** the standard value. Our ablation: 0.3 / 0.5 / 0.7 are the same on LFW (96.15 / 96.48 / 96.53);
  CFP-FP (pose) drops at 0.7 (79.39) because a too-large margin is hard to satisfy with little data.
- **Margin warmup (0 -> 0.5 over 2 epochs):** m = 0.5 from step 0 often fails to converge from scratch. This explains the
  **loss going up in epochs 1-2 while accuracy goes up**: the target is getting harder every step. (You were asked about this in practice; know it.)
- **Fallback when theta + m > pi:** cos(theta + m) stops being monotonic there, so the code switches to a linear penalty
  (`cos - m*sin(m)`-style term, constants `th`, `mm` in `set_margin`). Standard ArcFace implementation detail.
- **Head in fp32:** the head runs outside mixed precision because cos/sin near +-1 are numerically fragile in fp16.

### 3.3 CosFace
Subtracts a margin from the **cosine** instead: `logit_y = s * (cos(theta_y) - m)`. We used m = 0.35 (its usual value).
Result: 96.10 / 80.41 vs ArcFace 96.48 / 81.83. Slightly worse but **its margin wasn't tuned**, so don't claim ArcFace "beats" CosFace; say "similar, ArcFace slightly ahead in our setting."

### 3.4 Why not triplet loss / contrastive loss? (common question)
They need careful sampling of triplets/pairs (hard-negative mining) and scale poorly; classification-with-margin losses train
stably, use every sample every step, and are the dominant approach (ArcFace, CosFace). **[general knowledge, not tested by us]**

### 3.5 Reading the training logs
- `loss` stays high (e.g. 2.9 at the end) even at 96.7% train accuracy: the margin and scale make the loss strict. Loss value is not comparable to plain softmax.
- `train_acc` is computed from the *unmargined* cosines (`cos` returned by the head).

---

## 4. The network: IR-ResNet (`train/model.py`)

- **ResNet idea:** each block outputs `F(x) + x`. The skip connection lets gradients flow through deep stacks.
- **IR ("improved residual") block:** BN -> conv3x3 -> BN -> PReLU -> conv3x3 -> BN, then add the shortcut. Batch-norm first (pre-activation style) trains more stably.
- **PReLU** (learned leaky slope) instead of ReLU: the ArcFace reference design. (Your midterm draft said ReLU; fix that.)
- **No max-pool, 3x3 stride-1 stem:** faces are only 112x112 and fine detail matters; downsampling is done by stride-2 convs in the four stages: 112 -> 56 -> 28 -> 14 -> 7.
- **Head of the backbone:** BN -> flatten (512x7x7 = 25,088) -> **Dropout 0.4** -> Linear to 512 -> BatchNorm1d. The final BN stabilises the embedding scale.
- **Variants:** r18 = [2,2,2,2] blocks, r34 = [3,4,6,3], r50 = [3,4,14,3]. Params: 24.0M / 34.1M / 43.6M.
- **Why the fc layer dominates parameters:** 25,088 x 512 = 12.8M weights. That is why a 128-d embedding saves 9.6M params (34.1M -> 24.5M).
- **Fixed input size:** because of that fc over 7x7, the model only accepts 112x112.
- **Why r34 for the full run:** the ablation showed r18/r34/r50 are indistinguishable on LFW at 2k identities, so r34 was a middle choice on cost; **[we did not run r18/r50 on full data]** so we can't say r34 is best there.

---

## 5. Training recipe (`train/train.py`): every knob and why

| Setting | Value | Why |
|---|---|---|
| Optimiser | SGD, momentum 0.9 | Standard for face recognition; generalises better than Adam here **[convention]** |
| Weight decay | 5e-4 | Standard for ArcFace training; regularises the large fc/head |
| LR | 0.1 at batch 512, **scaled linearly** with batch (batch 256 -> 0.05; batch 128 -> 0.025) | Linear scaling rule keeps behaviour similar across batch sizes |
| LR schedule | 1 epoch linear warmup, then **cosine decay to 0** | Warmup avoids early instability; cosine ends with very small steps so the model settles |
| Margin warmup | 2 epochs | See 3.2 |
| Mixed precision (AMP) | on, head in fp32 | ~2x faster, half memory |
| Gradient clipping | norm 5 | Safety net against loss spikes |
| Dropout | 0.4 before fc | Regularisation against overfitting a 10k-identity set |
| Augmentation | horizontal flip, colour jitter (p 0.5), grayscale (p 0.05) | Mild; does **not** include blur/low-res/JPEG, which is exactly the weakness we later found |
| Batch | 128 (ablation), 256 (full) | Fits the GPU |
| Epochs | 8 (ablation), 25 (full) | Ablation shortened to fit a grid; full run stops improving by ~epoch 21 |
| Model selection | `best.pt` = best **mean val accuracy** over LFW/CFP-FP/AgeDB | Avoid picking on a single set |
| Resume | `last.pt` each epoch, `--resume` | Colab disconnects |

---

## 6. Data

- **Training:** CASIA-WebFace via InsightFace's `faces_webface_112x112` pack: **490,623 images, 10,572 identities**, already aligned to 112x112.
  Stored as MXNet `train.rec`; we wrote a pure-Python converter (`tools/rec_to_folder.py`) so no MXNet is needed.
- **Why CASIA:** small enough to train on one Colab GPU in hours, standard in papers, has the standard verification sets packaged with it.
- **Ablation subset:** 2,000 random identities (**91,385 images**) with a fixed seed so every run sees the same data.
- **Validation/test sets (.bin pairs):**
  - **LFW** (6,000 pairs): frontal celebrity photos; nearly saturated (99%+ for good models). Tests general verification.
  - **CFP-FP** (7,000 pairs): frontal vs **profile**; tests pose robustness.
  - **AgeDB-30** (6,000 pairs): same person, ~30 years apart; tests age robustness.
- **Overlap question:** standard practice assumes training identities are disjoint from LFW/CFP/AgeDB. **[I haven't verified this for this particular pack; say "assumed per standard practice".]**

---

## 7. Evaluation protocol (`train/eval_verification.py`)

- Each .bin is a list of image pairs plus "same / different" labels.
- **10-fold cross-validation:** split pairs into 10 folds; for each fold, pick the best distance **threshold on the other 9 folds**, measure accuracy on the held-out fold, average. This avoids tuning the threshold on the test pairs.
- **Noise level:** the standard error is about +-0.25 points on LFW and about +-0.5 on CFP-FP (my estimate for these set sizes). **Differences below ~0.5 points should not be called real.** Re-evaluating the same checkpoint gave LFW 99.13 vs 99.27, so ~0.15 jitter is normal.
- **Single seed:** every ablation row is one run. Be upfront about this.

---

## 8. Ablation study: what, how, what it showed

**Design:** r34 + ArcFace m=0.5 + 512-d is the *base*. Each other run changes exactly one thing. 2,000 identities, 8 epochs, batch 128.

| Run | Change | LFW | CFP-FP | Takeaway |
|---|---|---|---|---|
| base | r34, ArcFace 0.5, 512-d | 96.48 | 81.83 | reference |
| arch_r18 | smaller | 96.45 | 80.79 | same within noise |
| arch_r50 | bigger | 96.50 | 82.20 | same within noise |
| loss_softmax | no margin | 94.42 | 80.14 | **clearly worse (-2.1 LFW)** |
| loss_cosface | CosFace 0.35 | 96.10 | 80.41 | slightly worse, margin untuned |
| margin_0.3 | | 96.15 | 81.81 | within noise |
| margin_0.7 | | 96.53 | 79.39 | **pose drops (-2.4 CFP-FP)** |
| emb_128 | | 96.52 | 82.00 | same; fewer params |

**Conclusions you can defend:**
1. The angular margin is the one choice with a clear effect.
2. Depth doesn't matter at this data size => the bottleneck is data (confirmed: full data gives LFW 99.3, CFP-FP 94.3).
3. Too large a margin hurts the hardest (pose) set.
4. 128-d is as good as 512-d here (free compression).

**Caveats (say them before they're asked):** one seed; reduced scale (2k ids, 8 epochs); the speed column for `emb_128`, `margin_0.3`, `margin_0.7` (~320 img/s) comes from a slower session, so compare speed only for r18/r34/r50 (718/534/450 img/s).

---

## 9. Full-data run

r34, ArcFace, 512-d, all 10,572 identities, 25 epochs, batch 256, T4 (~307 img/s: Colab VM had only 2 CPU cores, so data loading was the bottleneck).

| | LFW | CFP-FP | AgeDB-30 |
|---|---|---|---|
| best.pt (epoch 22) | 99.27 | 94.16 | 93.45 |
| final (epoch 25) | 99.30 | 94.29 | 93.23 |

Reading the curve:
- 2k -> 10.5k identities: **+2.8 LFW, +12 CFP-FP**. Data scale is the lever.
- LFW saturates by epoch ~9; CFP-FP/AgeDB improve until ~epoch 21, then plateau. More epochs would not help.
- Train accuracy 96.7% vs validation plateau: the model fits training identities well; the remaining error is generalisation, not optimisation.
- Epochs 21-25 differ by noise; `best.pt` being epoch 22 is not meaningful.

---

## 10. Hard-case study (our key analysis)

**Design (`train/hard_cases.py`):** For each LFW pair, the *first* image stays clean (the "enrolled" photo) and the *second* (the "probe") is degraded, mimicking "good profile photo vs. bad photo in a group shot". Same pairs, same threshold protocol for every model. Degradations are synthetic and applied to the aligned 112x112 crop:
low-res (56/32/24/16 px, then upsampled), Gaussian blur (sigma 2, 4), mask (lower face filled), sunglasses (eye band), JPEG (q=20, q=5). Pose = CFP-FP, age = AgeDB-30.

| Condition | ours | pretrained | gap |
|---|---|---|---|
| LFW clean | 99.13 | 99.82 | -0.7 |
| 56 px / 32 px / blur 2 / JPEG 20 | 98.4-99.1 | 99.65-99.82 | -0.7 to -1.3 |
| 24 px | 96.20 | 99.42 | -3.2 |
| sunglasses | 96.85 | 99.72 | -2.9 |
| mask | 94.75 | 99.28 | -4.5 |
| CFP-FP (pose) | 94.31 | 97.44 | -3.1 |
| AgeDB-30 (age) | 93.43 | 98.12 | -4.7 |
| **16 px** | 82.23 | 98.17 | **-15.9** |
| **blur sigma 4** | 81.63 | 98.90 | **-17.3** |
| **JPEG q=5** | 81.63 | 98.88 | **-17.3** |

**Reading it:**
- Pretrained clean LFW 99.82 matches published numbers => our evaluation code and the ONNX path are correct.
- Mild degradation costs about the same as the clean gap: no special weakness.
- **Severe** degradation (16 px, sigma 4, q=5): ours collapses to ~82%; pretrained stays >98%. This is the main finding.
- Pose/age/occlusion are intermediate.

**Why the pretrained model is a tough baseline (fairness):** buffalo_l was trained on WebFace600K (~600k identities, millions of images, about 57x more identities than ours) with an R50. Don't present the gap as "our method is worse"; present it as "what a 10k-identity, no-quality-augmentation model can't do".

**Hypothesis for the collapse:** (a) no blur/low-res/JPEG augmentation in our training; (b) far fewer identities and less varied image quality.
**Test of the hypothesis:** fine-tune with quality augmentation (section 11).

**Limitations:** synthetic degradations on aligned crops (real photos blur *before* alignment, and detectors may fail on tiny faces, which we don't measure); mask/sunglasses are crude rectangles; LFW only for degradations.
**Oddity to check:** blur sigma 4 and JPEG q=5 gave exactly 81.63 for ours. Probably coincidence (both destroy high frequencies) but verify with the dumped example images before presenting both numbers.

---

## 11. The fix: quality-augmentation fine-tune (`RandomDegrade` in `train/data.py`)

- Start from the finished model (`r34_casia/last.pt`, backbone **and** ArcFace head).
- With probability 0.5 each training image gets 1-2 of: low-res (24-112 px), blur (sigma 0.5-3.5), JPEG (quality 10-95), random strength.
- **Ranges are deliberately milder than the hardest test cases** (16 px, sigma 4, q=5), so a recovery there means it *generalised* to stronger degradation instead of just memorising test conditions.
- Fine-tune 5 epochs, peak LR 0.01 (lower than the original 0.05, since we are refining), 0.3-epoch warmup, margin already at 0.5 (no warmup).
- Evaluate `last.pt` (final epoch), not `best.pt`, so checkpoint choice isn't driven by a clean-data metric.
- **Result:** [fill in after the run: collapse cases before -> after; clean LFW/CFP-FP/AgeDB before -> after.]
- **Missing control:** no fine-tune without augmentation. The original run had plateaued, so extra epochs alone shouldn't explain a big jump, but a control (`--degrade-p 0`) would settle it.
- Honest expectation: robustness up on degraded cases, possibly a small cost (a few tenths) on clean sets. Report it as a trade-off if so.

---

## 12. Engineering you may be asked about

- **Colab workflow:** code zip uploaded; dataset pack from Drive converted to image folders on the VM's fast local disk; checkpoints and logs written to Drive. `/content` is wiped on disconnect; Drive isn't.
- **Resumability:** `last.pt` saved every epoch (model, head, optimiser, scaler, epoch); `done.txt` written only at the end. The ablation runner skips runs with `done.txt` and resumes the rest. (An earlier version used `best.pt` as the "finished" marker, which would have silently skipped interrupted runs: fixed.)
- **Local vs Colab:** only synthetic smoke tests ran locally (RTX 4070 Laptop 8 GB); all real training ran on Colab.
- **Reproducibility:** fixed seeds, identity subset sampled with a fixed seed, same data and evaluation for every ablation.
- **Integration:** `pipeline/embedder.py` exposes `Embedder(ckpt).embed(aligned_faces)` as a drop-in for the app's InsightFace embedder. We have **not yet** run the full group-photo pipeline with our embedder.

---

## 13. What is NOT done (say it before they ask)

- Group-photo end-to-end evaluation with our embedder inside the teammate's tagger.
- HDBSCAN clustering for unknown faces; PIPA evaluation.
- Duplicate detection, image-quality scoring, best-shot selection from the brief.
- Multiple seeds; the no-augmentation fine-tune control; r18/r50 on full data; CosFace margin tuning.
- Detection robustness (we only study the embedder).

---

## 14. Likely questions and short answers

**Q: Isn't using ArcFace just copying a paper?**
A: We implemented the loss and network from scratch, trained them, and the contribution is the controlled ablations and the failure analysis with a measured fix. The method is standard; the study is ours.

**Q: Why not use the pretrained model?**
A: We do, as the baseline, and it is better. The point was to understand what drives performance and where a smaller-data model breaks.

**Q: Why is your model worse than pretrained?**
A: ~57x fewer identities and no quality augmentation. The gap is small on clean data (-0.7 LFW) and large only on severe degradation, which we traced and then addressed.

**Q: What does the margin do, in one sentence?**
A: It forces each face to be a fixed angle closer to its own identity than to others, which tightens each identity's cluster and makes unseen people easier to separate.

**Q: Why scale s = 64?**
A: Cosines are in [-1, 1]; without a scale, softmax can't become confident across thousands of classes.

**Q: Why does the loss increase early in training?**
A: Margin warmup raises the margin from 0 to 0.5 over two epochs; the task gets harder while accuracy still improves.

**Q: Why train on a 2,000-identity subset for ablations?**
A: To fit 8 runs in a few hours. Caveat: conclusions are at reduced scale; the full run confirmed the data effect.

**Q: How do you know the differences are real and not noise?**
A: We state a noise level (~0.25 LFW, ~0.5 CFP-FP) and only call out differences beyond it: softmax vs ArcFace (2 pts) and margin 0.7 on CFP-FP (2.4 pts). Single seed is a limitation.

**Q: How is the verification threshold chosen?**
A: Per fold, on the other nine folds; accuracy is measured on the held-out fold.

**Q: Why cosine / L2-normalised embeddings?**
A: The loss operates on angles, so the embedding's direction carries identity; length is discarded.

**Q: Why PReLU, no max-pool, dropout 0.4?**
A: Those follow the standard ArcFace IR-ResNet design: PReLU as a learned activation, no max-pool to keep detail at 112 px, dropout before the large fc layer to regularise. We didn't ablate them. **[say "design follows the reference, not ablated"]**

**Q: Is there overfitting?**
A: Train accuracy reaches 96.7% while verification accuracy plateaus; there is a generalisation gap, but verification on unseen identities kept improving until ~epoch 21, so we don't see classic degradation from overfitting.

**Q: Why does CFP-FP matter?**
A: Event photos include non-frontal faces; CFP-FP measures frontal-vs-profile verification.

**Q: Your degradations are fake. How realistic?**
A: They're controlled and repeatable but synthetic and applied after alignment. Real photos degrade before detection/alignment. It is a robustness probe, not a real-world benchmark.

**Q: Would it work in a real event album?**
A: Not shown end to end yet. The embedder plugs into the tagger, but the group-photo evaluation is future work.

**Q: What about privacy / bias?**
A: Face recognition has known demographic and privacy concerns; CASIA/celebrity data skews demographics, and the app stores face embeddings of real people. We didn't evaluate fairness. **[general caveat; we did no bias analysis]**

**Q: Why Colab and not your GPU?**
A: Data stayed off the laptop and Colab gave a bigger GPU; the local GPU (8 GB) was used for smoke tests.

**Q: Could you get to pretrained-level accuracy?**
A: Likely needs far more identities (hundreds of thousands) and longer training; our scaling evidence (2k -> 10.5k ids gave +12 CFP-FP) suggests data is the lever. **[extrapolation, not tested]**

---

## 15. Cheat-sheet of numbers

| Item | Value |
|---|---|
| Train set | 490,623 images, 10,572 identities |
| Ablation subset | 91,385 images, 2,000 identities, 8 epochs, batch 128 |
| Full run | r34, 25 epochs, batch 256, ~307 img/s on T4 |
| Params | r18 24.0M, r34 34.1M, r50 43.6M; r34 with 128-d: 24.5M |
| ArcFace | s=64, m=0.5, margin warmup 2 epochs |
| Optimiser | SGD 0.9, wd 5e-4, peak LR 0.1*batch/512, cosine |
| Ablation LFW | softmax 94.42; ArcFace 96.48; CosFace 96.10 |
| Full-run LFW / CFP-FP / AgeDB | 99.3 / 94.2-94.3 / 93.2-93.5 |
| Pretrained (buffalo_l) | LFW 99.82, CFP-FP 97.44, AgeDB 98.12 |
| Collapse cases (ours -> pretrained) | 16px 82.2->98.2; blur4 81.6->98.9; JPEG5 81.6->98.9 |
| Noise | ~0.25 (LFW), ~0.5 (CFP-FP) |

## 16. Glossary

- **Embedding:** vector summarising a face; similar people -> similar vectors.
- **Verification / identification / clustering:** same-or-not / who-is-this / group-unlabelled.
- **Cosine similarity / angle:** direction similarity between unit vectors.
- **Margin (m):** extra angular gap forced between identities in training.
- **Scale (s):** multiplier on logits so softmax can be confident.
- **TTA:** test-time augmentation (here, horizontal flip).
- **Probe / enrolled:** the new photo being tested / the stored reference photo.
- **AMP:** automatic mixed precision (fp16 with fp32 where needed).
- **Ablation:** change one factor at a time to measure its effect.
- **Fine-tuning:** continuing training from an existing model at lower LR.
- **Open-set:** test identities were never seen in training (our setting).
