# Group Photo Tagger

Finds every face in a group photo and labels who it is ("A and B are in this photo"),
or "unknown" for people not enrolled.

```
photo ─► SCRFD detector ─► 5-point alignment (112×112) ─► ArcFace R50 ─► 512-d embedding
                                                                              │
gallery (avg embedding per person) ─────────────── cosine similarity + Hungarian match ─► name / unknown
```

Adding a new person = dropping a few photos in a folder and re-running `enroll`. No retraining.

## Setup

```bash
pip install -r requirements.txt
# GPU (RTX 5090): swap in onnxruntime-gpu + its CUDA 13/cuDNN 9 runtime wheels, then add --gpu
pip uninstall -y onnxruntime && pip install "onnxruntime-gpu[cuda,cudnn]"
```
Pretrained models (`buffalo_l`) download automatically to `~/.insightface/models` on first run.

## Usage

```bash
# 1. enroll: data/gallery/<person_name>/*.jpg  (3-10 clear photos each is ideal)
python tagger.py enroll --gallery data/gallery --out gallery.npz

# 2. tag one photo -> prints who's in it, writes photo_tagged.jpg
python tagger.py tag --image some_group_photo.jpg

# 3. evaluate on labeled photos (data/test + data/test_labels.csv), with threshold sweep
python tagger.py eval
```

## No data? Generate a benchmark

`make_groups.py` builds hundreds of labeled synthetic group photos from LFW (13k photos of
5.7k people, downloaded automatically by scikit-learn). It holds out ~30% of people as
**distractors** so the model is also tested on saying "unknown". With the default
`--min-faces 10` that is 158 people (111 enrolled, 47 distractors).

```bash
python make_groups.py --source lfw --out data/lfw_groups --n-groups 500 --gpu
python tagger.py enroll --gallery data/lfw_groups/gallery --out data/lfw_groups/gallery.npz
python tagger.py eval --gallery-file data/lfw_groups/gallery.npz \
    --test-dir data/lfw_groups/test --labels data/lfw_groups/test_labels.csv \
    --out-dir data/lfw_groups/outputs --gpu
```

Metrics are photo-level: precision/recall over "who is in this photo", and exact-match rate
(the predicted set of people is exactly right).

## Find one person across an album

```bash
python search.py --person data/gallery/obama --photos data/test --gpu
```
Writes boxed copies of matching photos to `results/matches/` and every photo's best
similarity to `results/results.csv`. Album embeddings are cached, so later searches are
instant; add `--no-cache` to force re-indexing (e.g. for timing).

## Results so far

LFW groups benchmark: 500 synthetic group photos (2–8 faces each), 111 enrolled people, 47
distractors, buffalo_l (SCRFD-10G + ArcFace R50), default threshold **0.45**.

| Precision | Recall | F1 | Photos fully correct |
|---|---|---|---|
| 0.999 | 0.997 | 0.998 | 98.8% (494 / 500) |

**Threshold sweep** (cosine similarity). 0.42–0.45 tie for best, and recall starts to drop at 0.48:

| thr | precision | recall | F1 | exact |
|---|---|---|---|---|
| 0.15 | 0.767 | 0.999 | 0.868 | 0.330 |
| 0.20 | 0.917 | 0.999 | 0.956 | 0.732 |
| 0.25 | 0.977 | 0.998 | 0.988 | 0.918 |
| 0.30 | 0.989 | 0.997 | 0.993 | 0.956 |
| 0.35 | 0.996 | 0.997 | 0.996 | 0.978 |
| 0.40 | 0.998 | 0.997 | 0.997 | 0.986 |
| **0.45** | **0.999** | **0.997** | **0.998** | **0.988** |
| 0.50 | 0.999 | 0.994 | 0.996 | 0.978 |
| 0.55 | 0.999 | 0.983 | 0.991 | 0.942 |
| 0.60 | 0.999 | 0.954 | 0.976 | 0.842 |
| 0.65 | 0.999 | 0.905 | 0.950 | 0.700 |
| 0.70 | 0.999 | 0.782 | 0.877 | 0.452 |

### Benchmark fix: blurring bystanders
The first run scored 0.986 precision and 94.6% fully correct photos. Most of those "false
positives" were actually correct answers. Many LFW photos contain other people besides the
labeled one (George W. Bush kissing Laura Bush, Erdoğan next to Gül, Colin Powell in the
background). The model recognized them correctly, but the label lists only the main person, so
each one counted as an error: 22 of the 25 false positives were bystanders like these.

`make_groups.py` now detects every face in each source photo before tiling and heavily blurs
all faces except the main one (largest face, weighted toward the centre). Detection runs over
several paddings and scales and merges the results, because a single pass missed faces cut off
at the photo's edge and small background faces. The random seed still produces the same 500
groups, so before and after compare directly. Use `--keep-bystanders` to get the old behaviour.

### Remaining errors (6 photos)
- **4 real misses:**
  - **Nicole Kidman (×3):** all three of her gallery photos are blonde red-carpet shots, but her
    test photos are dark-haired film stills (similarity 0.24–0.28). This needs more varied
    gallery photos; changing the threshold won't fix it.
  - **Saddam Hussein (×1):** sunglasses and a blurry photo, similarity 0.37.
- **2 likely LFW labeling problems:** the photo labeled Michael Schumacher shows a face that
  looks like Rubens Barrichello, and in the photo labeled Erdoğan the centred face is Gül.

### Speed: CPU vs GPU (RTX 5090)
Detection + embedding, from `search.py --no-cache` on 200 photos with 1–3 faces each:

| | photos/s | ms/photo |
|---|---|---|
| CPU | 13.9 | 71.8 |
| GPU (CUDA, onnxruntime-gpu) | 44.5 | 22.5 |

On LFW groups (up to 12 faces per photo) the GPU drops to about 31 photos/s, because faces are
embedded one at a time. Batching should help there.

## Roadmap
- [x] Baseline numbers on LFW groups; pick threshold from the sweep (0.45)
- [x] Clean benchmark labels: blur unlabeled bystander faces in LFW source photos
- [ ] Hard cases: small faces, profiles, occlusion — measure each separately
- [ ] More varied gallery photos per person (e.g. Kidman misses)
- [ ] Cluster "unknown" faces across an album (HDBSCAN) so you can name a cluster once
- [ ] Train own embedder with ArcFace loss (CASIA-WebFace), compare to pretrained on LFW/CFP-FP
- [x] Throughput benchmark GPU vs CPU (`search.py` timing printout)
- [ ] TensorRT FP16/INT8 export; batch all faces per photo
- [ ] Gradio demo

## Data note
Only use photos of people who agreed to it, or public benchmark datasets. The sample
images in `data/` come from the MIT-licensed `ageitgey/face_recognition` repo examples.
