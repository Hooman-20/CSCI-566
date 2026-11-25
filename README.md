# Snap-AI: our own face embedder (CSCI 566)

Event-photo organization needs a model that turns a face into a vector so photos can be grouped by person.
We **trained our own embedder from scratch** (IR-ResNet-34 + ArcFace loss on CASIA-WebFace), ran ablations, compared
it to pretrained InsightFace `buffalo_l` under hard conditions, fixed its main weakness with quality augmentation, and
plugged it into the group-photo tagger from [Hooman-20/CSCI-566](https://github.com/Hooman-20/CSCI-566).

```
photo -> SCRFD detector (pretrained) -> 5-point align 112x112 -> OUR embedder (512-d) -> cosine + Hungarian match
```

## Results at a glance

| | LFW | CFP-FP (pose) | AgeDB-30 (age) |
|---|---|---|---|
| ours (r34, ArcFace, 25 epochs) | 99.30 | 94.29 | 93.23 |
| pretrained buffalo_l | 99.82 | 97.44 | 98.12 |

Hard cases (LFW, probe image degraded): before quality augmentation our model fell to ~82% on heavy blur, JPEG q=5 and
16 px faces (pretrained: >98%). After a 5-epoch augmentation fine-tune: blur 98.3, JPEG 95.8, 16 px 91.7. Masks,
sunglasses, pose and age gaps are unchanged.

Group-photo test (300 synthetic LFW photos, same detector and matching, only the embedder differs):

| model | F1 | exact-photo match |
|---|---|---|
| ours (before fix) | 0.989 | 92.3% |
| ours (after fix) | 0.990 | 93.3% |
| pretrained buffalo_l | 0.999 | 99.0% |

Full tables: [`results/`](results/). Details and reasoning: [`TEAM_BRIEF.md`](TEAM_BRIEF.md) (short),
[`PROJECT_EXPLAINED.md`](PROJECT_EXPLAINED.md) (full), [`PRESENTER_GUIDE.md`](PRESENTER_GUIDE.md) (concepts and Q&A).

## Layout

| Path | What |
|---|---|
| `train/model.py` | IR-ResNet r18/r34/r50 backbone |
| `train/arcface.py` | ArcFace, CosFace and softmax heads (written from scratch) |
| `train/train.py` | training loop (AMP, margin warmup, cosine LR, resume, `--init` fine-tune, `--degrade-p` augmentation) |
| `train/data.py` | image-folder dataset, flip/colour augmentation, `RandomDegrade` (low-res / blur / JPEG) |
| `train/eval_verification.py` | LFW/CFP-FP/AgeDB `.bin` evaluation (10-fold, flip TTA) |
| `train/ablate.py` | ablation grid runner and summary |
| `train/hard_cases.py` | degraded-probe study, ours vs pretrained ONNX |
| `tools/rec_to_folder.py` | converts InsightFace `train.rec` to image folders (no mxnet needed) |
| `pipeline/embedder.py` | drop-in embedder: aligned 112x112 BGR faces -> 512-d unit vectors |
| `pipeline/group_eval.py` | our embedder inside Hooman's tagger flow (imports his repo unmodified) |
| `notebooks/colab_train.ipynb` | the Colab notebook used for training |
| `results/` | tables, per-run training logs, pictures of each degradation |

## Setup

```bash
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements-train.txt
uv pip install insightface onnxruntime scipy scikit-learn   # only for hard_cases (pretrained baseline) and group_eval
```

## Weights

Model weights are **not in the repo** (about 140 MB each). Ask for the Drive link, then place them in `models/`:
`base_slim.pt` (trained model) and `aug_slim.pt` (after quality-augmentation fine-tune; use this one).

## Use our embedder

```python
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
from pipeline.embedder import Embedder

det = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"]); det.prepare(ctx_id=-1)
emb = Embedder("models/aug_slim.pt")
faces = det.get(img)                                            # img: BGR numpy array
vecs = emb.embed([norm_crop(img, f.kps, image_size=112) for f in faces])   # (n, 512), L2-normalized
```

Similarity is the dot product of two vectors. A threshold of 0.45 worked best for our model in the group-photo test.

## Reproduce

Training used Colab Pro (T4) and the InsightFace CASIA-WebFace pack (`faces_webface_112x112`: `train.rec`, `lfw.bin`,
`cfp_fp.bin`, `agedb_30.bin`). Convert, train, evaluate:

```bash
python tools/rec_to_folder.py --rec <pack>/train.rec --out data/casia_112
python -m train.train --data data/casia_112 --val-bins <pack>/lfw.bin <pack>/cfp_fp.bin <pack>/agedb_30.bin \
    --arch r34 --epochs 25 --batch 256 --out runs/r34_casia
# quality-augmentation fine-tune
python -m train.train --data data/casia_112 --val-bins <pack>/lfw.bin <pack>/cfp_fp.bin <pack>/agedb_30.bin \
    --arch r34 --epochs 5 --batch 256 --lr 0.02 --warmup-epochs 0.3 --margin-warmup-epochs 0 \
    --degrade-p 0.5 --init runs/r34_casia/last.pt --resume --out runs/r34_casia_aug
# ablations (2,000 ids, 8 epochs)
python -m train.ablate run --data data/casia_112 --val-bins <pack>/lfw.bin <pack>/cfp_fp.bin \
    --epochs 8 --extra --max-ids 2000
# hard cases: ours vs pretrained
python -m train.hard_cases --ckpt base=models/base_slim.pt aug=models/aug_slim.pt \
    --degrade-bin <pack>/lfw.bin --extra-bins <pack>/cfp_fp.bin <pack>/agedb_30.bin \
    --onnx ~/.insightface/models/buffalo_l/w600k_r50.onnx --out results/hard_cases_aug.md
```

Group-photo evaluation (clone Hooman's repo next to this one, build the synthetic LFW groups with his `make_groups.py`):

```bash
python make_groups.py --source lfw --out ../Project_Snap_AI/group_eval/lfw_groups --n-groups 300   # run inside his repo
python -m pipeline.group_eval --teammate ../CSCI-566 --gallery group_eval/lfw_groups/gallery \
    --test-dir group_eval/lfw_groups/test --labels group_eval/lfw_groups/test_labels.csv \
    --ckpt base=models/base_slim.pt aug=models/aug_slim.pt \
    --onnx ~/.insightface/models/buffalo_l/w600k_r50.onnx --out results/group_eval.md
```

## Caveats

Single seed; ablations at reduced scale (2,000 identities, 8 epochs); hard cases are synthetic degradations on aligned
crops; no no-augmentation control run; group-photo benchmark is synthetic LFW tiles with thresholds chosen on the same
photos. Not built: HDBSCAN clustering of unknown faces, evaluation on real event photos, dedup / quality / best-shot modules.
