"""Group-photo evaluation: our embedder vs. pretrained buffalo_l inside the teammate's tagger flow.

Same detector (SCRFD from buffalo_l), same 5-point alignment (norm_crop 112x112), same Hungarian matching and
photo-level scoring (imported from the teammate's repo, unmodified). ONLY the embedder differs.

  python -m pipeline.group_eval --teammate ../teammate_CSCI-566 \
      --gallery group_eval/lfw_groups/gallery --test-dir group_eval/lfw_groups/test \
      --labels group_eval/lfw_groups/test_labels.csv \
      --ckpt base=models/base_slim.pt aug=models/aug_slim.pt --onnx <w600k_r50.onnx> --out group_eval/results.md
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# This package lives in <repo>/scratch_embedder/pipeline/; the tagger code (tagger.py, evaluate.py) is at <repo>/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def onnx_embedder(path):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    print("pretrained providers:", sess.get_providers())
    fixed = sess.get_inputs()[0].shape[0]

    def embed(crops_bgr):                                   # same recipe as train.hard_cases (flip TTA)
        out = []
        step = fixed if isinstance(fixed, int) else 64
        for i in range(0, len(crops_bgr), step):
            x = np.stack([c[:, :, ::-1] for c in crops_bgr[i:i + step]]).astype(np.float32)
            x = np.ascontiguousarray(((x - 127.5) / 127.5).transpose(0, 3, 1, 2))
            e = sess.run(None, {inp: x})[0] + sess.run(None, {inp: np.ascontiguousarray(x[:, :, :, ::-1])})[0]
            out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
    return embed


def detect(app, img):
    """-> (list of boxes, list of aligned 112x112 BGR crops)."""
    faces = app.get(img)
    return [f.bbox for f in faces], [norm_crop(img, f.kps, image_size=112) for f in faces]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teammate", default=REPO_ROOT,
                    help="path to the tagger code (tagger.py, evaluate.py); default: the repo root")
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ckpt", nargs="*", default=[], help="name=path of our slim checkpoints")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--out", default="group_eval/results.md")
    ap.add_argument("--max-photos", type=int, default=None)
    a = ap.parse_args()

    # appended (not prepended) so this package's own modules are never shadowed
    sys.path.append(os.path.abspath(a.teammate))
    from evaluate import load_labels, score              # teammate code, unchanged
    from tagger import match_faces, read_image

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                       allowed_modules=["detection"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    from pipeline.embedder import Embedder
    models = {}
    for c in a.ckpt:
        name, path = c.split("=", 1)
        models[name] = Embedder(path).embed
    if a.onnx:
        models["pretrained"] = onnx_embedder(a.onnx)

    # ---- enrollment crops: largest face of each gallery image (as in tagger.enroll)
    people = {}
    for person in sorted(os.listdir(a.gallery)):
        d = os.path.join(a.gallery, person)
        if not os.path.isdir(d):
            continue
        crops = []
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(IMG_EXTS):
                boxes, cr = detect(app, read_image(os.path.join(d, f)))
                if boxes:
                    k = max(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
                    crops.append(cr[k])
        if crops:
            people[person] = crops
    names = list(people)
    print(f"{len(names)} enrolled people")

    gt = load_labels(a.labels)
    photos = sorted(gt)[:a.max_photos]
    cache = []                                           # (n_faces, crops, truth)
    for i, p in enumerate(photos):
        boxes, cr = detect(app, read_image(os.path.join(a.test_dir, p)))
        cache.append((len(boxes), cr, gt[p]))
        if (i + 1) % 50 == 0:
            print(f"  detected {i+1}/{len(photos)}")
    n_faces = sum(c[0] for c in cache)
    n_truth = sum(len(c[2]) for c in cache)
    print(f"{len(photos)} photos, {n_faces} faces detected, {n_truth} ground-truth enrolled people")

    thresholds = np.arange(0.15, 0.71, 0.05)
    lines = [f"{len(photos)} photos, {len(names)} enrolled, {n_faces} faces detected, {n_truth} enrolled appearances\n",
             "| model | thr | precision | recall | F1 | exact-photo |", "|---|---|---|---|---|---|"]
    for mname, embed in models.items():
        flat = [c for _, cr, _ in cache for c in cr]
        E = embed(flat) if flat else np.zeros((0, 512))
        cents = []
        for n in names:
            c = embed(people[n]).mean(0)
            cents.append(c / np.linalg.norm(c))
        cents = np.stack(cents)
        per_photo, k = [], 0
        for nf, _, truth in cache:
            per_photo.append([SimpleNamespace(normed_embedding=e) for e in E[k:k + nf]]); k += nf
        truths = [c[2] for c in cache]
        results = []
        for thr in thresholds:
            preds = [{n for n, _ in match_faces(fs, names, cents, float(thr)) if n != "unknown"} for fs in per_photo]
            results.append((thr,) + score(preds, truths))
        best = max(results, key=lambda r: r[3])
        print(f"{mname}: best thr {best[0]:.2f} P {best[1]:.3f} R {best[2]:.3f} F1 {best[3]:.3f} exact {best[4]:.3f}")
        for thr, p, r, f1, ex in results:
            mark = " **(best F1)**" if thr == best[0] else ""
            lines.append(f"| {mname} | {thr:.2f}{mark} | {p:.3f} | {r:.3f} | {f1:.3f} | {ex:.3f} |")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("written", a.out)


if __name__ == "__main__":
    main()
