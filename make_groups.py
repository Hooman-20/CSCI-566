"""
No data? Build a labeled benchmark of synthetic "group photos".

Takes a face dataset (one folder per person, or LFW via scikit-learn), splits each
person's photos into gallery (for enrollment) and probe (for testing), then tiles 2-8
probe photos into one image. Some tiles come from "distractor" people who are NOT
enrolled, so the model has to say "unknown" for them.

Output:
  <out>/gallery/<person>/*.jpg     -> run: python tagger.py enroll --gallery <out>/gallery
  <out>/test/group_XXXX.jpg        -> run: python tagger.py eval --test-dir <out>/test --labels <out>/test_labels.csv
  <out>/test_labels.csv

Examples:
  python make_groups.py --source lfw --out data/lfw_groups          # downloads LFW (~200MB) on your machine
  python make_groups.py --source data/gallery --out data/demo_groups
"""
import argparse
import csv
import os
import random

import cv2
import numpy as np

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_folder_dataset(root):
    people = {}
    for p in sorted(os.listdir(root)):
        d = os.path.join(root, p)
        if os.path.isdir(d):
            imgs = [cv2.imread(os.path.join(d, f)) for f in sorted(os.listdir(d))
                    if f.lower().endswith(IMG_EXTS)]
            imgs = [i for i in imgs if i is not None]
            if imgs:
                people[p] = imgs
    return people


def load_lfw(min_faces):
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(color=True, resize=1.0, min_faces_per_person=min_faces,
                           slice_=None, download_if_missing=True)
    people = {}
    for img, t in zip(lfw.images, lfw.target):
        name = lfw.target_names[t].replace(" ", "_")
        bgr = cv2.cvtColor((img * 255 if img.max() <= 1 else img).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        people.setdefault(name, []).append(bgr)
    return people


class BystanderBlur:
    """
    Many source photos (LFW especially) contain other people besides the labeled one.
    Left visible, the model correctly recognizes them, but the label only names the main
    person, so correct answers get scored as false positives. This blurs every face except
    the main one: the largest face, down-weighted by its distance from the image centre
    (LFW photos are centred on the labeled person).
    """

    def __init__(self, use_gpu=False, det_thresh=0.3):
        from insightface.app import FaceAnalysis
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        if use_gpu:
            import onnxruntime as ort
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        self.app = FaceAnalysis(name="buffalo_l", providers=providers, allowed_modules=["detection"])
        # low det_thresh: catch faint/partial bystanders too (we only blur them, never label them)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640), det_thresh=det_thresh)
        self.stats = {"images": 0, "no_face": 0, "with_bystanders": 0, "faces_blurred": 0}

    def detect_all(self, img, paddings=(0.0, 0.1, 0.3), scales=(1.0, 0.75)):
        """
        Union of detections over several paddings/scales. A single pass misses bystanders that
        the eval's detector later finds once the photo is resized and placed in a tile:
        faces cut off at the photo's edge need margin, small faces need less of it.
        """
        found = []  # (score, box in original coords)
        for s in scales:
            im = img if s == 1.0 else cv2.resize(img, None, fx=s, fy=s)
            for pf in paddings:
                pad = int(pf * max(im.shape[:2]))
                p = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(235, 235, 235))
                found += [(float(f.det_score), (f.bbox - pad) / s) for f in self.app.get(p)]
        kept = []
        for score, b in sorted(found, key=lambda t: -t[0]):   # greedy dedupe, best score first
            if all(self._overlap(b, k) < 0.5 for k in kept):
                kept.append(b)
        return kept

    @staticmethod
    def _overlap(a, b):
        """Intersection over the SMALLER box: also merges a partial box sitting inside a full one."""
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return iw * ih / small if small > 0 else 0.0

    @staticmethod
    def main_face(boxes, shape):
        h, w = shape[:2]
        half_diag = 0.5 * np.hypot(w, h)
        def score(b):
            area = (b[2] - b[0]) * (b[3] - b[1])
            d = np.hypot((b[0] + b[2]) / 2 - w / 2, (b[1] + b[3]) / 2 - h / 2)
            return area * max(0.05, 1 - d / half_diag)
        return max(range(len(boxes)), key=lambda i: score(boxes[i]))

    def __call__(self, img):
        self.stats["images"] += 1
        boxes = self.detect_all(img)
        if not boxes:
            self.stats["no_face"] += 1
            return img
        m = self.main_face(boxes, img.shape)
        others = [b for i, b in enumerate(boxes) if i != m]
        if not others:
            return img
        self.stats["with_bystanders"] += 1
        self.stats["faces_blurred"] += len(others)
        out = img.copy()
        H, W = img.shape[:2]
        for b in others:
            bw, bh = b[2] - b[0], b[3] - b[1]
            x1, y1 = int(max(0, b[0] - 0.25 * bw)), int(max(0, b[1] - 0.25 * bh))
            x2, y2 = int(min(W, b[2] + 0.25 * bw)), int(min(H, b[3] + 0.25 * bh))
            if x2 <= x1 or y2 <= y1:
                continue
            k = int(max(bw, bh) * 0.6) | 1          # heavy: must not be re-detectable as a face
            roi = out[y1:y2, x1:x2]
            out[y1:y2, x1:x2] = cv2.GaussianBlur(cv2.GaussianBlur(roi, (k, k), 0), (k, k), 0)
        # bystander boxes can overlap the subject (e.g. someone cheek-to-cheek): restore the main face
        mx1, my1, mx2, my2 = [int(v) for v in boxes[m]]
        mx1, my1 = max(0, mx1), max(0, my1)
        out[my1:my2, mx1:mx2] = img[my1:my2, mx1:mx2]
        return out


def tile(images, cell=300, cols=None, pad=10, rng=None):
    n = len(images)
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * (cell + pad) + pad, cols * (cell + pad) + pad, 3), 235, np.uint8)
    for k, img in enumerate(images):
        # random scale so faces have different sizes, like a real group photo
        s = rng.uniform(0.6, 1.0)
        h, w = img.shape[:2]
        f = s * cell / max(h, w)
        im = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))))
        r, c = divmod(k, cols)
        y = pad + r * (cell + pad) + rng.integers(0, cell - im.shape[0] + 1)
        x = pad + c * (cell + pad) + rng.integers(0, cell - im.shape[1] + 1)
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="'lfw' or a folder with one subfolder per person")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-groups", type=int, default=300)
    ap.add_argument("--gallery-per-person", type=int, default=3)
    ap.add_argument("--min-faces", type=int, default=10, help="LFW only")
    ap.add_argument("--distractor-frac", type=float, default=0.3,
                    help="fraction of people kept OUT of the gallery (should be 'unknown')")
    ap.add_argument("--max-per-group", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-bystanders", action="store_true",
                    help="don't blur non-main faces (old behaviour; unlabeled faces stay visible)")
    ap.add_argument("--gpu", action="store_true", help="run the bystander face detector on CUDA")
    a = ap.parse_args()
    blur = None if a.keep_bystanders else BystanderBlur(use_gpu=a.gpu)
    cleaned = {}  # (name, idx) -> probe image with bystanders blurred (computed lazily, only if used)

    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    people = load_lfw(a.min_faces) if a.source == "lfw" else load_folder_dataset(a.source)
    names = sorted(people)
    random.shuffle(names)

    n_distract = int(len(names) * a.distractor_frac)
    distractors = names[:n_distract]
    enrolled = []
    gallery_dir = os.path.join(a.out, "gallery")
    probes = {}  # name -> images usable in group photos

    for name in names:
        imgs = people[name][:]
        random.shuffle(imgs)
        if name in distractors:
            probes[name] = imgs
            continue
        if len(imgs) <= a.gallery_per_person:   # need at least 1 left over to test with
            g, p = imgs[:max(1, len(imgs) - 1)], imgs[max(1, len(imgs) - 1):]
        else:
            g, p = imgs[:a.gallery_per_person], imgs[a.gallery_per_person:]
        if not p:
            p = g[-1:]  # tiny datasets: reuse (flagged below)
        os.makedirs(os.path.join(gallery_dir, name), exist_ok=True)
        for i, im in enumerate(g):
            cv2.imwrite(os.path.join(gallery_dir, name, f"{i}.jpg"), im)
        probes[name] = p
        enrolled.append(name)

    test_dir = os.path.join(a.out, "test")
    os.makedirs(test_dir, exist_ok=True)
    max_k = min(a.max_per_group, len(names))
    rows = []
    for gi in range(a.n_groups):
        k = int(rng.integers(min(2, max_k), max_k + 1))
        chosen = rng.choice(names, size=k, replace=False)
        picks = [(n, int(rng.integers(len(probes[n])))) for n in chosen]
        if blur is None:
            imgs = [probes[n][j] for n, j in picks]
        else:
            # no RNG use here, so the group composition is identical with or without blurring
            for n, j in picks:
                if (n, j) not in cleaned:
                    cleaned[(n, j)] = blur(probes[n][j])
            imgs = [cleaned[(n, j)] for n, j in picks]
        fname = f"group_{gi:04d}.jpg"
        cv2.imwrite(os.path.join(test_dir, fname), tile(imgs, rng=rng))
        truth = sorted(n for n in chosen if n in enrolled)
        rows.append((fname, ";".join(truth)))

    with open(os.path.join(a.out, "test_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "people"])
        w.writerows(rows)

    print(f"{len(names)} people: {len(enrolled)} enrolled, {len(distractors)} distractors (unknown)")
    print(f"wrote {a.n_groups} group photos -> {test_dir}")
    if blur is not None:
        s = blur.stats
        print(f"bystander blur: {s['images']} probe photos, {s['with_bystanders']} had bystanders "
              f"({s['faces_blurred']} faces blurred), {s['no_face']} with no detectable face")
    print(f"next: python tagger.py enroll --gallery {gallery_dir} --out {a.out}/gallery.npz")
    print(f"      python tagger.py eval --gallery-file {a.out}/gallery.npz "
          f"--test-dir {test_dir} --labels {a.out}/test_labels.csv --out-dir {a.out}/outputs")


if __name__ == "__main__":
    main()
