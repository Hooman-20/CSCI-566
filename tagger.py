"""
Group photo tagger: detect faces -> align -> embed (ArcFace) -> match to a gallery.

Usage:
  python tagger.py enroll  --gallery data/gallery --out gallery.npz
  python tagger.py tag     --gallery-file gallery.npz --image path/to/photo.jpg --out tagged.jpg
  python tagger.py eval    --gallery-file gallery.npz --test-dir data/test
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.optimize import linear_sum_assignment

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_THRESHOLD = 0.45  # cosine similarity; best F1 + exact-match in the LFW-groups sweep (0.42-0.45 tie). Re-tune on your own data


# ---------------------------------------------------------------- models
TRT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trt_cache")


def _trt_options(**extra):
    """TensorRT EP in FP16 with an on-disk engine cache (first run builds engines, ~1 min)."""
    os.makedirs(TRT_CACHE_DIR, exist_ok=True)
    opts = {
        "trt_fp16_enable": "True",
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": TRT_CACHE_DIR,
        "trt_timing_cache_enable": "True",
        "trt_timing_cache_path": TRT_CACHE_DIR,
    }
    opts.update(extra)
    return opts


def load_models(det_size=640, use_gpu=False, use_trt=False, adapter=None, model=None):
    """adapter: path to an adapter .npz from train_adapter.py, applied to every embedding.
    model: path to a replacement recognition ONNX (e.g. models/arcface_r50_ft.onnx from
    train_finetune.py; same input layout as w600k_r50)."""
    app = _load_models(det_size, use_gpu, use_trt, model)
    if adapter:
        from adapter import Adapter
        app.adapter = Adapter(adapter)
    else:
        app.adapter = None
    return app


def file_id(path):
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def adapter_id(app):
    return app.adapter.id if getattr(app, "adapter", None) is not None else ""


def model_id(app):
    """'' for the stock buffalo_l recognizer, else a hash of the --model file. Embeddings from
    different recognizers live in different spaces: galleries and album caches record this."""
    return getattr(app, "model_id", "")


def _load_models(det_size=640, use_gpu=False, use_trt=False, model=None):
    # buffalo_l = SCRFD-10G detector + ArcFace ResNet50 (trained on WebFace600K)
    use_gpu = use_gpu or use_trt
    if use_gpu:
        import onnxruntime as ort
        # onnxruntime-gpu needs CUDA 13 + cuDNN 9 DLLs. They come from the nvidia-* pip wheels
        # (pip install "onnxruntime-gpu[cuda,cudnn]"), which are not on PATH: load them explicitly.
        # This MUST run before the first CUDA session, or ORT silently falls back to CPU.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers,
                       allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))
    app.model_id = ""
    if model:
        # swap the recognizer's session; keep ArcFaceONNX's preprocessing (mean/std 127.5)
        import onnxruntime as ort
        rec = app.models["recognition"]
        rec.model_file = os.path.abspath(model)
        rec.session = ort.InferenceSession(rec.model_file, providers=providers)
        rec.input_name = rec.session.get_inputs()[0].name
        rec.output_names = [o.name for o in rec.session.get_outputs()]
        app.model_id = file_id(model)
    if use_trt:
        # TensorRT 10 DLLs come from the tensorrt-cu13==10.x pip wheel (onnxruntime-gpu 1.31 is
        # built against TensorRT 10, not 11); importing it puts nvinfer_10 on the DLL path.
        import onnxruntime as ort
        import tensorrt  # noqa: F401
        from fastpath import EMBED_BATCH
        trt_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        def trt_session(model, shape):
            # one engine for exactly this input shape; each model gets its own options because
            # both call their input "input.1"
            prof = f"{model.input_name}:{shape}"
            return ort.InferenceSession(model.model_file, providers=trt_providers, provider_options=[
                _trt_options(trt_profile_min_shapes=prof, trt_profile_opt_shapes=prof,
                             trt_profile_max_shapes=prof), {}, {}])

        # Detector: insightface would clone a fixed-size session by copying provider options
        # back out of ORT, which breaks for TensorRT (it silently fell back to CUDA). Instead,
        # give it a fixed det_size engine and route every request to that main session.
        det = app.det_model
        det.session = trt_session(det, f"1x3x{det_size}x{det_size}")
        det.static_shape_sessions = False
        det._reset_resolution_session_pool()
        # ArcFace: fastpath always feeds exactly EMBED_BATCH faces (zero-padded)
        rec = app.models["recognition"]
        rec.session = trt_session(rec, f"{EMBED_BATCH}x3x112x112")
    if use_gpu:
        # onnxruntime silently falls back to CPU if CUDA/TensorRT libs fail to load -> fail loudly
        want = "TensorrtExecutionProvider" if use_trt else "CUDAExecutionProvider"
        for name, model in app.models.items():
            if want not in model.session.get_providers():
                raise RuntimeError(f"GPU requested but model '{name}' has no {want} "
                                   f"(providers: {model.session.get_providers()}); "
                                   "check onnxruntime-gpu / CUDA / cuDNN / TensorRT install")
    return app


def try_read_image(path):
    """cv2.imread can't open non-ASCII paths on Windows; decode from bytes instead. None on failure."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def read_image(path):
    img = try_read_image(path)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


def write_image(path, img):
    """Unicode-safe cv2.imwrite."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img)
    if not ok:
        raise ValueError(f"could not encode image: {path}")
    buf.tofile(path)


def detect_and_embed(app, img, timer=None, batch=True, adapt=True):
    """Returns list of faces; each has .bbox, .det_score, .normed_embedding (512-d, L2-normalized).
    All faces of the photo are embedded in one batched ArcFace call (see fastpath.py), then
    passed through the adapter if one was loaded (adapt=False returns raw ArcFace embeddings)."""
    import fastpath
    return fastpath.detect_and_embed(app, img, timer or fastpath.NO_TIMER, batch=batch, adapt=adapt)


# ---------------------------------------------------------------- enrollment
def enroll(app, gallery_dir):
    """One folder per person; each image should contain that person (largest face is used)."""
    names, centroids = [], []
    for person in sorted(os.listdir(gallery_dir)):
        pdir = os.path.join(gallery_dir, person)
        if not os.path.isdir(pdir):
            continue
        embs = []
        for f in sorted(os.listdir(pdir)):
            if not f.lower().endswith(IMG_EXTS):
                continue
            faces = detect_and_embed(app, read_image(os.path.join(pdir, f)))
            if not faces:
                print(f"  [warn] no face in {person}/{f}")
                continue
            # biggest face = the enrolled person
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            embs.append(face.normed_embedding)
        if not embs:
            print(f"  [warn] skipping {person}: no usable photos")
            continue
        c = np.mean(embs, axis=0)
        centroids.append(c / np.linalg.norm(c))
        names.append(person)
        print(f"  enrolled {person:<16} from {len(embs)} photo(s)")
    return names, np.stack(centroids)


def save_gallery(path, names, centroids, adapter="", model=""):
    """adapter / model: ids of the adapter and recognizer the embeddings came from
    ('' = stock buffalo_l ArcFace, no adapter)."""
    np.savez(path, names=np.array(names), centroids=centroids, adapter=np.array(adapter),
             model=np.array(model))


def load_gallery(path, expect_adapter=None, expect_model=None):
    """Refuse a gallery enrolled with a different adapter (expect_adapter) or recognizer
    (expect_model): their embeddings live in different spaces and must not be compared.
    None skips a check; galleries from before these fields count as ''."""
    d = np.load(path, allow_pickle=False)
    for field, want, flag in (("adapter", expect_adapter, "--adapter"),
                              ("model", expect_model, "--model")):
        if want is None:
            continue
        got = str(d[field]) if field in d else ""
        if got != want:
            def desc(x):
                return f"{field} {x}" if x else f"no {flag}"
            raise SystemExit(f"{path} was enrolled with {desc(got)} but this run uses "
                             f"{desc(want)}; re-enroll with the same {flag} setting")
    return [str(n) for n in d["names"]], d["centroids"]


# ---------------------------------------------------------------- matching
def match_faces(faces, names, centroids, threshold=DEFAULT_THRESHOLD):
    """
    One-to-one assignment (Hungarian): a person can appear at most once per photo.
    Returns list of (name_or_'unknown', similarity) aligned with faces.
    """
    if not faces:
        return []
    E = np.stack([f.normed_embedding for f in faces])   # (F, 512)
    S = E @ centroids.T                                   # (F, P) cosine similarity
    rows, cols = linear_sum_assignment(-S)                # maximize total similarity
    result = [("unknown", float(S[i].max())) for i in range(len(faces))]
    for r, c in zip(rows, cols):
        if S[r, c] >= threshold:
            result[r] = (names[c], float(S[r, c]))
    return result


def tag_image(app, img, names, centroids, threshold=DEFAULT_THRESHOLD):
    faces = detect_and_embed(app, img)
    labels = match_faces(faces, names, centroids, threshold)
    return faces, labels


def draw(img, faces, labels):
    out = img.copy()
    scale = max(out.shape[:2]) / 800
    for f, (name, sim) in zip(faces, labels):
        x1, y1, x2, y2 = f.bbox.astype(int)
        color = (60, 200, 60) if name != "unknown" else (60, 60, 230)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(2 * scale)))
        text = f"{name} {sim:.2f}"
        fs = 0.6 * scale
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        y_text = max(y1, th + 6)
        cv2.rectangle(out, (x1, y_text - th - 6), (x1 + tw + 6, y_text), color, -1)
        cv2.putText(out, text, (x1 + 3, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out


def summary(labels):
    found = sorted({n for n, _ in labels if n != "unknown"})
    unknown = sum(1 for n, _ in labels if n == "unknown")
    if not found:
        s = "No known people found"
    elif len(found) == 1:
        s = f"{found[0]} is in this photo"
    else:
        s = ", ".join(found[:-1]) + f" and {found[-1]} are in this photo"
    if unknown:
        s += f" (+{unknown} unknown face{'s' if unknown > 1 else ''})"
    return s


# ---------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("enroll")
    e.add_argument("--gallery", default="data/gallery")
    e.add_argument("--out", default="gallery.npz")

    t = sub.add_parser("tag")
    t.add_argument("--gallery-file", default="gallery.npz")
    t.add_argument("--image", required=True)
    t.add_argument("--out", default=None)
    t.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    v = sub.add_parser("eval")
    v.add_argument("--gallery-file", default="gallery.npz")
    v.add_argument("--test-dir", default="data/test")
    v.add_argument("--labels", default="data/test_labels.csv")
    v.add_argument("--out-dir", default="outputs")
    v.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    for sp in (e, t, v):
        sp.add_argument("--gpu", action="store_true", help="use CUDA via onnxruntime-gpu")
        sp.add_argument("--trt", action="store_true",
                        help="use the TensorRT EP in FP16 (implies --gpu; engines cached in trt_cache/)")
        sp.add_argument("--model", metavar="ONNX",
                        help="replacement recognizer, e.g. models/arcface_r50_ft.onnx from "
                             "train_finetune.py; the gallery must be enrolled with the same model")
        sp.add_argument("--adapter", metavar="NPZ",
                        help="apply a train_adapter.py adapter to every embedding; the gallery must "
                             "be enrolled with the same adapter")

    a = p.parse_args()
    app = load_models(use_gpu=a.gpu, use_trt=a.trt, adapter=a.adapter, model=a.model)

    if a.cmd == "enroll":
        names, cents = enroll(app, a.gallery)
        save_gallery(a.out, names, cents, adapter=adapter_id(app), model=model_id(app))
        print(f"saved {len(names)} people -> {a.out}")

    elif a.cmd == "tag":
        names, cents = load_gallery(a.gallery_file, expect_adapter=adapter_id(app),
                                    expect_model=model_id(app))
        img = read_image(a.image)
        faces, labels = tag_image(app, img, names, cents, a.threshold)
        for f, (n, s) in zip(faces, labels):
            print(f"  box={f.bbox.astype(int).tolist()}  {n}  sim={s:.3f}")
        print(summary(labels))
        out = a.out or os.path.splitext(a.image)[0] + "_tagged.jpg"
        write_image(out, draw(img, faces, labels))
        print(f"wrote {out}")

    elif a.cmd == "eval":
        from evaluate import run_eval
        names, cents = load_gallery(a.gallery_file, expect_adapter=adapter_id(app),
                                    expect_model=model_id(app))
        run_eval(app, names, cents, a.test_dir, a.labels, a.out_dir, a.threshold)


if __name__ == "__main__":
    sys.exit(main())
