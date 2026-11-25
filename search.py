"""
Google-Photos-style search: give a few reference photos of ONE person and/or a scene, scan a
folder of (group) photos, and find every photo that matches.

Usage:
  python search.py --person refs/ --photos my_album/ --out results/
  python search.py --person me.jpg --photos my_album/ --out results/ --threshold 0.45 --gpu
  python search.py --person refs/ --scene beach --photos my_album/ --gpu    # person AND scene
  python search.py --scene concert indoor --photos my_album/ --gpu         # scene only

Outputs in --out:
  matches/<photo>       copy of each matching photo, with the person boxed in green
  results.csv           every photo: matched yes/no, similarity, face count, scene tags
Face embeddings and CLIP scene probabilities for the album are cached (album_cache.npz), so new
searches are instant. Scene labels: see scene.py (concert, beach, jungle/forest, wedding,
restaurant, office, street, stage, sports event, party, indoor, outdoor).
"""
import argparse
import csv
import os
import shutil
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np

from fastpath import Profiler, iter_images
from tagger import (DEFAULT_THRESHOLD, IMG_EXTS, detect_and_embed, load_models, model_id,
                    read_image, try_read_image, write_image)

CACHE_NAME = "album_cache.npz"


def list_images(path):
    if os.path.isfile(path):
        return [os.path.abspath(path)]
    out = []
    for root, _, files in os.walk(path):
        out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(IMG_EXTS)]
    # absolute paths: the same album reached via different relative paths shares cache entries
    return sorted(os.path.abspath(p) for p in out)


def file_sig(path):
    """Cache key component: re-index a photo if it was edited/replaced since it was cached."""
    st = os.stat(path)
    return (st.st_mtime_ns, st.st_size)


def person_embedding(app, ref_path):
    """Average embedding of the largest face in each reference photo (adapted if --adapter)."""
    embs = []
    for p in list_images(ref_path):
        faces = detect_and_embed(app, read_image(p))
        if not faces:
            print(f"  [warn] no face found in reference {p}")
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embs.append(f.normed_embedding)
    if not embs:
        raise SystemExit("no usable reference photos")
    c = np.mean(embs, axis=0)
    print(f"reference built from {len(embs)} photo(s)")
    return c / np.linalg.norm(c)


SCENE_KEY = "__scene_key__"   # cache-level entry: which CLIP model/labels the scene probs are for
MODEL_KEY = "__model__"       # cache-level entry: which recognizer made the face embeddings


def index_album(app, photos_dir, cache_path, use_cache=True, batch=True, prefetch=True,
                profiler=None, scene_tagger=None):
    """Detect + embed every face in the album once, and (with scene_tagger) store CLIP scene
    probabilities per photo; reuse both on later searches.
    batch: one ArcFace call per photo; prefetch: decode images in a background thread;
    profiler: fastpath.Profiler to fill with a per-stage time breakdown.
    Photos whose faces are cached but whose scene tags are missing or were made with other
    labels/model only get the scene step.
    Returns (index, unreadable_paths, n_photos_processed, seconds_spent_indexing)."""
    paths = list_images(photos_dir)
    cache = {}
    if use_cache and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        cache = d["index"].item()
    sigs = {p: file_sig(p) for p in paths}
    # face embeddings from another recognizer (--model) are useless here: re-embed everything
    same_model = cache.get(MODEL_KEY, "") == model_id(app)
    new = {p for p in paths if not same_model or p not in cache or cache[p].get("sig") != sigs[p]}
    need_scene = set()
    if scene_tagger is not None:
        stale = cache.get(SCENE_KEY) != scene_tagger.key
        need_scene = {p for p in paths if p in new or stale or cache[p].get("scene") is None}
    todo = [p for p in paths if p in new or p in need_scene]

    def read(p):
        # runs in the prefetch thread: JPEG decode + CLIP resize/crop overlap GPU work
        img = try_read_image(p)
        clip_in = scene_tagger.preprocess(img) if img is not None and p in need_scene else None
        return img, clip_in

    unreadable = []
    timer = profiler
    t0 = time.perf_counter()
    if timer:
        timer.start()
    for i, (p, (img, clip_in)) in enumerate(iter_images(todo, read, prefetch=prefetch), 1):
        if timer:
            timer.lap("decode")
        if img is None:
            print(f"  [warn] could not read {p}")
            unreadable.append(p)
            cache.pop(p, None)
            continue
        if p in new:
            # cache raw ArcFace embeddings; any --adapter is applied at search time, so one
            # cache serves runs with and without an adapter
            faces = detect_and_embed(app, img, timer, batch=batch, adapt=False)
            cache[p] = {
                "sig": sigs[p],
                "boxes": np.array([f.bbox for f in faces]).reshape(-1, 4),
                "embs": np.array([f.normed_embedding for f in faces]).reshape(-1, 512),
            }
            if timer:
                timer.n_faces += len(faces)
        if p in need_scene:
            cache[p]["scene"] = scene_tagger.probs_from_input(clip_in)
            if timer:
                timer.lap("scene")
        if i % 100 == 0 or i == len(todo):
            print(f"  indexed {i}/{len(todo)} photos")
        if timer:
            timer.n_photos += 1
            timer.lap("overhead")
    elapsed = time.perf_counter() - t0
    cache[MODEL_KEY] = model_id(app)
    if scene_tagger is not None:
        cache[SCENE_KEY] = scene_tagger.key
    if todo or not os.path.exists(cache_path):
        np.savez(cache_path, index=np.array(cache, dtype=object))
    n_done = len(todo) - len(unreadable)
    return {p: cache[p] for p in paths if p in cache}, unreadable, n_done, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", help="reference photo or folder of photos of the person")
    ap.add_argument("--scene", nargs="+", metavar="LABEL",
                    help="only photos with ALL these scene tags, e.g. --scene beach or --scene "
                         "concert indoor (labels in scene.py)")
    ap.add_argument("--scene-min-p", type=float, default=None,
                    help="CLIP probability a scene label needs to count (default scene.SCENE_MIN_P)")
    ap.add_argument("--no-scenes", action="store_true",
                    help="skip CLIP scene tagging while indexing (face-only benchmarking)")
    ap.add_argument("--photos", required=True, help="folder of photos to search")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-index every photo (use for CPU vs GPU benchmarking)")
    ap.add_argument("--trt", action="store_true",
                    help="TensorRT EP in FP16 (implies --gpu; engines cached in trt_cache/)")
    ap.add_argument("--no-batch", action="store_true",
                    help="embed faces one ArcFace call at a time (old behaviour, for benchmarking)")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="decode images on the main thread (old behaviour, for benchmarking)")
    ap.add_argument("--model", metavar="ONNX",
                    help="replacement recognizer, e.g. models/arcface_r50_ft.onnx (train_finetune.py)")
    ap.add_argument("--adapter", metavar="NPZ",
                    help="apply a train_adapter.py adapter to the reference and album embeddings")
    ap.add_argument("--profile", metavar="MD_FILE",
                    help="time each stage while indexing and append a ms/photo table to MD_FILE")
    a = ap.parse_args()
    if a.trt and a.no_batch:
        ap.error("--no-batch is CUDA-only: the TensorRT engine is built for the fixed batch size")
    if not a.person and not a.scene:
        ap.error("give --person, --scene, or both")
    if a.scene and a.no_scenes:
        ap.error("--scene needs scene tags; drop --no-scenes")
    import scene as scn
    wanted = scn.parse_scene_args(a.scene) if a.scene else []
    min_p = scn.SCENE_MIN_P if a.scene_min_p is None else a.scene_min_p

    # matches/ reflects only THIS search; otherwise hits from a previous person linger
    matches_dir = os.path.join(a.out, "matches")
    shutil.rmtree(matches_dir, ignore_errors=True)
    os.makedirs(matches_dir)

    t_load = time.perf_counter()
    app = load_models(use_gpu=a.gpu, use_trt=a.trt, adapter=a.adapter, model=a.model)
    scene_tagger = None if a.no_scenes else scn.SceneTagger(use_gpu=a.gpu, use_trt=a.trt)
    t_load = time.perf_counter() - t_load
    # warm up the models before the timed loop
    ref = person_embedding(app, a.person) if a.person else None
    if ref is None:
        detect_and_embed(app, np.zeros((640, 640, 3), np.uint8))
    if scene_tagger is not None:
        scene_tagger.probs(np.zeros((224, 224, 3), np.uint8))
    profiler = Profiler() if a.profile else None
    index, unreadable, n_indexed, t_index = index_album(
        app, a.photos, os.path.join(a.out, CACHE_NAME), use_cache=not a.no_cache,
        batch=not a.no_batch, prefetch=not a.no_prefetch, profiler=profiler,
        scene_tagger=scene_tagger)

    rows, n_match = [], 0
    for path in unreadable:
        rows.append((path, "unreadable", "", 0, ""))
    for path, d in index.items():
        probs = d.get("scene")
        tags = ";".join(scn.tags_from_probs(probs, min_p)) if probs is not None else ""
        scene_ok = not wanted or (probs is not None and scn.matches_scene(probs, wanted, min_p))
        sim, best = "", None
        face_ok = ref is None
        if ref is not None and len(d["embs"]):
            embs = app.adapter(d["embs"]) if app.adapter is not None else d["embs"]
            sims = embs @ ref
            best = int(np.argmax(sims))
            sim = f"{sims[best]:.3f}"
            face_ok = sims[best] >= a.threshold
        hit = face_ok and scene_ok
        rows.append((path, "yes" if hit else "no", sim, len(d["embs"]), tags))
        if hit:
            n_match += 1
            img = read_image(path)
            if ref is not None:
                x1, y1, x2, y2 = d["boxes"][best].astype(int)
                t = max(2, max(img.shape[:2]) // 300)
                cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 60), t)
            rel = os.path.relpath(path, a.photos) if os.path.isdir(a.photos) else os.path.basename(path)
            write_image(os.path.join(matches_dir, rel.replace(os.sep, "__")), img)

    rows.sort(key=lambda r: (r[1] != "yes", -(float(r[2]) if r[2] else -1)))
    with open(os.path.join(a.out, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["photo", "match", "best_similarity", "faces", "scenes"])
        w.writerows(rows)

    what = " + ".join(filter(None, ["person" if a.person else "", " & ".join(wanted)]))
    print(f"\n{what} found in {n_match} of {len(index)} photos:")
    for p, m, s, n, tg in rows:
        if m == "yes":
            print(f"  {os.path.basename(p):<28} " + (f"sim={s}  " if s else "") +
                  f"({n} faces)  scenes: {tg.replace(';', ', ')}")
    if unreadable:
        print(f"\n{len(unreadable)} unreadable file(s) listed in results.csv")
    print(f"\nmarked copies -> {a.out}/matches/   full list -> {a.out}/results.csv")

    device = "GPU (TensorRT FP16)" if a.trt else "GPU (CUDA)" if a.gpu else "CPU"
    config = (f"{device}, {'per-face' if a.no_batch else 'batched'} embedding, "
              f"{'no prefetch' if a.no_prefetch else 'prefetch'}"
              f"{'' if a.no_scenes else ', CLIP scene tags'}")
    print(f"\ntiming [{config}]: model load {t_load:.1f}s")
    if n_indexed:
        print(f"  indexed {n_indexed} photo(s) in {t_index:.2f}s -> "
              f"{n_indexed / t_index:.1f} photos/s  ({1000 * t_index / n_indexed:.1f} ms/photo, "
              f"detect + embed{'' if a.no_scenes else ' + scene tags'})")
    else:
        print("  all photos served from cache, nothing indexed; use --no-cache to benchmark")
    if profiler and profiler.n_photos:
        table = profiler.table()
        print("\n" + table)
        with open(a.profile, "a", encoding="utf-8") as f:
            f.write(f"\n### {config}\n`{a.photos}`, {profiler.n_photos} photos\n\n{table}\n")
        print(f"profile appended to {a.profile}")


if __name__ == "__main__":
    main()
