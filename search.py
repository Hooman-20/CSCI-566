"""
Google-Photos-style person search: give a few reference photos of ONE person,
scan a folder of (group) photos, and find every photo that person is in.

Usage:
  python search.py --person refs/ --photos my_album/ --out results/
  python search.py --person me.jpg --photos my_album/ --out results/ --threshold 0.45 --gpu

Outputs in --out:
  matches/<photo>       copy of each matching photo with the person boxed in green
  results.csv           every photo: matched yes/no, similarity, face count
Embeddings for the album are cached (album_cache.npz) so new searches are instant.
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

from tagger import DEFAULT_THRESHOLD, IMG_EXTS, load_models, read_image, try_read_image, write_image

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
    """Average embedding of the largest face in each reference photo."""
    embs = []
    for p in list_images(ref_path):
        faces = app.get(read_image(p))
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


def index_album(app, photos_dir, cache_path, use_cache=True):
    """Detect + embed every face in the album once; reuse on later searches.
    Returns (index, unreadable_paths, n_indexed_now, seconds_spent_indexing)."""
    paths = list_images(photos_dir)
    cache = {}
    if use_cache and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        cache = d["index"].item()
    sigs = {p: file_sig(p) for p in paths}
    new = [p for p in paths if p not in cache or cache[p].get("sig") != sigs[p]]
    unreadable = []
    t0 = time.perf_counter()
    for i, p in enumerate(new, 1):
        img = try_read_image(p)
        if img is None:
            print(f"  [warn] could not read {p}")
            unreadable.append(p)
            cache.pop(p, None)
            continue
        faces = app.get(img)
        cache[p] = {
            "sig": sigs[p],
            "boxes": np.array([f.bbox for f in faces]).reshape(-1, 4),
            "embs": np.array([f.normed_embedding for f in faces]).reshape(-1, 512),
        }
        if i % 25 == 0 or i == len(new):
            print(f"  indexed {i}/{len(new)} new photos")
    elapsed = time.perf_counter() - t0
    if new or not os.path.exists(cache_path):
        np.savez(cache_path, index=np.array(cache, dtype=object))
    n_done = len(new) - len(unreadable)
    return {p: cache[p] for p in paths if p in cache}, unreadable, n_done, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="reference photo or folder of photos of the person")
    ap.add_argument("--photos", required=True, help="folder of photos to search")
    ap.add_argument("--out", default="results")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-index every photo (use for CPU vs GPU benchmarking)")
    a = ap.parse_args()

    # matches/ reflects only THIS search; otherwise hits from a previous person linger
    matches_dir = os.path.join(a.out, "matches")
    shutil.rmtree(matches_dir, ignore_errors=True)
    os.makedirs(matches_dir)

    t_load = time.perf_counter()
    app = load_models(use_gpu=a.gpu)
    t_load = time.perf_counter() - t_load
    ref = person_embedding(app, a.person)   # also warms up the models before the timed loop
    index, unreadable, n_indexed, t_index = index_album(
        app, a.photos, os.path.join(a.out, CACHE_NAME), use_cache=not a.no_cache)

    rows, n_match = [], 0
    for path in unreadable:
        rows.append((path, "unreadable", "", 0))
    for path, d in index.items():
        if len(d["embs"]) == 0:
            rows.append((path, "no", "", 0))
            continue
        sims = d["embs"] @ ref
        best = int(np.argmax(sims))
        hit = sims[best] >= a.threshold
        rows.append((path, "yes" if hit else "no", f"{sims[best]:.3f}", len(sims)))
        if hit:
            n_match += 1
            img = read_image(path)
            x1, y1, x2, y2 = d["boxes"][best].astype(int)
            t = max(2, max(img.shape[:2]) // 300)
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 200, 60), t)
            rel = os.path.relpath(path, a.photos) if os.path.isdir(a.photos) else os.path.basename(path)
            write_image(os.path.join(matches_dir, rel.replace(os.sep, "__")), img)

    rows.sort(key=lambda r: (r[1] != "yes", -(float(r[2]) if r[2] else -1)))
    with open(os.path.join(a.out, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["photo", "match", "best_similarity", "faces"])
        w.writerows(rows)

    print(f"\nperson found in {n_match} of {len(index)} photos:")
    for p, m, s, n in rows:
        if m == "yes":
            print(f"  {os.path.basename(p):<28} sim={s}  ({n} faces)")
    if unreadable:
        print(f"\n{len(unreadable)} unreadable file(s) listed in results.csv")
    print(f"\nmarked copies -> {a.out}/matches/   full list -> {a.out}/results.csv")

    device = "GPU (CUDA)" if a.gpu else "CPU"
    print(f"\ntiming [{device}]: model load {t_load:.1f}s")
    if n_indexed:
        print(f"  indexed {n_indexed} photo(s) in {t_index:.2f}s -> "
              f"{n_indexed / t_index:.1f} photos/s  ({1000 * t_index / n_indexed:.1f} ms/photo, "
              f"detect + embed)")
    else:
        print("  all photos served from cache, nothing indexed; use --no-cache to benchmark")


if __name__ == "__main__":
    main()
