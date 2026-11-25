"""
Data for train_finetune.py, built with the inference stack (insightface detector, runs in .venv).

Same identities and validation split as train_adapter.py:
  train: LFW people NOT in the benchmark (checked: >= --bench-min-faces rule, benchmark galleries
         and labels), minus the validation people. For each photo: path, identity, main-face box
         and the 2x3 alignment matrix to the 112x112 ArcFace template. train_finetune.py degrades
         the full photo and warps it with this matrix (plus jitter) on the fly.
  val:   the identity-disjoint validation people of train_adapter.py, as fixed aligned crops: clean
         + K degraded per photo, produced exactly like the adapter's training embeddings
         (same seeds, landmarks re-detected on the degraded photo). Sanity check: pretrained
         ArcFace on these crops must reproduce the adapter cache embeddings.

Output: results/finetune/data.npz
Usage:  python finetune_data.py      (needs results/adapter/embeddings_k8.npz from train_adapter.py)
"""
import argparse
import os
import time

import numpy as np

OUT = "results/finetune/data.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=["data/lfw_groups", "data/lfw_groups_hard"])
    ap.add_argument("--bench-min-faces", type=int, default=10)
    ap.add_argument("--k-degraded", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from insightface.utils import face_align

    import fastpath
    from degrade import degrade
    from make_groups import BystanderBlur
    from tagger import load_models, try_read_image
    from train_adapter import (benchmark_identities, cache_path, embed_crops, iou, list_lfw,
                               split_identities)

    people = list_lfw()
    bench, seen = benchmark_identities(people, a.bench, a.bench_min_faces)
    train_people = {n: fs for n, fs in people.items() if n not in bench}
    assert seen <= bench and not (set(train_people) & (bench | seen)), "identity overlap"
    print(f"training pool {len(train_people)} people; overlap with {len(bench)} benchmark identities: 0")

    # validation split: identical to train_adapter.py (same names array, same rng)
    cache = np.load(cache_path(a.k_degraded))
    c_names, c_clean, c_degr = cache["names"].astype(str), cache["clean"], cache["degr"]
    val_ids = split_identities(c_names, a.val_frac, np.random.default_rng(a.seed))

    app = load_models(use_gpu=True)
    det, rec = app.det_model, app.models["recognition"]
    items = [(n, p) for n in sorted(train_people) for p in train_people[n]]   # same order as adapter
    tr_path, tr_name, tr_box, tr_M = [], [], [], []
    va_name, va_crops, kept = [], [], 0
    t0 = time.time()
    for idx, (name, path) in enumerate(items):
        img = try_read_image(path)
        dets, kpss = fastpath.detect(det, img)
        if len(dets) == 0:
            continue
        m = BystanderBlur.main_face([d[:4] for d in dets], img.shape)
        box, kps = dets[m, :4], kpss[m]
        assert c_names[kept] == name, "photo order differs from the adapter cache"
        kept += 1
        if name not in val_ids:
            tr_path.append(path)
            tr_name.append(name)
            tr_box.append(box)
            tr_M.append(face_align.estimate_norm(kps, 112))
            continue
        crops = [face_align.norm_crop(img, landmark=kps, image_size=112)]
        for k in range(a.k_degraded):          # exactly train_adapter.build_embeddings
            rng = np.random.default_rng([a.seed, 2, idx, k])
            dimg, _ = degrade(img, box, rng)
            d2, k2 = fastpath.detect(det, dimg)
            best = max(range(len(d2)), key=lambda j: iou(d2[j], box), default=None)
            kp = k2[best] if best is not None and iou(d2[best], box) > 0.4 else kps
            crops.append(face_align.norm_crop(dimg, landmark=kp, image_size=112))
        va_name.append(name)
        va_crops.append(np.stack(crops))
        if len(va_name) % 200 == 0:
            print(f"  val photos {len(va_name)}  ({time.time() - t0:.0f}s)")
    va_crops = np.stack(va_crops)          # [N, 1+K, 112, 112, 3] uint8 BGR

    # sanity: pretrained embeddings of the val crops == adapter cache embeddings
    va_mask = np.array([n in val_ids for n in c_names])
    e = embed_crops(rec, list(va_crops.reshape(-1, 112, 112, 3))).reshape(len(va_name), -1, 512)
    ref = np.concatenate([c_clean[va_mask][:, None], c_degr[va_mask]], 1)
    cos = np.sum(e * ref, -1)
    print(f"val crops reproduce adapter cache: min cos {cos.min():.6f}")
    assert cos.min() > 0.999

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, tr_path=np.array(tr_path), tr_name=np.array(tr_name),
             tr_box=np.array(tr_box, np.float32), tr_M=np.array(tr_M, np.float32),
             va_name=np.array(va_name), va_crops=va_crops,
             tr_teacher=c_clean[~va_mask], k_degraded=a.k_degraded, seed=a.seed)
    print(f"train {len(tr_path)} photos / {len(set(tr_name))} identities, val {len(va_name)} photos / "
          f"{len(set(va_name))} identities -> {OUT}")


if __name__ == "__main__":
    main()
