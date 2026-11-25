"""
Train a residual MLP adapter on frozen ArcFace embeddings for degraded faces.

1. Data: LFW people that are NOT in the benchmark. make_groups.py uses every LFW person with
   >= --bench-min-faces (10) photos (enrolled AND distractors), so training only uses people
   with fewer photos. The script also checks against the gallery folders and test labels of each
   --bench dir and refuses to run if any identity overlaps.
2. For every photo: detect the main face, align it to 112x112, and embed the clean crop plus
   --k-degraded degraded versions (degrade.py, the same generator as make_groups.py --hard)
   with the frozen ArcFace R50 (onnxruntime, GPU). Cached in results/adapter/embeddings_k<K>.npz.
3. Train y = normalize(x + MLP(x)) (zero-initialised last layer, so it starts as the identity)
   with   CosFace(clean and degraded, train identities)
        + lambda_pull * (1 - cos(adapted degraded, clean ArcFace embedding))
        + lambda_id   * (1 - cos(adapted clean,    clean ArcFace embedding))
   The last two terms anchor the adapter to the original embedding space. That keeps clean
   accuracy, and it means templates/thresholds keep their meaning.
4. Validation: identities held out entirely (open-set, like the benchmark). Metric is top-1
   identification of a query photo among all val identities, templates = mean clean embedding of
   that person's OTHER photos (leave-one-out); reported for clean and degraded queries.

Usage:
  python train_adapter.py                       # build embeddings (first run) + train
  python train_adapter.py --epochs 30 --lambda-id 2
"""
import argparse
import json
import os
import time
from collections import Counter

import cv2
import numpy as np

LFW_DIR = os.path.expanduser("~/scikit_learn_data/lfw_home/lfw_funneled")
def cache_path(k):
    return f"results/adapter/embeddings_k{k}.npz"


# ---------------------------------------------------------------- data + identity split
def list_lfw():
    people = {}
    for name in sorted(os.listdir(LFW_DIR)):
        d = os.path.join(LFW_DIR, name)
        if os.path.isdir(d):
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
            if files:
                people[name] = [os.path.join(d, f) for f in files]
    return people


def benchmark_identities(people, bench_dirs, min_faces):
    """Every identity that can appear in a benchmark built by make_groups.py."""
    bench = {n for n, fs in people.items() if len(fs) >= min_faces}
    seen = set()
    for b in bench_dirs:
        g = os.path.join(b, "gallery")
        if os.path.isdir(g):
            seen |= set(os.listdir(g))
        lab = os.path.join(b, "test_labels.csv")
        if os.path.exists(lab):
            import csv
            with open(lab) as f:
                for row in csv.DictReader(f):
                    seen |= {p for p in row["people"].split(";") if p}
    return bench, seen


# ---------------------------------------------------------------- embedding extraction
def embed_crops(rec, crops, batch=16):
    """Frozen ArcFace on aligned 112x112 crops, fixed batch (padded) -> L2-normalized [N,512]."""
    blob = cv2.dnn.blobFromImages(crops, 1.0 / rec.input_std, rec.input_size,
                                  (rec.input_mean,) * 3, swapRB=True).astype(np.float32)
    out = []
    for i in range(0, len(blob), batch):
        chunk = blob[i:i + batch]
        n = len(chunk)
        if n < batch:
            chunk = np.concatenate([chunk, np.zeros((batch - n, *chunk.shape[1:]), np.float32)])
        out.append(rec.session.run(rec.output_names, {rec.input_name: chunk})[0][:n])
    e = np.concatenate(out)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def build_embeddings(train_people, k_degraded, seed):
    from insightface.utils import face_align

    import fastpath
    from degrade import degrade
    from make_groups import BystanderBlur
    from tagger import load_models, try_read_image

    app = load_models(use_gpu=True)
    det, rec = app.det_model, app.models["recognition"]
    names, clean, degr = [], [], []
    stats = Counter()
    t0 = time.time()
    items = [(n, p) for n in sorted(train_people) for p in train_people[n]]
    for idx, (name, path) in enumerate(items):
        img = try_read_image(path)
        dets, kpss = fastpath.detect(det, img)
        if len(dets) == 0:
            stats["no_face"] += 1
            continue
        m = BystanderBlur.main_face([d[:4] for d in dets], img.shape)
        box, kps = dets[m, :4], kpss[m]
        crops = [face_align.norm_crop(img, landmark=kps, image_size=112)]
        for k in range(k_degraded):
            rng = np.random.default_rng([seed, 2, idx, k])
            dimg, _ = degrade(img, box, rng)
            # align with landmarks found on the degraded photo, like at test time; if the
            # detector loses the face, fall back to the clean landmarks
            d2, k2 = fastpath.detect(det, dimg)
            best = max(range(len(d2)), key=lambda j: iou(d2[j], box), default=None)
            if best is not None and iou(d2[best], box) > 0.4:
                kp = k2[best]
                stats["deg_detected"] += 1
            else:
                kp = kps
                stats["deg_lost"] += 1
            crops.append(face_align.norm_crop(dimg, landmark=kp, image_size=112))
        e = embed_crops(rec, crops)
        names.append(name)
        clean.append(e[0])
        degr.append(e[1:])
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}/{len(items)} photos  ({time.time() - t0:.0f}s)")
    print(f"embedded {len(names)} photos x (1 clean + {k_degraded} degraded); "
          f"no face: {stats['no_face']}; degraded face re-detected {stats['deg_detected']}, "
          f"lost {stats['deg_lost']}")
    return np.array(names), np.stack(clean).astype(np.float32), np.stack(degr).astype(np.float32)


# ---------------------------------------------------------------- model
def make_model(hidden, scale, dropout=0.0):
    import torch
    import torch.nn as nn

    class ResidualAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(512, hidden)
            self.fc2 = nn.Linear(hidden, 512)
            self.drop = nn.Dropout(dropout)   # training only; inference (adapter.py) is unaffected
            nn.init.zeros_(self.fc2.weight)   # starts as the identity map
            nn.init.zeros_(self.fc2.bias)

        def forward(self, e):
            x = nn.functional.normalize(e, dim=-1) * scale
            y = x + self.fc2(self.drop(torch.relu(self.fc1(x))))
            return nn.functional.normalize(y, dim=-1)

    return ResidualAdapter()


class CosFace:
    def __init__(self, n_classes, s, m):
        import torch
        self.W = torch.nn.Parameter(torch.randn(n_classes, 512) * 0.01)
        self.s, self.m = s, m

    def __call__(self, emb, y):
        import torch
        import torch.nn.functional as F
        logits = emb @ F.normalize(self.W, dim=-1).T
        acc = (logits.argmax(1) == y).float().mean()
        logits = logits - self.m * F.one_hot(y, logits.shape[1])
        return F.cross_entropy(self.s * logits, y), acc


# ---------------------------------------------------------------- validation metrics
def loo_scores(q_emb, q_lab, q_img, c_emb, c_lab):
    """Query-vs-template cosine matrix and each query's own column. Templates = mean clean
    embedding per identity; the query's own template leaves out the query's own photo
    (q_img indexes into c_emb)."""
    ids = np.unique(c_lab)
    lab_i = {n: i for i, n in enumerate(ids)}
    ci = np.array([lab_i[n] for n in c_lab])
    sums = np.zeros((len(ids), c_emb.shape[1]), np.float64)
    np.add.at(sums, ci, c_emb)
    T = sums / np.linalg.norm(sums, axis=1, keepdims=True)
    S = q_emb @ T.T
    qi = np.array([lab_i[n] for n in q_lab])
    own = sums[qi] - c_emb[q_img]
    S[np.arange(len(qi)), qi] = np.sum(q_emb * own, 1) / np.linalg.norm(own, axis=1)
    return S, qi


def score_metrics(S, qi, far=1e-3):
    """top-1 identification, and TAR at a fixed FAR: the fraction of genuine (query, own template)
    scores above the threshold that only `far` of impostor scores exceed. The benchmark fails on
    degraded faces mostly because genuine scores drop below the threshold, so TAR@FAR is the
    metric that tracks it; top-1 is nearly saturated."""
    rows = np.arange(len(qi))
    gen = S[rows, qi]
    mask = np.ones_like(S, bool)
    mask[rows, qi] = False
    imp = S[mask]
    thr = np.quantile(imp, 1 - far)
    return float(np.mean(S.argmax(1) == qi)), float(np.mean(gen > thr)), float(thr)


def val_metrics(f, clean, degr, labels):
    """f: callable [N,512] -> [N,512]. Returns dict of clean/degraded top-1 and pull loss."""
    K = degr.shape[1]
    a_c = f(clean)
    a_d = f(degr.reshape(-1, 512))
    idx = np.arange(len(clean))
    t1c, tarc, _ = score_metrics(*loo_scores(a_c, labels, idx, a_c, labels))
    t1d, tard, _ = score_metrics(*loo_scores(a_d, np.repeat(labels, K), np.repeat(idx, K), a_c, labels))
    return {
        "top1_clean": t1c, "top1_degraded": t1d, "tar_clean": tarc, "tar_degraded": tard,
        "pull": float(np.mean(1 - np.sum(a_d * np.repeat(clean, K, 0), 1))),
    }


def split_identities(names, val_frac, rng):
    """Validation identities: a random val_frac of the people with >= 2 photos (needed for
    leave-one-out templates). eval_adapter.py calls this with default_rng(seed) to get the
    same split."""
    counts = Counter(names)
    multi = sorted(n for n, c in counts.items() if c >= 2)
    return set(rng.choice(multi, size=int(len(multi) * val_frac), replace=False))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=["data/lfw_groups", "data/lfw_groups_hard"])
    ap.add_argument("--bench-min-faces", type=int, default=10,
                    help="make_groups.py --min-faces used for the benchmark")
    ap.add_argument("--k-degraded", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true", help="re-extract embeddings")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of multi-photo identities")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0, help="on the hidden layer")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--lambda-pull", type=float, default=1.0)
    ap.add_argument("--lambda-id", type=float, default=1.0)
    ap.add_argument("--s", type=float, default=30.0, help="CosFace scale")
    ap.add_argument("--m", type=float, default=0.35, help="CosFace margin")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/adapter.npz")
    ap.add_argument("--curves", default="docs/adapter_curves.png")
    a = ap.parse_args()

    # ---- identities: no overlap with the benchmark, checked three ways
    people = list_lfw()
    bench, seen = benchmark_identities(people, a.bench, a.bench_min_faces)
    train_people = {n: fs for n, fs in people.items() if n not in bench}
    assert seen <= bench, f"benchmark has identities outside the >= {a.bench_min_faces} rule: " \
                          f"{sorted(seen - bench)[:5]}"
    overlap = set(train_people) & (bench | seen)
    assert not overlap, f"identity overlap with benchmark: {sorted(overlap)[:5]}"
    print(f"LFW: {len(people)} people. Benchmark identities (>= {a.bench_min_faces} photos): "
          f"{len(bench)}; seen in benchmark galleries/labels: {len(seen)}. "
          f"Training pool: {len(train_people)} people, "
          f"{sum(map(len, train_people.values()))} photos. Overlap: 0 (verified)")

    # ---- embeddings (cached)
    cache = cache_path(a.k_degraded)
    if a.rebuild or not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        names, clean, degr = build_embeddings(train_people, a.k_degraded, a.seed)
        np.savez(cache, names=names, clean=clean, degr=degr)
    d = np.load(cache)
    names, clean, degr = d["names"].astype(str), d["clean"], d["degr"]
    assert not (set(names) & bench), "cached embeddings contain benchmark identities; use --rebuild"
    K = degr.shape[1]

    # ---- identity-disjoint split
    rng = np.random.default_rng(a.seed)
    val_ids = split_identities(names, a.val_frac, rng)
    is_val = np.array([n in val_ids for n in names])
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    train_names = sorted(set(names[tr]))
    lab = {n: i for i, n in enumerate(train_names)}
    y_tr = np.array([lab[n] for n in names[tr]])
    print(f"train: {len(train_names)} identities, {len(tr)} photos x {K + 1} versions | "
          f"val: {len(val_ids)} unseen identities, {len(va)} photos")

    import torch
    torch.manual_seed(a.seed)
    model = make_model(a.hidden, float(np.sqrt(512)), a.dropout)
    head = CosFace(len(train_names), a.s, a.m)
    opt = torch.optim.AdamW(list(model.parameters()) + [head.W], lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = int(np.ceil(len(tr) * K / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch,
                                                pct_start=0.1)
    C_tr, D_tr = torch.from_numpy(clean[tr]), torch.from_numpy(degr[tr])
    Y_tr = torch.from_numpy(y_tr)

    def adapt_np(x):
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(np.ascontiguousarray(x))).numpy()

    base = val_metrics(lambda x: x, clean[va], degr[va], names[va])
    print(f"pretrained (no adapter) on val: {base}")
    hist = {k: [] for k in ["loss", "cls", "pull", "id", "train_acc", "val_pull",
                            "val_top1_clean", "val_top1_degraded", "val_tar_clean", "val_tar_degraded"]}
    best, best_state = -1.0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        # every (photo, degraded version) once per epoch, each paired with its clean version
        pairs = rng.permutation(len(tr) * K)
        sums = Counter()
        for s in range(steps_per_epoch):
            p = torch.from_numpy(pairs[s * a.batch:(s + 1) * a.batch])
            i, k = p // K, p % K
            c, dg, y = C_tr[i], D_tr[i, k], Y_tr[i]
            a_c, a_d = model(c), model(dg)
            l_cls, acc = head(torch.cat([a_c, a_d]), torch.cat([y, y]))
            l_pull = (1 - (a_d * c).sum(-1)).mean()
            l_id = (1 - (a_c * c).sum(-1)).mean()
            loss = a.lambda_cls * l_cls + a.lambda_pull * l_pull + a.lambda_id * l_id
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            n = len(i)
            for key, v in (("loss", loss), ("cls", l_cls), ("pull", l_pull), ("id", l_id),
                           ("train_acc", acc)):
                sums[key] += float(v.detach()) * n
            sums["n"] += n
        for key in ("loss", "cls", "pull", "id", "train_acc"):
            hist[key].append(sums[key] / sums["n"])
        vm = val_metrics(adapt_np, clean[va], degr[va], names[va])
        hist["val_pull"].append(vm["pull"])
        hist["val_top1_clean"].append(vm["top1_clean"])
        hist["val_top1_degraded"].append(vm["top1_degraded"])
        hist["val_tar_clean"].append(vm["tar_clean"])
        hist["val_tar_degraded"].append(vm["tar_degraded"])
        print(f"ep {ep:>2}  loss {hist['loss'][-1]:.3f} (cls {hist['cls'][-1]:.3f} pull "
              f"{hist['pull'][-1]:.3f} id {hist['id'][-1]:.4f})  train acc {hist['train_acc'][-1]:.3f} | "
              f"val pull {vm['pull']:.3f}  top1 clean/deg {vm['top1_clean']:.3f}/"
              f"{vm['top1_degraded']:.3f}  TAR@1e-3 clean/deg {vm['tar_clean']:.3f}/{vm['tar_degraded']:.3f}")
        # keep the best degraded TAR among epochs that don't hurt clean accuracy
        ok_clean = (vm["top1_clean"] >= base["top1_clean"] - 0.005
                    and vm["tar_clean"] >= base["tar_clean"] - 0.005)
        if ok_clean and vm["tar_degraded"] > best:
            best = vm["tar_degraded"]
            best_state = (ep, {k: v.detach().clone() for k, v in model.state_dict().items()})

    if best_state is None:
        print("[warn] no epoch kept val clean top-1/TAR within 0.5 pt of pretrained; saving last epoch")
        best_ep = a.epochs
    else:
        best_ep = best_state[0]
        model.load_state_dict(best_state[1])
    final = val_metrics(adapt_np, clean[va], degr[va], names[va])
    print(f"selected epoch {best_ep}: val {final}  (pretrained {base})")

    from adapter import save_adapter
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sd = model.state_dict()
    meta = {k: v for k, v in vars(a).items() if k not in ("bench",)}
    meta.update(epoch=best_ep, val=final, val_pretrained=base, train_identities=len(train_names),
                val_identities=len(val_ids))
    save_adapter(a.out, sd["fc1.weight"].numpy(), sd["fc1.bias"].numpy(), sd["fc2.weight"].numpy(),
                 sd["fc2.bias"].numpy(), np.sqrt(512), meta)
    print(f"saved adapter -> {a.out}")

    # numpy inference must match torch
    from adapter import Adapter
    diff = np.abs(Adapter(a.out)(clean[va[:100]]) - adapt_np(clean[va[:100]])).max()
    assert diff < 1e-4, diff

    plot_curves(hist, base, best_ep, a.curves)
    with open(os.path.splitext(a.curves)[0] + ".json", "w") as f:
        json.dump({"history": hist, "pretrained_val": base, "selected_epoch": best_ep,
                   "args": meta}, f, indent=1)


def plot_curves(hist, base, best_ep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, hist["cls"], label="CosFace (train)")
    ax[0].set_title("classification loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(ep, hist["pull"], label="pull (train)")
    ax[1].plot(ep, hist["val_pull"], label="pull (val, unseen people)")
    ax[1].axhline(base["pull"], ls="--", c="gray", label="pretrained (val)")
    ax[1].plot(ep, hist["id"], label="identity (train)")
    ax[1].set_title("1 - cos(adapted degraded, clean)")
    ax[1].set_xlabel("epoch")
    ax[2].plot(ep, hist["train_acc"], c="C0", label="train classification acc")
    ax[2].plot(ep, hist["val_tar_degraded"], c="C1", label="val TAR@FAR=1e-3, degraded")
    ax[2].plot(ep, hist["val_tar_clean"], c="C2", label="val TAR@FAR=1e-3, clean")
    ax[2].plot(ep, hist["val_top1_degraded"], c="C1", ls=":", label="val top-1, degraded")
    ax[2].axhline(base["tar_degraded"], ls="--", c="C1", alpha=0.6, label="pretrained TAR, degraded")
    ax[2].axhline(base["tar_clean"], ls="--", c="C2", alpha=0.6, label="pretrained TAR, clean")
    ax[2].set_title("accuracy (val = 302 unseen identities)")
    ax[2].set_xlabel("epoch")
    for x in ax:
        x.axvline(best_ep, c="k", lw=0.6, alpha=0.4)
        x.legend(fontsize=8)
        x.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"curves -> {path}")


if __name__ == "__main__":
    main()
