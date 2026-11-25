"""
Pretrained ArcFace vs ArcFace + adapter on the clean and hard LFW group benchmarks.

Two comparisons:
  (1) both models at the default threshold 0.45
  (2) each model at its own threshold, chosen on the held-out VALIDATION people of
      train_adapter.py (never on the benchmark): open-set identification on those 302 unseen
      identities, 30% of them left out of the gallery as distractors, queries = clean photos
      + their degraded versions (equal weight); the threshold with the best F1 is used.
      One threshold per model, applied to both benchmarks, because a deployed tagger doesn't
      know whether a photo is "clean" or "hard".

Outputs: docs/adapter_results.md (tables + full threshold sweeps),
         docs/recall_vs_threshold_hard.png, docs/val_threshold_selection.png

Usage: python eval_adapter.py --adapter models/adapter.npz
"""
import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

from adapter import Adapter
from evaluate import load_labels, score
from train_adapter import cache_path, split_identities

GRID = np.round(np.arange(0.10, 0.801, 0.01), 2)


# ---------------------------------------------------------------- validation threshold
def val_curves(f, names, clean, degr, distract_frac, seed):
    """Weighted per-face precision/recall/F1 vs threshold on the validation people.
    Returns dict of arrays over GRID (combined, plus clean-only and degraded-only F1)."""
    K = degr.shape[1]
    a_c = f(clean)
    a_d = f(degr.reshape(-1, 512))
    ids = sorted(set(names))
    rng = np.random.default_rng([seed, 3])
    distract = set(rng.choice(ids, size=int(len(ids) * distract_frac), replace=False))
    enrolled = [n for n in ids if n not in distract]
    col = {n: i for i, n in enumerate(enrolled)}
    sums = np.zeros((len(enrolled), 512))
    for e, n in zip(a_c, names):
        if n in col:
            sums[col[n]] += e
    T = sums / np.linalg.norm(sums, axis=1, keepdims=True)

    idx = np.arange(len(clean))
    q = np.concatenate([a_c, a_d])
    q_img = np.concatenate([idx, np.repeat(idx, K)])
    q_lab = np.concatenate([names, np.repeat(names, K)])
    w = np.concatenate([np.full(len(a_c), float(K)), np.ones(len(a_d))])   # clean = degraded mass
    is_deg = np.concatenate([np.zeros(len(a_c), bool), np.ones(len(a_d), bool)])
    own = np.array([col.get(n, -1) for n in q_lab])
    S = q @ T.T
    m = own >= 0
    loo = sums[own[m]] - a_c[q_img[m]]      # own template without the query's own photo
    S[np.where(m)[0], own[m]] = np.sum(q[m] * loo, 1) / np.linalg.norm(loo, axis=1)
    best, top = S.argmax(1), S.max(1)

    out = {k: [] for k in ("precision", "recall", "f1", "f1_clean", "f1_degraded")}
    for thr in GRID:
        pred = np.where(top >= thr, best, -1)
        tp = m & (pred == own)
        fp = (pred >= 0) & (pred != own)
        fn = m & (pred != own)
        res = []
        for sel in (np.ones_like(m), ~is_deg, is_deg):
            TP, FP, FN = (w * (tp & sel)).sum(), (w * (fp & sel)).sum(), (w * (fn & sel)).sum()
            p = TP / (TP + FP) if TP + FP else 1.0
            r = TP / (TP + FN) if TP + FN else 1.0
            res.append((p, r, 2 * p * r / (p + r) if p + r else 0.0))
        out["precision"].append(res[0][0])
        out["recall"].append(res[0][1])
        out["f1"].append(res[0][2])
        out["f1_clean"].append(res[1][2])
        out["f1_degraded"].append(res[2][2])
    out = {k: np.array(v) for k, v in out.items()}
    out["n"] = (len(enrolled), len(distract), len(clean), len(a_d))
    return out


# ---------------------------------------------------------------- benchmark
def bench_embeddings(app, bench_dir, cache_path):
    """Raw (un-adapted) ArcFace embeddings: gallery per person (largest face per photo) and every
    face of every test photo. Cached, since both models share them."""
    from tagger import IMG_EXTS, detect_and_embed, read_image
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d["gallery"].item(), d["tests"].item()
    gallery = {}
    gdir = os.path.join(bench_dir, "gallery")
    for person in sorted(os.listdir(gdir)):
        embs = []
        for f in sorted(os.listdir(os.path.join(gdir, person))):
            if not f.lower().endswith(IMG_EXTS):
                continue
            faces = detect_and_embed(app, read_image(os.path.join(gdir, person, f)), adapt=False)
            if faces:
                fbig = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embs.append(fbig.normed_embedding)
        if embs:
            gallery[person] = np.stack(embs)
    tests = {}
    for img_name in sorted(load_labels(os.path.join(bench_dir, "test_labels.csv"))):
        faces = detect_and_embed(app, read_image(os.path.join(bench_dir, "test", img_name)), adapt=False)
        tests[img_name] = np.array([f.normed_embedding for f in faces]).reshape(-1, 512)
    np.savez(cache_path, gallery=np.array(gallery, dtype=object), tests=np.array(tests, dtype=object))
    return gallery, tests


def bench_sweep(f, gallery, tests, labels):
    """precision/recall/F1/exact over GRID; same rule as tagger.match_faces (Hungarian, then
    threshold), with the assignment computed once per photo since it doesn't depend on thr."""
    names = sorted(gallery)
    C = np.stack([f(gallery[n]).mean(0) for n in names])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    assigned, truths = [], []
    for img, E in tests.items():
        truths.append(labels[img])
        if len(E) == 0:
            assigned.append([])
            continue
        S = f(E) @ C.T
        r, c = linear_sum_assignment(-S)
        assigned.append([(names[j], S[i, j]) for i, j in zip(r, c)])
    rows = []
    for thr in GRID:
        preds = [{n for n, s in a if s >= thr} for a in assigned]
        rows.append(score(preds, truths))
    return np.array(rows)  # [len(GRID), 4] precision, recall, f1, exact


def at(sweep, thr):
    return sweep[int(np.argmin(np.abs(GRID - thr)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="models/adapter.npz")
    ap.add_argument("--clean", default="data/lfw_groups")
    ap.add_argument("--hard", default="data/lfw_groups_hard")
    ap.add_argument("--distractor-frac", type=float, default=0.3)
    ap.add_argument("--default-thr", type=float, default=0.45)
    ap.add_argument("--md", default="docs/adapter_results.md")
    ap.add_argument("--plot", default="docs/recall_vs_threshold_hard.png")
    ap.add_argument("--val-plot", default="docs/val_threshold_selection.png")
    a = ap.parse_args()

    adp = Adapter(a.adapter)
    seed, val_frac = adp.meta.get("seed", 0), adp.meta.get("val_frac", 0.2)
    models = {"pretrained": lambda x: np.asarray(x, np.float32), "adapter": adp}

    # ---- thresholds from the validation people
    d = np.load(cache_path(adp.meta.get("k_degraded", 8)))
    names, clean, degr = d["names"].astype(str), d["clean"], d["degr"]
    val_ids = split_identities(names, val_frac, np.random.default_rng(seed))
    va = np.array([n in val_ids for n in names])
    vc = {m: val_curves(f, names[va], clean[va], degr[va], a.distractor_frac, seed)
          for m, f in models.items()}
    thr = {m: float(GRID[int(np.argmax(c["f1"]))]) for m, c in vc.items()}
    ne, nd, nc, ndg = vc["pretrained"]["n"]
    print(f"validation: {ne} enrolled + {nd} distractor identities, {nc} clean + {ndg} degraded queries")
    for m in models:
        i = int(np.argmax(vc[m]["f1"]))
        print(f"  {m:<10} best val F1 {vc[m]['f1'][i]:.3f} at threshold {thr[m]:.2f}")

    # ---- benchmarks
    from tagger import load_models
    app = load_models(use_gpu=True)
    os.makedirs("results/adapter", exist_ok=True)
    sweeps = {}
    for bname, bdir in (("clean", a.clean), ("hard", a.hard)):
        labels = load_labels(os.path.join(bdir, "test_labels.csv"))
        gallery, tests = bench_embeddings(app, bdir, f"results/adapter/bench_{bname}.npz")
        for m, f in models.items():
            sweeps[(bname, m)] = bench_sweep(f, gallery, tests, labels)

    # ---- report
    L = ["## Pretrained vs adapter", "",
         f"Adapter `{a.adapter}` (id {adp.id}, epoch {adp.meta.get('epoch')}). Thresholds chosen on "
         f"the validation people: pretrained **{thr['pretrained']:.2f}**, adapter "
         f"**{thr['adapter']:.2f}** ({ne} enrolled + {nd} distractor unseen identities, "
         f"{nc} clean + {ndg} degraded queries, best F1).", "",
         "| benchmark | threshold rule | model | thr | precision | recall | F1 | exact |",
         "|---|---|---|---:|---:|---:|---:|---:|"]
    for bname in ("clean", "hard"):
        for rule in ("default", "val"):
            for m in models:
                t = a.default_thr if rule == "default" else thr[m]
                p, r, f1, ex = at(sweeps[(bname, m)], t)
                rl = f"default {a.default_thr:.2f}" if rule == "default" else "chosen on val people"
                L.append(f"| {bname} | {rl} | {m} | {t:.2f} | {p:.3f} | {r:.3f} | {f1:.3f} | {ex:.3f} |")
    L += ["", "### Threshold sweeps (benchmark; for reference only, thresholds above were NOT picked here)", ""]
    for bname in ("clean", "hard"):
        L += [f"**{bname}**", "",
              "| thr | pre P | pre R | pre F1 | pre exact | adp P | adp R | adp F1 | adp exact |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for t in np.round(np.arange(0.15, 0.701, 0.05), 2):
            pp, ad = at(sweeps[(bname, "pretrained")], t), at(sweeps[(bname, "adapter")], t)
            L.append(f"| {t:.2f} | " + " | ".join(f"{v:.3f}" for v in (*pp, *ad)) + " |")
        L.append("")
    L += ["### Validation-people sweep (used to choose the thresholds)", "",
          "| thr | pre F1 | pre F1 clean | pre F1 degraded | adp F1 | adp F1 clean | adp F1 degraded |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for t in np.round(np.arange(0.15, 0.701, 0.05), 2):
        i = int(np.argmin(np.abs(GRID - t)))
        v = [vc[m][k][i] for m in models for k in ("f1", "f1_clean", "f1_degraded")]
        L.append(f"| {t:.2f} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    os.makedirs(os.path.dirname(a.md) or ".", exist_ok=True)
    with open(a.md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L[:4 + 2 + 8 + 1]))
    print(f"\nfull report -> {a.md}")

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"pretrained": "C0", "adapter": "C3"}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in models:
        s = sweeps[("hard", m)]
        ax.plot(GRID, s[:, 1], c=colors[m], lw=2, label=f"{m}: recall")
        ax.plot(GRID, s[:, 0], c=colors[m], lw=1, ls="--", label=f"{m}: precision")
        ax.axvline(thr[m], c=colors[m], lw=1, ls=":", label=f"{m}: val-chosen thr {thr[m]:.2f}")
    ax.axvline(a.default_thr, c="gray", lw=1, label=f"default thr {a.default_thr:.2f}")
    ax.set_xlabel("cosine similarity threshold")
    ax.set_ylabel("photo-level precision / recall")
    ax.set_title("Hard LFW groups: recall vs threshold")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.plot, dpi=120)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in models:
        ax.plot(GRID, vc[m]["f1"], c=colors[m], lw=2, label=f"{m}: F1 (clean + degraded)")
        ax.plot(GRID, vc[m]["f1_degraded"], c=colors[m], lw=1, ls="--", label=f"{m}: F1 degraded only")
        ax.plot(GRID, vc[m]["f1_clean"], c=colors[m], lw=1, ls=":", label=f"{m}: F1 clean only")
        ax.axvline(thr[m], c=colors[m], lw=0.8, alpha=0.6)
    ax.set_xlabel("cosine similarity threshold")
    ax.set_ylabel("per-face F1")
    ax.set_title("Threshold selection on held-out validation people")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.val_plot, dpi=120)
    print(f"plots -> {a.plot}, {a.val_plot}")


if __name__ == "__main__":
    main()
