"""
Fine-tune the last stage(s) of ArcFace R50 (buffalo_l w600k_r50) for degraded faces, on CUDA.

Runs in the TRAINING venv (.venv-train: CUDA torch + onnx2torch). Inference stays in .venv on
onnxruntime; the result is exported back to ONNX and used with --model.

  python finetune_data.py                         # in .venv: identities, alignment, val crops
  .venv-train/Scripts/python train_finetune.py --stages 1
  python tagger.py eval ... --model models/arcface_r50_ft.onnx

Model: the ONNX graph is converted to PyTorch with onnx2torch (exact: max diff 1e-6 vs ORT).
--stages 1 trains stage 4 + the embedding head (~27M of 43.6M params); --stages 2 also stage 3.
Everything earlier is frozen, and BatchNorm stays in eval mode (statistics are not re-estimated
on 8k photos).

Data (results/finetune/data.npz): LFW people outside the benchmark, identity-disjoint
validation people, same split as train_adapter.py. Each step takes a photo, makes a degraded
copy with degrade.py (the same generator as the hard benchmark), and warps both with the
photo's alignment matrix plus small jitter (rotation/scale/shift, like landmark noise on
degraded faces) and a shared random flip.

Loss:  CosFace(clean + degraded crops, training identities; class weights start at each
       identity's pretrained mean embedding)
     + lambda_pull   * (1 - cos(student(degraded), teacher(clean)))
     + lambda_anchor * (1 - cos(student(clean),    teacher(clean)))
teacher = frozen copy of the original model. The anchor keeps clean embeddings, and therefore
clean accuracy, gallery templates and the 0.45 threshold, close to the original space.

Checkpoint: best val TAR@FAR=1e-3 on degraded queries among epochs whose clean top-1 and TAR stay
within 0.5 pt of pretrained (same rule as the adapter); validation people only.
"""
import argparse
import copy
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from degrade import degrade
from train_adapter import val_metrics

DATA = "results/finetune/data.npz"
ARCFACE = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")


def to_tensor(bgr):
    """ArcFace input: RGB, (x - 127.5) / 127.5, CHW (same as ArcFaceONNX.get_feat)."""
    x = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB).astype(np.float32)
    return torch.from_numpy(((x - 127.5) / 127.5).transpose(2, 0, 1).copy())


def jitter(M, rng):
    """Small random similarity transform around the crop centre, composed after M."""
    ang = np.deg2rad(rng.uniform(-4, 4))
    s = rng.uniform(0.95, 1.05)
    tx, ty = rng.uniform(-2, 2, 2)
    c = 56.0
    A = np.array([[s * np.cos(ang), -s * np.sin(ang), 0], [s * np.sin(ang), s * np.cos(ang), 0], [0, 0, 1]])
    T1 = np.array([[1, 0, -c], [0, 1, -c], [0, 0, 1]])
    T2 = np.array([[1, 0, c + tx], [0, 1, c + ty], [0, 0, 1]])
    return (T2 @ A @ T1 @ np.vstack([M, [0, 0, 1]]))[:2].astype(np.float32)


class PairSet(torch.utils.data.Dataset):
    """(clean crop, degraded crop, label), same geometry for both."""

    def __init__(self, paths, labels, boxes, Ms):
        self.paths, self.labels, self.boxes, self.Ms = paths, labels, boxes, Ms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        img = cv2.imdecode(np.fromfile(self.paths[i], np.uint8), cv2.IMREAD_COLOR)
        dimg, _ = degrade(img, self.boxes[i], rng)
        M = jitter(self.Ms[i], rng)
        c = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        d = cv2.warpAffine(dimg, M, (112, 112), borderValue=0.0)
        if rng.random() < 0.5:
            c, d = c[:, ::-1], d[:, ::-1]
        return to_tensor(c), to_tensor(d), int(self.labels[i])


def load_backbone():
    import onnx
    from onnx2torch import convert
    return convert(onnx.load(ARCFACE))   # (a path hits a Windows temp-file bug in onnx2torch)


def freeze_early(model, stages):
    """Freeze everything before the first conv of the last `stages` stages; returns its name."""
    leaves = [(n, m) for n, m in model.named_modules() if n and not list(m.children())]
    key = {1: (256, 512), 2: (128, 256)}[stages]
    i = next(j for j, (_, m) in enumerate(leaves)
             if isinstance(m, nn.Conv2d) and (m.in_channels, m.out_channels) == key)
    if i > 0 and isinstance(leaves[i - 1][1], nn.BatchNorm2d):   # the block's pre-conv BN
        i -= 1
    for j, (_, m) in enumerate(leaves):
        for p in m.parameters(recurse=False):
            p.requires_grad = j >= i
    return leaves[i][0]


@torch.no_grad()
def embed_crops(model, crops, bs=512):
    """crops: uint8 BGR [N,112,112,3] -> L2-normalized float32 [N,512]."""
    out = []
    for i in range(0, len(crops), bs):
        x = torch.stack([to_tensor(c) for c in crops[i:i + bs]]).cuda(non_blocking=True)
        out.append(F.normalize(model(x), dim=-1).float().cpu())
    return torch.cat(out).numpy()


def val_eval(model, va_crops, va_name):
    N, V = va_crops.shape[:2]
    e = embed_crops(model, va_crops.reshape(-1, 112, 112, 3)).reshape(N, V, 512)
    return val_metrics(lambda x: x, e[:, 0], e[:, 1:], va_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=1, choices=[1, 2], help="last N stages to train")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5, help="backbone")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--lambda-pull", type=float, default=1.0)
    ap.add_argument("--lambda-anchor", type=float, default=1.0)
    ap.add_argument("--s", type=float, default=30.0)
    ap.add_argument("--m", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/arcface_r50_ft.onnx")
    ap.add_argument("--curves", default="docs/finetune_curves.png")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    d = np.load(DATA)
    tr_name, va_name = d["tr_name"].astype(str), d["va_name"].astype(str)
    assert not (set(tr_name) & set(va_name)), "train/val identity overlap"
    classes = sorted(set(tr_name))
    lab = {n: i for i, n in enumerate(classes)}
    y = np.array([lab[n] for n in tr_name])
    va_crops = d["va_crops"]
    print(f"train {len(y)} photos / {len(classes)} identities | val {len(va_name)} photos / "
          f"{len(set(va_name))} unseen identities x {va_crops.shape[1]} versions")

    model = load_backbone().cuda().eval()
    first = freeze_early(model, a.stages)
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable from {first}: {n_tr / 1e6:.1f}M of "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # CosFace weights start at each identity's pretrained mean (clean) embedding
    t = d["tr_teacher"]
    W0 = np.zeros((len(classes), 512), np.float32)
    np.add.at(W0, y, t)
    W = nn.Parameter(F.normalize(torch.from_numpy(W0), dim=-1).cuda())

    loader = torch.utils.data.DataLoader(
        PairSet(d["tr_path"].astype(str), y, d["tr_box"], d["tr_M"]), batch_size=a.batch,
        shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True,
        persistent_workers=True)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr},
        {"params": [W], "lr": a.lr_head, "weight_decay": 0.0}], weight_decay=a.wd)
    total = a.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr_head], total_steps=total,
                                                pct_start=0.1)

    base = val_eval(teacher, va_crops, va_name)
    print(f"pretrained on val: {base}")
    hist = {k: [] for k in ["loss", "cls", "pull", "anchor", "train_acc", "val_pull",
                            "val_top1_clean", "val_top1_degraded", "val_tar_clean", "val_tar_degraded"]}
    best, best_ep, best_state = -1.0, None, None
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        sums = {k: 0.0 for k in ["loss", "cls", "pull", "anchor", "train_acc"]}
        n = 0
        for c, dg, yy in loader:
            c, dg, yy = c.cuda(non_blocking=True), dg.cuda(non_blocking=True), yy.cuda(non_blocking=True)
            with torch.no_grad():
                tc = F.normalize(teacher(c), dim=-1)
            s = F.normalize(model(torch.cat([c, dg])), dim=-1)
            sc, sd = s[:len(c)], s[len(c):]
            yy2 = torch.cat([yy, yy])
            logits = s @ F.normalize(W, dim=-1).T
            acc = (logits.argmax(1) == yy2).float().mean()
            l_cls = F.cross_entropy(a.s * (logits - a.m * F.one_hot(yy2, len(classes))), yy2)
            l_pull = (1 - (sd * tc).sum(1)).mean()
            l_anchor = (1 - (sc * tc).sum(1)).mean()
            loss = a.lambda_cls * l_cls + a.lambda_pull * l_pull + a.lambda_anchor * l_anchor
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k, v in (("loss", loss), ("cls", l_cls), ("pull", l_pull), ("anchor", l_anchor),
                         ("train_acc", acc)):
                sums[k] += float(v.detach()) * len(c)
            n += len(c)
        for k in sums:
            hist[k].append(sums[k] / n)
        vm = val_eval(model, va_crops, va_name)
        for k in ("pull", "top1_clean", "top1_degraded", "tar_clean", "tar_degraded"):
            hist["val_" + k].append(vm[k])
        print(f"ep {ep:>2} ({time.time() - t0:.0f}s) loss {hist['loss'][-1]:.3f} (cls {hist['cls'][-1]:.3f} "
              f"pull {hist['pull'][-1]:.3f} anchor {hist['anchor'][-1]:.4f}) acc {hist['train_acc'][-1]:.3f} | "
              f"val pull {vm['pull']:.3f} top1 clean/deg {vm['top1_clean']:.3f}/{vm['top1_degraded']:.3f} "
              f"TAR@1e-3 clean/deg {vm['tar_clean']:.3f}/{vm['tar_degraded']:.3f}", flush=True)
        ok_clean = (vm["top1_clean"] >= base["top1_clean"] - 0.005
                    and vm["tar_clean"] >= base["tar_clean"] - 0.005)
        if ok_clean and vm["tar_degraded"] > best:
            best, best_ep = vm["tar_degraded"], ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        print("[warn] no epoch kept clean val accuracy within 0.5 pt; exporting the last epoch")
        best_ep = a.epochs
    else:
        model.load_state_dict(best_state)
    final = val_eval(model, va_crops, va_name)
    print(f"selected epoch {best_ep}: val {final}  (pretrained {base})")

    # ---- export back to ONNX (same input name/layout as the original)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    m_cpu = model.cpu().eval()
    dummy = torch.randn(2, 3, 112, 112)
    torch.onnx.export(m_cpu, (dummy,), a.out, input_names=["input.1"], output_names=["embedding"],
                      dynamic_axes={"input.1": {0: "batch"}, "embedding": {0: "batch"}},
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = m_cpu(dummy).numpy()
    meta = {k: v for k, v in vars(a).items()}
    meta.update(selected_epoch=best_ep, val=final, val_pretrained=base, trainable_from=first,
                trainable_params=n_tr, history=hist)
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    np.savez(os.path.splitext(a.out)[0] + "_check.npz", x=dummy.numpy(), y=ref)   # for ORT check
    print(f"exported -> {a.out}")
    plot(hist, base, best_ep, a.curves)


def plot(hist, base, best_ep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = np.arange(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, hist["cls"], label="CosFace (train)")
    ax[0].set_title("classification loss")
    ax[1].plot(ep, hist["pull"], label="pull (train)")
    ax[1].plot(ep, hist["val_pull"], label="pull (val, unseen people)")
    ax[1].axhline(base["pull"], ls="--", c="gray", label="pretrained (val)")
    ax[1].plot(ep, hist["anchor"], label="anchor (train)")
    ax[1].set_title("1 - cos(student degraded, teacher clean)")
    ax[2].plot(ep, hist["train_acc"], c="C0", label="train classification acc")
    ax[2].plot(ep, hist["val_tar_degraded"], c="C1", label="val TAR@FAR=1e-3, degraded")
    ax[2].plot(ep, hist["val_tar_clean"], c="C2", label="val TAR@FAR=1e-3, clean")
    ax[2].plot(ep, hist["val_top1_degraded"], c="C1", ls=":", label="val top-1, degraded")
    ax[2].axhline(base["tar_degraded"], ls="--", c="C1", alpha=0.6, label="pretrained TAR, degraded")
    ax[2].axhline(base["tar_clean"], ls="--", c="C2", alpha=0.6, label="pretrained TAR, clean")
    ax[2].set_title("accuracy (val = 302 unseen identities)")
    for x in ax:
        x.set_xlabel("epoch")
        x.axvline(best_ep, c="k", lw=0.6, alpha=0.4)
        x.legend(fontsize=8)
        x.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"curves -> {path}")


if __name__ == "__main__":
    main()
