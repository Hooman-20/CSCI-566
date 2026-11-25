"""Train a face embedder with ArcFace loss.

  python -m train.train --data data/casia_112 --val-bins data/bins/lfw.bin data/bins/cfp_fp.bin \
         --arch r34 --epochs 25 --batch 128 --out runs/r34_casia

Checkpoints: <out>/last.pt (resumable), <out>/best.pt (best mean val accuracy), <out>/log.csv
"""
import argparse
import csv
import math
import os
import time

import torch
import torch.nn.functional as F

from train.arcface import build_head
from train.data import FaceFolder, make_loader
from train.eval_verification import evaluate, load_bin
from train.model import build_backbone


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder: <identity>/*.jpg, aligned 112x112")
    ap.add_argument("--val-bins", nargs="*", default=[], help="verification .bin files (lfw.bin ...)")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--arch", default="r34", choices=["r18", "r34", "r50"])
    ap.add_argument("--emb-size", type=int, default=512)
    ap.add_argument("--loss", default="arcface", choices=["arcface", "cosface", "softmax"],
                    help="classification head; --margin is the angular (arcface) / cosine (cosface) margin")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1, help="LR for batch 512; scaled linearly to --batch")
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--margin-warmup-epochs", type=float, default=2.0,
                    help="ramp margin 0 -> --margin; m=0.5 from step 0 often fails to converge from scratch")
    ap.add_argument("--scale", type=float, default=64.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-ids", type=int, default=None, help="use a random subset of identities (quick runs)")
    ap.add_argument("--max-iters", type=int, default=None, help="stop each epoch early (smoke tests)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="fine-tune: start from this checkpoint's backbone (+head if it has one); "
                    "ignored when --resume finds a last.pt in --out")
    ap.add_argument("--degrade-p", type=float, default=0.0,
                    help="probability of applying random low-res/blur/JPEG degradation to a training image")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda" and not a.no_amp
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)

    ds = FaceFolder(a.data, max_ids=a.max_ids, seed=a.seed, degrade_p=a.degrade_p)
    loader = make_loader(ds, a.batch, a.workers)
    print(f"{len(ds)} images, {ds.n_classes} identities, {len(loader)} iters/epoch, device={dev}, amp={amp}")
    bins = {os.path.basename(p).replace(".bin", ""): load_bin(p) for p in a.val_bins}

    net = build_backbone(a.arch, a.emb_size).to(dev)
    head = build_head(a.loss, a.emb_size, ds.n_classes, s=a.scale, m=a.margin).to(dev)
    peak_lr = a.lr * a.batch / 512
    opt = torch.optim.SGD([{"params": net.parameters()}, {"params": head.parameters()}],
                          lr=peak_lr, momentum=0.9, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best = 0, 0.0
    last_path = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=dev)
        net.load_state_dict(ck["backbone"]); head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start_epoch}")
    elif a.init:
        ck = torch.load(a.init, map_location=dev)
        assert ck["arch"] == a.arch and ck["emb_size"] == a.emb_size, "--init checkpoint has a different architecture"
        net.load_state_dict(ck["backbone"])
        try:                                               # head only loads if loss type and class count match
            if "head" not in ck or ck.get("loss") != a.loss:
                raise ValueError("no matching head")
            head.load_state_dict(ck["head"])
            print(f"initialised backbone + head from {a.init}")
        except (ValueError, RuntimeError):
            print(f"initialised backbone only from {a.init} (head starts fresh)")

    iters = len(loader) if a.max_iters is None else min(len(loader), a.max_iters)
    total, warm = a.epochs * iters, a.warmup_epochs * iters
    m_warm = max(1.0, a.margin_warmup_epochs * iters)

    log_path = os.path.join(a.out, "log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "loss", "train_acc", "lr", "margin", "img_per_s"]
                                   + [f"val_{k}" for k in bins])

    for epoch in range(start_epoch, a.epochs):
        net.train(); head.train()
        t0, seen, loss_sum, acc_sum, n_it = time.time(), 0, 0.0, 0.0, 0
        for it, (x, y) in enumerate(loader):
            if it >= iters:
                break
            step = epoch * iters + it
            lr = peak_lr * (step + 1) / warm if step < warm else \
                0.5 * peak_lr * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = lr
            head.set_margin(a.margin * min(1.0, step / m_warm))

            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                emb = net(x)
            logits, cos = head(emb, y)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt); scaler.update()

            seen += len(x); n_it += 1
            loss_sum += loss.item()
            acc_sum += (cos.argmax(1) == y).float().mean().item()
            if (it + 1) % 100 == 0:
                print(f"ep {epoch+1}/{a.epochs} it {it+1}/{iters} loss {loss_sum/n_it:.3f} "
                      f"acc {acc_sum/n_it:.3f} lr {lr:.4f} m {head.m:.2f} {seen/(time.time()-t0):.0f} img/s")

        speed = seen / (time.time() - t0)
        val = {k: evaluate(net, v, dev)[0] for k, v in bins.items()}
        mean_val = sum(val.values()) / len(val) if val else -loss_sum / max(1, n_it)
        print(f"== epoch {epoch+1}: loss {loss_sum/n_it:.3f} train_acc {acc_sum/n_it:.3f} "
              + " ".join(f"{k} {v:.4f}" for k, v in val.items()))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{loss_sum/n_it:.4f}", f"{acc_sum/n_it:.4f}", f"{lr:.5f}",
                                    f"{head.m:.3f}", f"{speed:.0f}"] + [f"{v:.4f}" for v in val.values()])

        meta = {"arch": a.arch, "emb_size": a.emb_size, "loss": a.loss}
        torch.save({**meta, "backbone": net.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": max(best, mean_val)}, last_path)
        if mean_val > best:
            best = mean_val
            torch.save({**meta, "backbone": net.state_dict(), "val": val, "epoch": epoch},
                       os.path.join(a.out, "best.pt"))
    with open(os.path.join(a.out, "done.txt"), "w") as f:      # completion marker (used by train.ablate)
        f.write(f"best_mean_val_acc={best:.4f}\n")
    print(f"done. best mean val acc: {best:.4f}")


if __name__ == "__main__":
    main()
