"""Ablation study runner: trains one model per config with otherwise identical settings, then
builds a comparison table (best verification accuracy, params, training speed).

  uv run python -m train.ablate run --data data/casia_112 --val-bins <pack>/lfw.bin <pack>/cfp_fp.bin \
        --epochs 15 --max-ids 5000                      # run everything (hours; use --only to pick)
  uv run python -m train.ablate run ... --only arch_r18 loss_softmax
  uv run python -m train.ablate summarize                # table from runs/ablation/*/log.csv

Every run changes ONE thing relative to the base config (r34, ArcFace m=0.5, 512-d).
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys

BASE = {"arch": "r34", "loss": "arcface", "margin": "0.5", "emb-size": "512"}
GRID = {
    "base":         {},
    # backbone size: accuracy vs. cost
    "arch_r18":     {"arch": "r18"},
    "arch_r50":     {"arch": "r50"},
    # loss function: does the angular margin matter?
    "loss_softmax": {"loss": "softmax"},
    "loss_cosface": {"loss": "cosface", "margin": "0.35"},
    # ArcFace margin sweep
    "margin_0.3":   {"margin": "0.3"},
    "margin_0.7":   {"margin": "0.7"},
    # embedding size
    "emb_128":      {"emb-size": "128"},
}


def run(a):
    os.makedirs(a.root, exist_ok=True)
    for name in (a.only or GRID):
        cfg = {**BASE, **GRID[name]}
        out = os.path.join(a.root, name)
        if os.path.exists(os.path.join(out, "done.txt")) and not a.force:
            print(f"[skip] {name}: already finished (use --force to redo)")
            continue
        if a.force:
            shutil.rmtree(out, ignore_errors=True)      # --force = start this run from scratch
        # a run interrupted by a Colab disconnect has last.pt but no done.txt: continue it
        cmd = [sys.executable, "-m", "train.train", "--data", a.data, "--out", out,
               "--epochs", str(a.epochs), "--resume"]
        for k, v in cfg.items():
            cmd += [f"--{k}", v]
        if a.val_bins:
            cmd += ["--val-bins", *a.val_bins]
        cmd += a.extra
        print("\n=== " + name + ": " + " ".join(cmd))
        subprocess.run(cmd, check=True)
    summarize(a)


def count_params(arch, emb):
    from train.model import build_backbone
    return sum(p.numel() for p in build_backbone(arch, emb).parameters()) / 1e6


def summarize(a):
    rows = []
    for name in sorted(os.listdir(a.root)):
        log = os.path.join(a.root, name, "log.csv")
        if not os.path.exists(log):
            continue
        with open(log) as f:
            r = list(csv.DictReader(f))
        if not r:
            continue
        cfg = {**BASE, **GRID.get(name, {})}
        vcols = [c for c in r[0] if c.startswith("val_")]
        best = {c: max(float(x[c]) for x in r) for c in vcols}
        rows.append((name, cfg, count_params(cfg["arch"], int(cfg["emb-size"])),
                     sum(float(x["img_per_s"]) for x in r) / len(r), len(r), best, vcols))
    if not rows:
        print("no finished runs found")
        return
    vcols = rows[0][6]
    head = ["run", "arch", "loss", "margin", "emb", "params(M)", "img/s", "epochs"] + [c[4:] for c in vcols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, cfg, par, spd, ep, best, _ in rows:
        cells = [name, cfg["arch"], cfg["loss"], cfg["margin"] if cfg["loss"] != "softmax" else "-",
                 cfg["emb-size"], f"{par:.1f}", f"{spd:.0f}", str(ep)] + [f"{best[c]*100:.2f}" for c in vcols]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(a.root, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(f"\nwritten to {os.path.join(a.root, 'summary.md')} (verification accuracy in %)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--val-bins", nargs="*", default=[])
    r.add_argument("--epochs", type=int, default=15)
    r.add_argument("--only", nargs="*", choices=list(GRID))
    r.add_argument("--force", action="store_true")
    r.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to train.train")
    for p in (r, sub.add_parser("summarize")):
        p.add_argument("--root", default="runs/ablation")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summarize(a)


if __name__ == "__main__":
    main()
