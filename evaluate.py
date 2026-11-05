"""
Photo-level evaluation: for each test photo, compare the SET of people the model says
are present against the ground-truth set in a CSV (image,people with ';' separators).
Also sweeps the similarity threshold so you can pick one from data, not by guessing.
"""
import csv
import os

import cv2
import numpy as np

from tagger import draw, match_faces, read_image, summary, write_image


def load_labels(path):
    labels = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            people = [p.strip() for p in row["people"].split(";") if p.strip()]
            labels[row["image"]] = set(people)
    return labels


def score(pred_sets, true_sets):
    tp = sum(len(p & t) for p, t in zip(pred_sets, true_sets))
    fp = sum(len(p - t) for p, t in zip(pred_sets, true_sets))
    fn = sum(len(t - p) for p, t in zip(pred_sets, true_sets))
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    exact = np.mean([p == t for p, t in zip(pred_sets, true_sets)])
    return prec, rec, f1, exact


def run_eval(app, names, centroids, test_dir, labels_csv, out_dir, threshold):
    gt = load_labels(labels_csv)
    os.makedirs(out_dir, exist_ok=True)

    cache = []  # (image, faces, true_set) -> reuse detections for the threshold sweep
    for img_name in sorted(gt):
        img = read_image(os.path.join(test_dir, img_name))
        faces = app.get(img)
        cache.append((img_name, img, faces, gt[img_name]))

    print(f"\n{'image':<24}{'faces':>6}  predicted  |  truth")
    preds, truths = [], []
    for img_name, img, faces, truth in cache:
        labels = match_faces(faces, names, centroids, threshold)
        pred = {n for n, _ in labels if n != "unknown"}
        preds.append(pred)
        truths.append(truth)
        ok = "OK " if pred == truth else "ERR"
        if len(cache) <= 30 or pred != truth:   # big benchmarks: only show mistakes
            print(f"{ok} {img_name:<20}{len(faces):>6}  {sorted(pred)}  |  {sorted(truth)}")
            print(f"      -> {summary(labels)}")
        write_image(os.path.join(out_dir, os.path.splitext(img_name)[0] + "_tagged.jpg"),
                    draw(img, faces, labels))

    p, r, f1, ex = score(preds, truths)
    print(f"\nthreshold={threshold:.2f}  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}  "
          f"exact-photo-match={ex:.3f}")

    print("\nthreshold sweep:")
    print(f"{'thr':>6}{'prec':>8}{'rec':>8}{'F1':>8}{'exact':>8}")
    for thr in np.arange(0.15, 0.71, 0.05):
        ps = []
        for _, _, faces, _ in cache:
            labels = match_faces(faces, names, centroids, thr)
            ps.append({n for n, _ in labels if n != "unknown"})
        p, r, f1, ex = score(ps, truths)
        print(f"{thr:>6.2f}{p:>8.3f}{r:>8.3f}{f1:>8.3f}{ex:>8.3f}")
    print(f"\ntagged images written to {out_dir}/")
