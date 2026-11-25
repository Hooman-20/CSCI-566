"""Face verification benchmark (LFW, CFP-FP, AgeDB-30, ...) from the InsightFace '.bin' packs.

A .bin is a pickle of (list_of_encoded_images, issame_list); images 2i and 2i+1 form pair i.
Protocol: 10-fold CV. On each fold the distance threshold is chosen on the other 9 folds and
accuracy is measured on the held-out fold. (Only unpickle files from the official dataset pack.)
"""
import pickle

import cv2
import numpy as np
import torch


def load_bin(path):
    with open(path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")
    imgs = np.stack([cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR),
                                  cv2.COLOR_BGR2RGB) for b in bins])      # (2N, 112, 112, 3) RGB
    return imgs, np.asarray(issame, dtype=bool)


@torch.no_grad()
def embed_all(model, imgs, device, batch=256):
    """Embeddings with flip test-time augmentation, L2-normalized."""
    model.eval()
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i:i + batch]).to(device).permute(0, 3, 1, 2).float()
        x = (x - 127.5) / 127.5
        e = model(x) + model(torch.flip(x, dims=[3]))
        out.append(torch.nn.functional.normalize(e, dim=1).cpu())
    return torch.cat(out).numpy()


def k_fold_accuracy(emb, issame, k=10):
    a, b = emb[0::2], emb[1::2]
    dist = np.sum((a - b) ** 2, axis=1)                      # squared L2 on unit vectors, in [0, 4]
    thresholds = np.arange(0, 4, 0.01)
    folds = np.array_split(np.arange(len(issame)), k)
    accs, best_thr = [], []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        tr_acc = [np.mean((dist[train] < t) == issame[train]) for t in thresholds]
        t = thresholds[int(np.argmax(tr_acc))]
        accs.append(np.mean((dist[test] < t) == issame[test]))
        best_thr.append(t)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(best_thr))


def evaluate(model, bin_data, device):
    imgs, issame = bin_data
    emb = embed_all(model, imgs, device)
    return k_fold_accuracy(emb, issame)


if __name__ == "__main__":
    import argparse
    from train.model import build_backbone

    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on verification .bin files")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bins", nargs="+", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=dev)
    net = build_backbone(ck["arch"], ck["emb_size"]).to(dev)
    net.load_state_dict(ck["backbone"])
    for p in a.bins:
        acc, std, thr = evaluate(net, load_bin(p), dev)
        print(f"{p}: acc={acc:.4f} +- {std:.4f}  (mean threshold {thr:.2f})")
