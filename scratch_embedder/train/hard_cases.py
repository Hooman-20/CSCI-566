"""Hard-case study: how does verification accuracy degrade for small / blurry / occluded / compressed
faces, and for profile and age variation, for OUR model vs. the pretrained baseline?

Degradations are applied to the second image of every verification pair (the "probe"), the first stays
clean, mimicking "good enrolled photo vs. bad photo in a group shot". Same pairs, same seed for every model.

  uv run python -m train.hard_cases --ckpt runs/ablation/base/best.pt \
      --degrade-bin <pack>/lfw.bin --extra-bins <pack>/cfp_fp.bin <pack>/agedb_30.bin \
      --onnx ~/.insightface/models/buffalo_l/w600k_r50.onnx          # optional pretrained baseline

Pose is covered by CFP-FP (frontal vs. profile), age by AgeDB-30; small faces, blur, occlusion and
compression by the synthetic degradations below.
"""
import argparse
import os

import cv2
import numpy as np
import torch

from train.eval_verification import embed_all, k_fold_accuracy, load_bin
from train.model import build_backbone

# ---------------------------------------------------------------- degradations (RGB uint8 112x112)


def lowres(img, size):
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (112, 112), interpolation=cv2.INTER_LINEAR)


def blur(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


def mask(img):                                      # surgical-mask style: covers nose-down
    out = img.copy(); out[62:, :] = (150, 190, 215); return out


def sunglasses(img):                                # eye band
    out = img.copy(); out[36:60, 10:102] = 20; return out


def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


CASES = {
    "clean":            lambda im: im,
    "small 56px":       lambda im: lowres(im, 56),
    "small 32px":       lambda im: lowres(im, 32),
    "small 24px":       lambda im: lowres(im, 24),
    "small 16px":       lambda im: lowres(im, 16),
    "blur sigma=2":     lambda im: blur(im, 2),
    "blur sigma=4":     lambda im: blur(im, 4),
    "mask (lower face)": mask,
    "sunglasses":       sunglasses,
    "jpeg q=20":        lambda im: jpeg(im, 20),
    "jpeg q=5":         lambda im: jpeg(im, 5),
}


def degrade_bin(imgs, fn):
    out = imgs.copy()
    for i in range(1, len(out), 2):                 # probe image of each pair
        out[i] = fn(out[i])
    return out


# ---------------------------------------------------------------- embedders: ours + ONNX baseline
def ours(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    net = build_backbone(ck["arch"], ck["emb_size"]).to(dev).eval()
    net.load_state_dict(ck["backbone"])
    return lambda imgs: embed_all(net, imgs, dev)


def onnx_baseline(path):
    """Pretrained InsightFace recognition model (e.g. buffalo_l/w600k_r50.onnx), same preprocessing."""
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):                    # lets onnxruntime-gpu find the CUDA/cuDNN libs shipped with torch
        try:
            import torch  # noqa: F401  (loads CUDA libs first)
            ort.preload_dlls()
        except Exception as e:
            print("preload_dlls failed:", e)
    ort.set_default_logger_severity(3)                  # silence the harmless "Expected shape {1,512}" warning per batch
    sess = ort.InferenceSession(os.path.expanduser(path),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    prov = sess.get_providers()
    print("onnxruntime providers:", prov)
    if prov[0] != "CUDAExecutionProvider":
        print("WARNING: baseline is running on CPU; ~260k forward passes of an R50 will take hours. "
              "Fix onnxruntime-gpu (or use a GPU runtime) before continuing.")
    fixed = sess.get_inputs()[0].shape[0]               # an int if the model was exported with a fixed batch
    default_batch = fixed if isinstance(fixed, int) else 128

    def fn(imgs, batch=default_batch):
        out = []
        for i in range(0, len(imgs), batch):
            x = (imgs[i:i + batch].astype(np.float32).transpose(0, 3, 1, 2) - 127.5) / 127.5
            x = np.ascontiguousarray(x)
            e = sess.run(None, {inp: x})[0] + sess.run(None, {inp: np.ascontiguousarray(x[:, :, :, ::-1])})[0]
            out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(out)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="one or more checkpoints; 'name=path' to label them (default label: parent folder name)")
    ap.add_argument("--degrade-bin", required=True, help="clean verification .bin to degrade (lfw.bin)")
    ap.add_argument("--extra-bins", nargs="*", default=[], help="evaluated as-is (cfp_fp.bin, agedb_30.bin)")
    ap.add_argument("--onnx", default=None, help="pretrained baseline recognition model (.onnx)")
    ap.add_argument("--out", default="runs/hard_cases.md")
    ap.add_argument("--dump-examples", default=None, help="folder to save a picture of each degradation (sanity check)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for c in a.ckpt:
        name, path = c.split("=", 1) if "=" in c else (os.path.basename(os.path.dirname(os.path.abspath(c))), c)
        models[name] = ours(path, dev)
    ours_names = list(models)
    if a.onnx:
        models["pretrained"] = onnx_baseline(a.onnx)

    lfw_imgs, lfw_same = load_bin(a.degrade_bin)
    if a.dump_examples:
        os.makedirs(a.dump_examples, exist_ok=True)
        for case, fn in CASES.items():
            strip = np.concatenate([fn(lfw_imgs[i]) for i in (1, 3, 5, 7)], axis=1)
            cv2.imwrite(os.path.join(a.dump_examples, case.replace(" ", "_").replace("=", "") + ".png"),
                        cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    # enrolled images (even index) never change: embed them once per model; only probes are re-embedded per case
    enrolled = {m: f(lfw_imgs[0::2]) for m, f in models.items()}
    rows = []
    for case, fn in CASES.items():
        probes = np.stack([fn(im) for im in lfw_imgs[1::2]])
        accs = {}
        for m, f in models.items():
            pe = f(probes)
            emb = np.empty((2 * len(pe), pe.shape[1]), dtype=pe.dtype)
            emb[0::2], emb[1::2] = enrolled[m], pe
            accs[m] = k_fold_accuracy(emb, lfw_same)[0]
        rows.append((f"LFW / {case}", accs))
    for p in a.extra_bins:
        imgs, same = load_bin(p)
        rows.append((os.path.basename(p).replace(".bin", ""),
                     {m: k_fold_accuracy(f(imgs), same)[0] for m, f in models.items()}))

    names = list(models)
    gaps = ours_names if "pretrained" in models else []          # gap column per model of ours, vs pretrained
    head = ["condition"] + names + [f"gap {n}" for n in gaps]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for cond, acc in rows:
        cells = [f"{acc[m]*100:.2f}" for m in names] + [f"{(acc[n] - acc['pretrained'])*100:+.2f}" for n in gaps]
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(table + "\n")
    print(f"\nwritten to {a.out} (verification accuracy %, probe image degraded)")


if __name__ == "__main__":
    main()
