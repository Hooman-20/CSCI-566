"""
Random real-world degradations for a face photo, shared by make_groups.py --hard (benchmark)
and train_adapter.py (training data).

degrade(img, face_box, rng) applies a random subset (at least one) of:
  blur       gaussian blur, sigma 1.5-4% of the face width
  lowres     downscale so the face is 24-48 px wide, then upscale back (lost detail)
  occlusion  synthetic sunglasses (dark band over the eyes) or mask (lower face)
  dark       low brightness: x0.2-0.45 plus gamma, like an underexposed photo
  jpeg       JPEG compression at quality 8-25
The face box (x1, y1, x2, y2) only sets the scale and where occlusions go.
"""
import cv2
import numpy as np

KINDS = ("blur", "lowres", "occlusion", "dark", "jpeg")
P_EACH = 0.5


def default_face_box(shape):
    """LFW photos are centred on the labeled face at a fairly fixed scale; used when no
    detection is available."""
    h, w = shape[:2]
    return np.array([0.3 * w, 0.28 * h, 0.7 * w, 0.78 * h], np.float32)


def _occlude(img, box, rng):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    out = img.copy()
    color = tuple(int(c) for c in rng.integers(0, 45, 3))
    if rng.random() < 0.5:   # sunglasses: band over both eyes
        top, bot = y1 + bh * rng.uniform(0.25, 0.32), y1 + bh * rng.uniform(0.47, 0.55)
        left, right = x1 - 0.05 * bw, x2 + 0.05 * bw
        kind = "sunglasses"
    else:                    # mask: nose tip down to chin
        top, bot = y1 + bh * rng.uniform(0.55, 0.62), y2 + 0.05 * bh
        left, right = x1 - 0.02 * bw, x2 + 0.02 * bw
        color = tuple(int(c) for c in rng.integers(150, 240, 3)) if rng.random() < 0.5 else color
        kind = "mask"
    H, W = img.shape[:2]
    p1 = (int(max(0, left)), int(max(0, top)))
    p2 = (int(min(W - 1, right)), int(min(H - 1, bot)))
    cv2.rectangle(out, p1, p2, color, -1)
    return out, kind


def degrade(img, face_box, rng, kinds=None):
    """Return (degraded image, list of applied degradation names)."""
    box = np.asarray(face_box, np.float32)
    face_w = max(8.0, float(box[2] - box[0]))
    if kinds is None:
        kinds = [k for k in KINDS if rng.random() < P_EACH]
        if not kinds:
            kinds = [KINDS[int(rng.integers(len(KINDS)))]]
    out = img
    applied = []
    if "occlusion" in kinds:
        out, kind = _occlude(out, box, rng)
        applied.append(kind)
    if "blur" in kinds:
        sigma = face_w * rng.uniform(0.015, 0.04)
        out = cv2.GaussianBlur(out, (0, 0), sigma)
        applied.append("blur")
    if "lowres" in kinds:
        target = rng.uniform(24, 48)          # face width in pixels after downscaling
        f = min(1.0, target / face_w)
        h, w = out.shape[:2]
        small = cv2.resize(out, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        applied.append("lowres")
    if "dark" in kinds:
        gain, gamma = rng.uniform(0.2, 0.45), rng.uniform(1.0, 1.6)
        x = (out.astype(np.float32) / 255.0) ** gamma * gain
        out = np.clip(x * 255 + rng.normal(0, 2.0, out.shape), 0, 255).astype(np.uint8)
        applied.append("dark")
    if "jpeg" in kinds:
        q = int(rng.integers(8, 26))
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        applied.append("jpeg")
    return out, applied
