"""
Detect + embed path shared by tagger.py, search.py and evaluate.py.

Same math as insightface's FaceAnalysis.get (SCRFD detect -> 5-point norm_crop -> ArcFace),
re-written so that
  * every stage can be timed (Profiler) for a ms/photo breakdown,
  * all faces of a photo go through ArcFace in ONE batched call (batch=True) instead of
    one session.run per face,
  * images can be decoded in a background thread while the GPU works (iter_images).
"""
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from insightface.app.common import Face
from insightface.model_zoo.scrfd import distance2bbox, distance2kps
from insightface.utils import face_align

STAGES = ["decode", "det_pre", "det_infer", "det_post", "align", "embed", "overhead"]
STAGE_DESC = {
    "decode": "read file + JPEG decode (with prefetch: time spent waiting for the next image)",
    "det_pre": "resize + pad to 640x640 + normalize to NCHW blob",
    "det_infer": "SCRFD-10G session.run",
    "det_post": "anchor decode + score filter + NMS",
    "align": "5-point similarity warp to 112x112, per face",
    "embed": "ArcFace R50 preprocessing + session.run",
    "overhead": "Face objects, result bookkeeping, Python loop",
}
# ArcFace always runs with this exact batch size (chunks padded with zeros). onnxruntime's CUDA EP
# re-plans its cuDNN convolutions whenever the input shape changes: with a different face count
# per photo every call cost ~26 ms, versus 4.2 ms for a fixed batch of 16 (RTX 5090).
# The TensorRT session in tagger.load_models is built for this same fixed shape.
EMBED_BATCH = 16


class Profiler:
    """Accumulates wall time per stage. lap(stage) charges the time since the previous lap."""

    def __init__(self):
        self.totals = defaultdict(float)
        self.n_photos = 0
        self.n_faces = 0
        self._last = None

    def start(self):
        self._last = time.perf_counter()

    def lap(self, stage):
        now = time.perf_counter()
        self.totals[stage] += now - self._last
        self._last = now

    def total(self):
        return sum(self.totals.values())

    def table(self):
        """Markdown table of ms/photo per stage."""
        n = max(self.n_photos, 1)
        tot = self.total()
        lines = ["| stage | ms/photo | share | what |", "|---|---:|---:|---|"]
        for s in STAGES:
            t = self.totals.get(s, 0.0)
            lines.append(f"| {s} | {1000 * t / n:.2f} | {100 * t / tot:.1f}% | {STAGE_DESC[s]} |")
        lines.append(f"| **total** | **{1000 * tot / n:.2f}** | 100% | "
                     f"{n / tot:.1f} photos/s, {self.n_faces / n:.2f} faces/photo |")
        return "\n".join(lines)


class _NoTimer:
    def lap(self, stage):
        pass


NO_TIMER = _NoTimer()


# ---------------------------------------------------------------- detection
def _decode_outputs(det, net_outs, in_h, in_w, threshold):
    """SCRFD raw outputs -> per-stride (scores, boxes, kps) above threshold (from SCRFD.forward)."""
    scores_list, bboxes_list, kpss_list = [], [], []
    fmc = det.fmc
    for idx, stride in enumerate(det._feat_stride_fpn):
        if det.batched:
            scores = net_outs[idx][0]
            bbox_preds = net_outs[idx + fmc][0] * stride
            kps_preds = net_outs[idx + fmc * 2][0] * stride
        else:
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride
            kps_preds = net_outs[idx + fmc * 2] * stride
        h, w = in_h // stride, in_w // stride
        key = (h, w, stride)
        centers = det.center_cache.get(key)
        if centers is None:
            centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            if det._num_anchors > 1:
                centers = np.stack([centers] * det._num_anchors, axis=1).reshape((-1, 2))
            if len(det.center_cache) < 100:
                det.center_cache[key] = centers
        pos = np.where(scores >= threshold)[0]
        scores_list.append(scores[pos])
        bboxes_list.append(distance2bbox(centers, bbox_preds)[pos])
        kpss = distance2kps(centers, kps_preds)
        kpss_list.append(kpss.reshape((kpss.shape[0], -1, 2))[pos])
    return scores_list, bboxes_list, kpss_list


def detect(det, img, timer=NO_TIMER):
    """Equivalent to SCRFD.detect(img) with max_num=0. Returns (dets[N,5], kpss[N,5,2])."""
    assert det.use_kps, "alignment needs a detector with 5-point landmarks"
    threshold = getattr(det, "det_thresh", 0.5)
    pre_dets, pre_kpss = [], []
    for input_size in det._resolve_input_sizes(None):
        # --- pre: letterbox into the detector input, build blob
        im_ratio = img.shape[0] / img.shape[1]
        if im_ratio > input_size[1] / input_size[0]:
            new_h = input_size[1]
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_size[0]
            new_h = int(new_w * im_ratio)
        det_scale = new_h / img.shape[0]
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = cv2.resize(img, (new_w, new_h))
        blob = det._prepare_input_blob(det_img, tuple(input_size))
        session = det._session_for_input_size(tuple(input_size))
        timer.lap("det_pre")
        # --- infer
        net_outs = session.run(det.output_names, {det.input_name: blob})
        timer.lap("det_infer")
        # --- post (per size)
        scores_l, bboxes_l, kpss_l = _decode_outputs(det, net_outs, blob.shape[2], blob.shape[3],
                                                     threshold)
        if sum(s.size for s in scores_l):
            scores = np.vstack(scores_l)
            order = np.argsort(-scores.ravel(), kind="stable")
            bboxes = np.vstack(bboxes_l) / det_scale
            kpss = np.vstack(kpss_l) / det_scale
            pre_dets.append(np.hstack((bboxes, scores)).astype(np.float32, copy=False)[order])
            pre_kpss.append(kpss[order])
        timer.lap("det_post")
    if not pre_dets:
        return np.empty((0, 5), np.float32), np.empty((0, 5, 2), np.float32)
    pre_det = np.vstack(pre_dets).astype(np.float32, copy=False)
    order = np.argsort(-pre_det[:, 4], kind="stable")
    pre_det = pre_det[order]
    kpss = np.vstack(pre_kpss)[order]
    keep = det.nms(pre_det)
    timer.lap("det_post")
    return pre_det[keep], kpss[keep]


# ---------------------------------------------------------------- recognition
def embed(rec, img, kpss, timer=NO_TIMER, batch=True):
    """Align every face and return raw ArcFace embeddings [N,512]."""
    crops = [face_align.norm_crop(img, landmark=k, image_size=rec.input_size[0]) for k in kpss]
    timer.lap("align")
    if not batch:
        feats = np.concatenate([rec.get_feat(c) for c in crops])
        timer.lap("embed")
        return feats
    # same preprocessing as ArcFaceONNX.get_feat, then pad to a fixed batch of EMBED_BATCH
    blob = cv2.dnn.blobFromImages(crops, 1.0 / rec.input_std, rec.input_size,
                                  (rec.input_mean,) * 3, swapRB=True)
    blob = np.ascontiguousarray(blob, dtype=np.float32)
    feats = []
    for i in range(0, len(blob), EMBED_BATCH):
        chunk = blob[i:i + EMBED_BATCH]
        n = len(chunk)
        if n < EMBED_BATCH:
            chunk = np.concatenate([chunk, np.zeros((EMBED_BATCH - n, *chunk.shape[1:]), np.float32)])
        feats.append(rec.session.run(rec.output_names, {rec.input_name: chunk})[0][:n])
    timer.lap("embed")
    return np.concatenate(feats)


def detect_and_embed(app, img, timer=NO_TIMER, batch=True):
    """Faces with .bbox, .kps, .det_score, .embedding / .normed_embedding (like FaceAnalysis.get)."""
    dets, kpss = detect(app.det_model, img, timer)
    if len(dets) == 0:
        return []
    feats = embed(app.models["recognition"], img, kpss, timer, batch=batch)
    faces = []
    for d, k, e in zip(dets, kpss, feats):
        f = Face(bbox=d[:4], kps=k, det_score=d[4])
        f.embedding = e
        faces.append(f)
    timer.lap("overhead")
    return faces


# ---------------------------------------------------------------- input pipeline
def iter_images(paths, read_fn, prefetch=True, depth=4):
    """Yield (path, image_or_None). With prefetch, a background thread decodes up to `depth`
    images ahead (cv2.imdecode releases the GIL), overlapping decoding with GPU inference."""
    if not prefetch:
        for p in paths:
            yield p, read_fn(p)
        return
    it = iter(paths)
    with ThreadPoolExecutor(max_workers=1) as ex:
        pending = deque()
        for p in it:
            pending.append((p, ex.submit(read_fn, p)))
            if len(pending) >= depth:
                break
        while pending:
            p, fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, ex.submit(read_fn, nxt)))
            yield p, fut.result()
