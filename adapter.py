"""
Embedding adapter: a small residual MLP on top of the frozen ArcFace embedding, trained by
train_adapter.py to make degraded faces (blur, low resolution, occlusion, low light, JPEG) land
closer to the same person's clean embedding.

    y = normalize(x + W2 relu(W1 x + b1) + b2),   x = sqrt(512) * arcface_embedding

Inference is plain numpy, so tagger.py / search.py don't need torch. The adapter must be applied
to BOTH sides: enroll the gallery with --adapter, and search/tag/eval with the same --adapter.
"""
import hashlib
import json

import numpy as np


class Adapter:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.scale = float(d["scale"])
        self.meta = json.loads(str(d["meta"])) if "meta" in d else {}
        with open(path, "rb") as f:
            self.id = hashlib.sha1(f.read()).hexdigest()[:12]  # recorded in galleries enrolled with it

    def __call__(self, embs):
        """embs: [N,512] (any norm) -> adapted, L2-normalized [N,512] float32."""
        e = np.asarray(embs, np.float32).reshape(-1, self.W1.shape[1])
        x = e / np.linalg.norm(e, axis=1, keepdims=True) * self.scale
        y = x + np.maximum(x @ self.W1.T + self.b1, 0) @ self.W2.T + self.b2
        return (y / np.linalg.norm(y, axis=1, keepdims=True)).astype(np.float32)


def save_adapter(path, W1, b1, W2, b2, scale, meta):
    np.savez(path, W1=W1.astype(np.float32), b1=b1.astype(np.float32), W2=W2.astype(np.float32),
             b2=b2.astype(np.float32), scale=np.float32(scale), meta=json.dumps(meta))
