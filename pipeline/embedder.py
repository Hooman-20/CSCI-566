"""Drop-in embedder for the app: aligned 112x112 BGR faces -> L2-normalized 512-d vectors.

Use it after any detector/aligner, e.g. with InsightFace's detector + alignment:

    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import norm_crop
    det = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"]); det.prepare(ctx_id=-1)
    emb = Embedder("runs/r34_casia/best.pt")
    faces = det.get(img)
    vecs = emb.embed([norm_crop(img, f.kps, image_size=112) for f in faces])
"""
import cv2
import numpy as np
import torch

from train.model import build_backbone


class Embedder:
    def __init__(self, ckpt, device=None, flip_tta=True):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt, map_location=self.dev)
        self.net = build_backbone(ck["arch"], ck["emb_size"]).to(self.dev).eval()
        self.net.load_state_dict(ck["backbone"])
        self.flip_tta = flip_tta

    @torch.no_grad()
    def embed(self, faces_bgr, batch=128):
        if len(faces_bgr) == 0:
            return np.zeros((0, 512), np.float32)
        out = []
        for i in range(0, len(faces_bgr), batch):
            rgb = [cv2.cvtColor(cv2.resize(f, (112, 112)), cv2.COLOR_BGR2RGB) for f in faces_bgr[i:i + batch]]
            x = torch.from_numpy(np.stack(rgb)).to(self.dev).permute(0, 3, 1, 2).float()
            x = (x - 127.5) / 127.5                        # same normalization as training
            e = self.net(x)
            if self.flip_tta:
                e = e + self.net(torch.flip(x, dims=[3]))
            out.append(torch.nn.functional.normalize(e, dim=1).cpu())
        return torch.cat(out).numpy()
