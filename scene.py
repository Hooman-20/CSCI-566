"""
Zero-shot scene tagging with CLIP (open_clip ViT-B/32, LAION-2B weights).

Each photo gets probabilities for two independent questions:
  scene    softmax over SCENES (concert, beach, jungle/forest, ...)
  setting  softmax over indoor / outdoor
They are separate softmaxes because "indoor" isn't an alternative to "concert": a concert is
also indoor or outdoor. Tags = up to 2 scenes with p >= SCENE_MIN_P, plus indoor or outdoor.

Runtime: the image encoder is exported once to ONNX and runs on onnxruntime (CUDA with --gpu,
after ort.preload_dlls(), like the face models). Text embeddings for the labels (prompt ensemble)
are computed once with open_clip on CPU torch. Both are cached in clip_cache/ (~350 MB).
"""
import hashlib
import json
import os

import cv2
import numpy as np

SCENES = ["concert", "beach", "jungle/forest", "wedding", "restaurant", "office", "street",
          "stage", "sports event", "party"]
SETTINGS = ["indoor", "outdoor"]
LABELS = SCENES + SETTINGS
SCENE_MIN_P = 0.3
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_cache")

# natural-language phrase per label, plugged into several templates (prompt ensembling)
_PHRASE = {
    "concert": "a concert", "beach": "the beach", "jungle/forest": "a jungle or forest",
    "wedding": "a wedding", "restaurant": "a restaurant", "office": "an office",
    "street": "a city street", "stage": "a stage performance", "sports event": "a sports event",
    "party": "a party",
}
_TEMPLATES = ["a photo of {}.", "a photo taken at {}.", "a group photo at {}.", "people at {}.",
              "a blurry photo of {}.", "a photo of people at {}."]
_SETTING_PROMPTS = {
    "indoor": ["a photo taken indoors.", "an indoor scene.", "a photo inside a room or building.",
               "people inside a building."],
    "outdoor": ["a photo taken outdoors.", "an outdoor scene.", "a photo outside under the open sky.",
                "people outside."],
}
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def prompts_for(label):
    if label in _SETTING_PROMPTS:
        return _SETTING_PROMPTS[label]
    return [t.format(_PHRASE[label]) for t in _TEMPLATES]


def labels_key():
    """Identifies model + labels + prompts; stored in the album cache to detect stale tags."""
    spec = json.dumps([MODEL, PRETRAINED, LABELS, [prompts_for(lb) for lb in LABELS]])
    return hashlib.sha1(spec.encode()).hexdigest()[:12]


def _build_assets(onnx_path, text_path):
    """One-time: download CLIP, export the image encoder to ONNX, embed the label prompts."""
    import open_clip
    import torch
    print(f"[scene] building CLIP assets in {CACHE_DIR} (one time) ...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval()
    tok = open_clip.get_tokenizer(MODEL)
    with torch.no_grad():
        T = []
        for lb in LABELS:
            e = model.encode_text(tok(prompts_for(lb)))
            e = e / e.norm(dim=-1, keepdim=True)
            m = e.mean(0)
            T.append((m / m.norm()).numpy())
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model.visual, (dummy,), onnx_path, input_names=["image"],
                          output_names=["embedding"], dynamic_axes={"image": {0: "batch"}},
                          opset_version=17, dynamo=False)
        ref = model.visual(dummy).numpy()
    np.savez(text_path, text=np.stack(T).astype(np.float32), key=labels_key(),
             logit_scale=float(model.logit_scale.exp()))
    import onnxruntime as ort
    out = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(
        None, {"image": dummy.numpy()})[0]
    assert np.abs(out - ref).max() < 1e-3, "ONNX export does not match torch"
    print("[scene] done")


class SceneTagger:
    def __init__(self, use_gpu=False, use_trt=False):
        """use_trt: TensorRT FP16 engine for a fixed 1x3x224x224 input (cached in trt_cache/)."""
        os.makedirs(CACHE_DIR, exist_ok=True)
        onnx_path = os.path.join(CACHE_DIR, f"{MODEL}_{PRETRAINED}_visual.onnx")
        text_path = os.path.join(CACHE_DIR, f"text_{labels_key()}.npz")
        if not (os.path.exists(onnx_path) and os.path.exists(text_path)):
            _build_assets(onnx_path, text_path)
        d = np.load(text_path)
        self.text, self.logit_scale = d["text"], float(d["logit_scale"])
        import onnxruntime as ort
        use_gpu = use_gpu or use_trt
        if use_gpu:
            ort.preload_dlls()   # CUDA/cuDNN DLLs from the nvidia-* wheels; else silent CPU fallback
        if use_trt:
            import tensorrt  # noqa: F401  (puts nvinfer_10 on the DLL path, see tagger.load_models)
            from tagger import _trt_options
            shape = "image:1x3x224x224"
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            options = [_trt_options(trt_profile_min_shapes=shape, trt_profile_opt_shapes=shape,
                                    trt_profile_max_shapes=shape), {}, {}]
        elif use_gpu:
            providers, options = ["CUDAExecutionProvider", "CPUExecutionProvider"], [{}, {}]
        else:
            providers, options = ["CPUExecutionProvider"], [{}]
        self.session = ort.InferenceSession(onnx_path, providers=providers, provider_options=options)
        want = providers[0]
        if want not in self.session.get_providers():
            raise RuntimeError(f"CLIP requested {want} but got {self.session.get_providers()}; "
                               "check onnxruntime-gpu / TensorRT install")
        self.key = labels_key()

    @staticmethod
    def preprocess(img_bgr):
        """open_clip's eval transform: resize short side to 224 (bicubic), center crop, normalize."""
        from PIL import Image   # PIL's bicubic antialiases like torchvision; cv2 drifts (cos ~0.98)
        h, w = img_bgr.shape[:2]
        s = 224 / min(h, w)
        nh, nw = max(224, int(h * s)), max(224, int(w * s))
        im = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).resize((nw, nh), Image.BICUBIC)
        y, x = int(round((nh - 224) / 2.0)), int(round((nw - 224) / 2.0))
        im = np.asarray(im, np.float32)[y:y + 224, x:x + 224] / 255.0
        return np.ascontiguousarray(((im - _MEAN) / _STD).transpose(2, 0, 1)[None])

    def probs(self, img_bgr):
        """[len(LABELS)] probabilities: scene part sums to 1, indoor+outdoor sums to 1."""
        return self.probs_from_input(self.preprocess(img_bgr))

    def probs_from_input(self, x):
        """Same as probs() for an already preprocessed [1,3,224,224] input (search.py runs
        preprocess() in its prefetch thread so it overlaps GPU work)."""
        e = self.session.run(None, {"image": x})[0][0]
        logits = self.logit_scale * (self.text @ (e / np.linalg.norm(e)))
        out = np.empty(len(LABELS), np.float32)
        for sl in (slice(0, len(SCENES)), slice(len(SCENES), len(LABELS))):
            z = np.exp(logits[sl] - logits[sl].max())
            out[sl] = z / z.sum()
        return out


def tags_from_probs(p, min_p=SCENE_MIN_P):
    """Up to 2 scene labels with p >= min_p, plus indoor or outdoor."""
    p = np.asarray(p)
    sc = p[:len(SCENES)]
    tags = [SCENES[i] for i in np.argsort(-sc)[:2] if sc[i] >= min_p]
    tags.append(SETTINGS[int(np.argmax(p[len(SCENES):]))])
    return tags


def matches_scene(p, wanted, min_p=SCENE_MIN_P):
    """True if every label in `wanted` applies: scenes need p >= min_p, indoor/outdoor > 0.5."""
    for lb in wanted:
        i = LABELS.index(lb)
        if p[i] < (0.5 if lb in SETTINGS else min_p):
            return False
    return True


def parse_scene_args(values):
    """Accept labels case-insensitively; 'jungle' or 'forest' -> 'jungle/forest', 'sports' ->
    'sports event'."""
    alias = {"jungle": "jungle/forest", "forest": "jungle/forest", "sports": "sports event",
             "sport": "sports event", "inside": "indoor", "outside": "outdoor"}
    out = []
    for v in values:
        v = v.strip().lower().replace("_", " ")
        v = alias.get(v, v)
        if v not in LABELS:
            raise SystemExit(f"unknown scene '{v}'; choose from: {', '.join(LABELS)}")
        out.append(v)
    return out
