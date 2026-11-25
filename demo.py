"""
Gradio demo: upload a group photo -> who is in it (boxes + names) and CLIP scene tags.

  python demo.py --gallery-file gallery.npz --gpu            # http://127.0.0.1:7860
  python demo.py --gallery-file data/lfw_groups/gallery.npz --gpu --adapter models/adapter.npz
"""
import argparse

import cv2

import scene as scn
from tagger import (DEFAULT_THRESHOLD, adapter_id, detect_and_embed, draw, load_gallery,
                    load_models, match_faces, summary)


def build(app, scene_tagger, names, cents, threshold):
    import gradio as gr

    def analyze(img_rgb, thr):
        if img_rgb is None:
            return None, "Upload a photo.", {}, ""
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        faces = detect_and_embed(app, img)
        labels = match_faces(faces, names, cents, thr)
        boxed = cv2.cvtColor(draw(img, faces, labels), cv2.COLOR_BGR2RGB)
        rows = "\n".join(f"| {i + 1} | {n} | {s:.2f} |" for i, (n, s) in enumerate(labels))
        faces_md = f"**{summary(labels)}**\n\n| face | name | similarity |\n|---|---|---|\n{rows}" \
            if labels else "**No faces found**"
        p = scene_tagger.probs(img)
        scenes = {lb: float(p[i]) for i, lb in enumerate(scn.SCENES)}
        tags = scn.tags_from_probs(p)
        setting = f"indoor {p[-2]:.0%} / outdoor {p[-1]:.0%}"
        return boxed, faces_md, scenes, f"**Scene tags:** {', '.join(tags)}  \n{setting}"

    with gr.Blocks(title="Group photo tagger") as ui:
        gr.Markdown("# Group photo tagger\nFaces: SCRFD + ArcFace matched to the enrolled gallery. "
                    "Scenes: CLIP ViT-B/32 zero-shot.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="Photo")
                thr = gr.Slider(0.1, 0.8, value=threshold, step=0.01, label="Face match threshold")
                btn = gr.Button("Tag photo", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Tagged faces")
                faces_md = gr.Markdown()
                tags_md = gr.Markdown()
                scene_lbl = gr.Label(num_top_classes=5, label="Scene probabilities")
        btn.click(analyze, [inp, thr], [out_img, faces_md, scene_lbl, tags_md])
        inp.upload(analyze, [inp, thr], [out_img, faces_md, scene_lbl, tags_md])
    return ui, analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-file", default="gallery.npz")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--trt", action="store_true")
    ap.add_argument("--adapter", metavar="NPZ", help="gallery must be enrolled with the same adapter")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    a = ap.parse_args()
    app = load_models(use_gpu=a.gpu, use_trt=a.trt, adapter=a.adapter)
    names, cents = load_gallery(a.gallery_file, expect_adapter=adapter_id(app))
    scene_tagger = scn.SceneTagger(use_gpu=a.gpu, use_trt=a.trt)
    ui, _ = build(app, scene_tagger, names, cents, a.threshold)
    print(f"gallery: {len(names)} people from {a.gallery_file}")
    ui.launch(server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
