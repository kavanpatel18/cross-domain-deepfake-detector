"""
Gradio demo: upload a face image, get a Real/Fake prediction + confidence.

Usage:
    python src/app.py --checkpoint checkpoints/model_adv.pt

On Colab, this launches with share=True so you get a public link automatically.
"""

import argparse

import gradio as gr
import torch
from PIL import Image
from transformers import CLIPImageProcessor

from model import GeneratorInvariantDetector, ModelConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--clip-name", type=str, default="openai/clip-vit-base-patch16")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--share", action="store_true", help="Create a public Gradio link (useful on Colab).")
    return p.parse_args()


def build_predict_fn(model, processor, device):
    @torch.no_grad()
    def predict(image: Image.Image):
        if image is None:
            return {"Real": 0.0, "Fake": 0.0}
        pixel_values = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(device)
        rf_logits, _ = model(pixel_values)
        probs = torch.softmax(rf_logits, dim=1)[0]
        return {"Real": float(probs[0]), "Fake": float(probs[1])}
    return predict


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = GeneratorInvariantDetector(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    processor = CLIPImageProcessor.from_pretrained(args.clip_name)
    predict_fn = build_predict_fn(model, processor, args.device)

    demo = gr.Interface(
        fn=predict_fn,
        inputs=gr.Image(type="pil", label="Upload a face image"),
        outputs=gr.Label(num_top_classes=2, label="Prediction"),
        title="Cross-Domain Deepfake Detection",
        description=(
            "Frozen CLIP ViT-B/16 backbone + class-conditional adversarial training. "
            "Trained on StyleGAN and DiFF (diffusion) faces; this checkpoint's "
            "generalization to unseen forgery mechanisms is the whole point of the project — "
            "try a face-swap or reenactment image (e.g. from FaceForensics++) to see the "
            "out-of-distribution case."
        ),
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
