"""
Evaluation script for the RGB + frequency fusion baseline (Row 1).

Produces the same metrics as evaluate.py so the numbers can be directly
compared in the ablation table:
  - ID (val) accuracy, F1, ROC-AUC
  - OOD (Environment C) accuracy, F1, ROC-AUC
  - Generalization gap = ID accuracy - OOD accuracy
  - JPEG-compression robustness sweep on the OOD set

Usage:
    python src/evaluate_fusion.py --data-root data \
        --checkpoint checkpoints/model_fusion.pt \
        --jpeg-qualities 30 50 70
"""

import argparse
import io
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torchvision import transforms

from dataset import build_train_val_test
from fusion_model import FusionBaseline

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Dataset that applies optional JPEG corruption before the transform
# ---------------------------------------------------------------------------

class FusionEvalDataset(Dataset):
    def __init__(self, samples, jpeg_quality=None):
        self.samples      = samples
        self.jpeg_quality = jpeg_quality

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s.path).convert("RGB")

        if self.jpeg_quality is not None:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        return {
            "pixel_values":   EVAL_TRANSFORM(img),
            "real_fake_label": torch.tensor(s.real_fake_label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",      type=str, default="data")
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--batch-size",     type=int, default=32)
    p.add_argument("--num-workers",    type=int, default=2)
    p.add_argument("--jpeg-qualities", type=int, nargs="*", default=[30, 50, 70])
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str) -> FusionBaseline:
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FusionBaseline(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device):
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        pv     = batch["pixel_values"].to(device)
        labels = batch["real_fake_label"]
        logits = model(pv)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return all_labels, all_preds, all_probs


def compute_metrics(labels, preds, probs) -> dict:
    m = {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, zero_division=0),
    }
    try:
        m["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        m["roc_auc"] = float("nan")
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = args.device
    model  = load_model(args.checkpoint, device)

    _, val_ds_clip, test_ds_clip = build_train_val_test(
        Path(args.data_root), seed=42,
        clip_name="openai/clip-vit-base-patch16",
    )

    val_samples  = val_ds_clip.samples
    test_samples = test_ds_clip.samples

    val_loader  = DataLoader(FusionEvalDataset(val_samples),
                             batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(FusionEvalDataset(test_samples),
                             batch_size=args.batch_size, num_workers=args.num_workers)

    id_labels,  id_preds,  id_probs  = predict(model, val_loader,  device)
    ood_labels, ood_preds, ood_probs = predict(model, test_loader, device)

    id_metrics  = compute_metrics(id_labels,  id_preds,  id_probs)
    ood_metrics = compute_metrics(ood_labels, ood_preds, ood_probs)
    gap = id_metrics["accuracy"] - ood_metrics["accuracy"]

    result = {
        "checkpoint":         str(args.checkpoint),
        "id":                 id_metrics,
        "ood":                ood_metrics,
        "generalization_gap": gap,
    }

    if args.jpeg_qualities:
        result["jpeg_robustness"] = {}
        for q in args.jpeg_qualities:
            jl = DataLoader(FusionEvalDataset(test_samples, jpeg_quality=q),
                            batch_size=args.batch_size, num_workers=args.num_workers)
            jlabels, jpreds, jprobs = predict(model, jl, device)
            result["jpeg_robustness"][f"q{q}"] = compute_metrics(jlabels, jpreds, jprobs)["accuracy"]

    # Print
    print(f"\n=== Fusion Baseline (Row 1) ===")
    print(f"  ID  (val)         : acc={id_metrics['accuracy']:.4f}  "
          f"f1={id_metrics['f1']:.4f}  auc={id_metrics['roc_auc']:.4f}")
    print(f"  OOD (Env C)       : acc={ood_metrics['accuracy']:.4f}  "
          f"f1={ood_metrics['f1']:.4f}  auc={ood_metrics['roc_auc']:.4f}")
    print(f"  Generalization gap: {gap:+.4f}  (ID accuracy - OOD accuracy; smaller is better)")
    if "jpeg_robustness" in result:
        line = "  ".join(f"q{k[1:]}={v:.4f}" for k, v in result["jpeg_robustness"].items())
        print(f"  JPEG robustness   : {line}")

    # Save
    out_path = Path(args.checkpoint).with_suffix(".eval.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
