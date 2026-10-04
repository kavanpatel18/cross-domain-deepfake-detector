"""
Evaluation script.

Produces:
  - ID (validation) and OOD (Environment C) Accuracy, F1, ROC-AUC
  - Generalization gap = ID accuracy - OOD accuracy
  - A JPEG-compression robustness sweep on the OOD set
  - When checkpoints for both --adversarial 0 and --adversarial 1 runs are
    given, prints the 3-row ablation table (rows 2 and 3; row 1 is your
    earlier RGB+frequency fusion model and isn't produced by this script)

Usage:
    python src/evaluate.py --data-root data --checkpoint checkpoints/model_adv.pt
    python src/evaluate.py --data-root data \
        --checkpoint checkpoints/model_baseline.pt --compare checkpoints/model_adv.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset import build_train_val_test, DeepfakeEnvironmentDataset
from model import GeneratorInvariantDetector, ModelConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--compare", type=str, default=None,
                    help="Optional second checkpoint, to print the 2-row ablation comparison.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--jpeg-qualities", type=int, nargs="*", default=[30, 50, 70],
                    help="Quality levels for the robustness sweep; pass nothing to skip it.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str) -> GeneratorInvariantDetector:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = GeneratorInvariantDetector(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: GeneratorInvariantDetector, loader: DataLoader, device: str):
    all_labels, all_preds, all_probs = [], [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["real_fake_label"]
        rf_logits, _ = model(pixel_values)
        probs = torch.softmax(rf_logits, dim=1)[:, 1]  # P(fake)
        preds = rf_logits.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    return all_labels, all_preds, all_probs


def compute_metrics(labels, preds, probs) -> dict:
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        # Happens if a split has only one class present (e.g. tiny smoke-test data)
        metrics["roc_auc"] = float("nan")
    return metrics


def evaluate_checkpoint(checkpoint_path: str, data_root: Path, args) -> dict:
    device = args.device
    model = load_model(checkpoint_path, device)

    _, val_ds, test_ds = build_train_val_test(data_root, seed=42)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    id_labels, id_preds, id_probs = predict(model, val_loader, device)
    ood_labels, ood_preds, ood_probs = predict(model, test_loader, device)

    id_metrics = compute_metrics(id_labels, id_preds, id_probs)
    ood_metrics = compute_metrics(ood_labels, ood_preds, ood_probs)
    gap = id_metrics["accuracy"] - ood_metrics["accuracy"]

    result = {
        "checkpoint": str(checkpoint_path),
        "id": id_metrics,
        "ood": ood_metrics,
        "generalization_gap": gap,
    }

    if args.jpeg_qualities:
        result["jpeg_robustness"] = {}
        for q in args.jpeg_qualities:
            jpeg_ds = DeepfakeEnvironmentDataset(test_ds.samples, jpeg_quality=q)
            jpeg_loader = DataLoader(jpeg_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            j_labels, j_preds, j_probs = predict(model, jpeg_loader, device)
            result["jpeg_robustness"][f"q{q}"] = compute_metrics(j_labels, j_preds, j_probs)["accuracy"]

    return result


def print_result(name: str, result: dict):
    print(f"\n=== {name} ===")
    print(f"  ID  (val)         : acc={result['id']['accuracy']:.4f}  "
          f"f1={result['id']['f1']:.4f}  auc={result['id']['roc_auc']:.4f}")
    print(f"  OOD (Env C)       : acc={result['ood']['accuracy']:.4f}  "
          f"f1={result['ood']['f1']:.4f}  auc={result['ood']['roc_auc']:.4f}")
    print(f"  Generalization gap: {result['generalization_gap']:+.4f}  "
          f"(ID accuracy - OOD accuracy; smaller is better)")
    if "jpeg_robustness" in result:
        line = "  ".join(f"q{k[1:]}={v:.4f}" for k, v in result["jpeg_robustness"].items())
        print(f"  JPEG robustness   : {line}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    result_a = evaluate_checkpoint(args.checkpoint, data_root, args)
    print_result(Path(args.checkpoint).stem, result_a)

    out_path = Path(args.checkpoint).with_suffix(".eval.json")
    out_path.write_text(json.dumps(result_a, indent=2))
    print(f"\nSaved detailed results to {out_path}")

    if args.compare:
        result_b = evaluate_checkpoint(args.compare, data_root, args)
        print_result(Path(args.compare).stem, result_b)

        print("\n=== ABLATION TABLE (rows 2-3 — add your fusion baseline as row 1 by hand) ===")
        print(f"{'Configuration':<38}{'ID Acc':>10}{'OOD Acc':>10}{'Gap':>10}")
        for label, r in [(Path(args.checkpoint).stem, result_a), (Path(args.compare).stem, result_b)]:
            print(f"{label:<38}{r['id']['accuracy']:>10.4f}{r['ood']['accuracy']:>10.4f}"
                  f"{r['generalization_gap']:>+10.4f}")


if __name__ == "__main__":
    main()
