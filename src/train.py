"""
Training script for the generator-invariant deepfake detector.

Usage (on Colab, after populating data/ as described in README.md):

    python src/train.py --data-root data --epochs 8 --adversarial 1
    python src/train.py --data-root data --epochs 8 --adversarial 0   # ablation: frozen CLIP, no GRL term

The --adversarial flag is what lets you reproduce the 3-row ablation table
from the methodology doc/PPT:
    row 1 (fusion baseline)      <- your earlier RGB+frequency model, not this script
    row 2 (frozen CLIP only)     <- this script with --adversarial 0
    row 3 (frozen CLIP + adv.)   <- this script with --adversarial 1
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import build_train_val_test
from model import GeneratorInvariantDetector, ModelConfig
from losses import compute_loss
from grl import grl_lambda_schedule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--clip-name", type=str, default="openai/clip-vit-base-patch16")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--adversarial", type=int, default=1, choices=[0, 1],
                    help="1 = train with the class-conditional adversarial term (proposed); "
                         "0 = frozen-CLIP baseline only, for the ablation table's middle row.")
    p.add_argument("--use-lora", type=int, default=1, choices=[0, 1],
                    help="1 = enable Low-Rank Adaptation (LoRA) on CLIP backbone; 0 = fully frozen backbone")
    p.add_argument("--sbi-ratio", type=float, default=0.0,
                    help="Fraction of training pool to synthesize as SBI pseudo-fakes (e.g. 0.3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="checkpoints")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_epoch(model, loader, optimizer, device, lambda_t, train: bool):
    model.train(mode=train and not model.cfg.freeze_backbone)
    # heads always need train() for dropout; backbone stays in eval() when frozen
    model.real_fake_head.train(train)
    model.generator_id_head.train(train)

    total_cls, total_adv, total_correct, total_n = 0.0, 0.0, 0, 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        real_fake_labels = batch["real_fake_label"].to(device)
        generator_labels = batch["generator_label"].to(device)
        is_sbi = batch.get("is_sbi", torch.zeros_like(real_fake_labels, dtype=torch.bool)).to(device)

        with torch.set_grad_enabled(train):
            rf_logits, gen_logits = model(pixel_values)
            out = compute_loss(
                rf_logits, gen_logits, real_fake_labels, generator_labels, lambda_t, is_sbi=is_sbi
            )

            if train:
                optimizer.zero_grad()
                out.total.backward()
                optimizer.step()

        total_cls += out.cls_loss.item() * pixel_values.size(0)
        total_adv += out.adv_loss.item() * pixel_values.size(0)
        preds = rf_logits.argmax(dim=1)
        total_correct += (preds == real_fake_labels).sum().item()
        total_n += pixel_values.size(0)

    return {
        "cls_loss": total_cls / total_n,
        "adv_loss": total_adv / total_n,
        "accuracy": total_correct / total_n,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Building datasets from {data_root} (sbi_ratio={args.sbi_ratio}) ...")
    train_ds, val_ds, test_ds = build_train_val_test(
        data_root, val_fraction=args.val_fraction, sbi_ratio=args.sbi_ratio, seed=args.seed, clip_name=args.clip_name
    )
    n_train_sbi = sum(1 for s in train_ds.samples if s.is_sbi)
    n_val_sbi = sum(1 for s in val_ds.samples if s.is_sbi)
    print(f"      train={len(train_ds)} (SBI={n_train_sbi})  val={len(val_ds)} (SBI={n_val_sbi})  test(OOD)={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"[2/4] Building model on {args.device} (adversarial={'on' if args.adversarial else 'off'}, lora={'on' if args.use_lora else 'off'}) ...")
    cfg = ModelConfig(
        clip_name=args.clip_name,
        num_generators=2,
        freeze_backbone=not bool(args.use_lora),
        use_lora=bool(args.use_lora),
    )
    model = GeneratorInvariantDetector(cfg).to(args.device)
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)

    print("[3/4] Training ...")
    history = []
    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        lambda_t = grl_lambda_schedule(progress) if args.adversarial else 0.0
        model.set_grl_lambda(lambda_t)

        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, args.device, lambda_t, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, args.device, lambda_t, train=False)
        dt = time.time() - t0

        row = {
            "epoch": epoch + 1, "lambda": round(lambda_t, 4),
            "train_cls_loss": round(train_metrics["cls_loss"], 4),
            "train_adv_loss": round(train_metrics["adv_loss"], 4),
            "train_acc": round(train_metrics["accuracy"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "seconds": round(dt, 1),
        }
        history.append(row)
        print(f"  epoch {row['epoch']:>2}/{args.epochs}  "
              f"lambda={row['lambda']:.3f}  "
              f"train_cls={row['train_cls_loss']:.4f}  train_adv={row['train_adv_loss']:.4f}  "
              f"train_acc={row['train_acc']:.4f}  val_acc={row['val_acc']:.4f}  "
              f"({row['seconds']}s)")

    tag = "adv" if args.adversarial else "baseline"
    ckpt_path = out_dir / f"model_{tag}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "history": history,
        "args": vars(args),
    }, ckpt_path)

    history_path = out_dir / f"history_{tag}.json"
    history_path.write_text(json.dumps(history, indent=2))

    print(f"[4/4] Done. Saved checkpoint to {ckpt_path} and history to {history_path}")
    print("      Next: run src/evaluate.py on this checkpoint to get ID/OOD accuracy, "
          "F1, ROC-AUC, and the generalization gap.")


if __name__ == "__main__":
    main()
