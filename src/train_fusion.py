"""
Training script for the RGB + Frequency fusion baseline (Row 1 of ablation table).

Mirrors the structure of train.py but WITHOUT any GRL / adversarial logic —
this model is a purely supervised binary classifier that we EXPECT to show
a large generalization gap (high ID accuracy, poor OOD accuracy), motivating
the class-conditional adversarial approach in Rows 2 & 3.

Usage (on Colab, after populating data/ per README.md):
    python src/train_fusion.py --data-root data --epochs 10

Saves checkpoint to checkpoints/model_fusion.pt and history to
checkpoints/history_fusion.json.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import build_train_val_test, DeepfakeEnvironmentDataset
from fusion_model import FusionBaseline


# ---------------------------------------------------------------------------
# ImageNet-style preprocessing (instead of CLIP's CLIPImageProcessor)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transform(augment: bool) -> transforms.Compose:
    ops = [transforms.Resize(256), transforms.CenterCrop(224)]
    if augment:
        ops = [transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
               transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class FusionDataset(torch.utils.data.Dataset):
    """Thin wrapper that applies torchvision transforms instead of CLIPImageProcessor."""

    def __init__(self, base_dataset: DeepfakeEnvironmentDataset, augment: bool = False):
        self.samples = base_dataset.samples
        self.transform = get_transform(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        from PIL import Image
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        pixel_values = self.transform(img)
        return {
            "pixel_values": pixel_values,
            "real_fake_label": torch.tensor(s.real_fake_label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train RGB + frequency fusion baseline (Row 1 of ablation table)."
    )
    p.add_argument("--data-root",    type=str,   default="data")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Lower LR than GRL scripts because ResNet branches are fine-tuned, not frozen.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--out-dir",      type=str,   default="checkpoints")
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    for batch in loader:
        pixel_values    = batch["pixel_values"].to(device)
        real_fake_labels = batch["real_fake_label"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(pixel_values)
            loss = torch.nn.functional.cross_entropy(logits, real_fake_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss    += loss.item() * pixel_values.size(0)
        total_correct += (logits.argmax(dim=1) == real_fake_labels).sum().item()
        total_n       += pixel_values.size(0)

    return {"loss": total_loss / total_n, "accuracy": total_correct / total_n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build datasets (reuse the same 3-env split, but wrap with our transform) ---
    print(f"[1/4] Building datasets from {data_root} ...")
    # We call build_train_val_test only to get the Sample lists, then rewrap.
    # Passing a dummy clip_name is fine — we'll override with FusionDataset.
    train_ds_clip, val_ds_clip, _ = build_train_val_test(
        data_root, val_fraction=args.val_fraction, seed=args.seed,
        clip_name="openai/clip-vit-base-patch16",
    )
    train_ds = FusionDataset(train_ds_clip, augment=True)
    val_ds   = FusionDataset(val_ds_clip,   augment=False)
    print(f"      train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # --- Model ---
    print(f"[2/4] Building FusionBaseline on {args.device} ...")
    model     = FusionBaseline(pretrained=True).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # --- Train ---
    print("[3/4] Training ...")
    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, optimizer, args.device, train=True)
        val_m   = run_epoch(model, val_loader,   optimizer, args.device, train=False)
        scheduler.step()
        dt = time.time() - t0

        row = {
            "epoch":      epoch + 1,
            "train_loss": round(train_m["loss"],     4),
            "train_acc":  round(train_m["accuracy"], 4),
            "val_acc":    round(val_m["accuracy"],   4),
            "seconds":    round(dt, 1),
        }
        history.append(row)
        print(f"  epoch {row['epoch']:>2}/{args.epochs}  "
              f"train_loss={row['train_loss']:.4f}  "
              f"train_acc={row['train_acc']:.4f}  "
              f"val_acc={row['val_acc']:.4f}  "
              f"({row['seconds']}s)")

    # --- Save ---
    ckpt_path = out_dir / "model_fusion.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history":          history,
        "args":             vars(args),
    }, ckpt_path)

    history_path = out_dir / "history_fusion.json"
    history_path.write_text(json.dumps(history, indent=2))

    print(f"[4/4] Done. Saved checkpoint -> {ckpt_path}")
    print("      Next: run src/evaluate_fusion.py to get ID/OOD accuracy and the gen. gap.")


if __name__ == "__main__":
    main()
