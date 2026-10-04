"""
Dataset for the 3-environment leave-one-out protocol.

Expected directory layout (see README.md for exactly how to populate this
from the public sources — 140k Real and Fake Faces, DiFF, FaceForensics++):

    data/
      environment_a_stylegan/
        real/*.jpg
        fake/*.jpg
      environment_b_diff/
        real/*.jpg
        fake/*.jpg
      environment_c_faceforensics/
        real/*.jpg
        fake/*.jpg

Environments A and B are TRAIN (their fake images get generator-id labels
0 and 1 respectively, used only for the adversarial head). Environment C
is the held-out OOD test set — a mechanism (face-swap/reenactment) never
seen during training, and its generator-id label is unused.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPImageProcessor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class Sample:
    path: Path
    real_fake_label: int      # 0 = real, 1 = fake
    generator_label: int      # meaningful only when real_fake_label == 1; -1 otherwise
    is_sbi: bool = False      # True for self-blended pseudo-fakes


def _list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def build_environment_samples(env_dir: Path, generator_label: int) -> list:
    """One environment folder -> list[Sample], using generator_label for its fakes."""
    samples = []
    for p in _list_images(env_dir / "real"):
        samples.append(Sample(p, real_fake_label=0, generator_label=-1, is_sbi=False))
    for p in _list_images(env_dir / "fake"):
        samples.append(Sample(p, real_fake_label=1, generator_label=generator_label, is_sbi=False))
    return samples


def build_sbi_samples(real_image_paths: list, n_samples: int) -> list:
    """
    Constructs n_samples SBI Sample references from available real images.
    Generation occurs dynamically on-the-fly inside __getitem__.
    """
    if not real_image_paths or n_samples <= 0:
        return []
    sampled_paths = random.choices(real_image_paths, k=n_samples)
    return [
        Sample(path=p, real_fake_label=1, generator_label=-1, is_sbi=True)
        for p in sampled_paths
    ]


class DeepfakeEnvironmentDataset(Dataset):
    """
    Wraps a list of Sample objects and applies CLIP's exact preprocessing
    (resize, center-crop, normalization) via CLIPImageProcessor, so the
    input distribution matches what the pretrained CLIP backbone expects.
    """

    def __init__(self, samples: list, clip_name: str = "openai/clip-vit-base-patch16",
                 augment: bool = False, jpeg_quality: Optional[int] = None):
        self.samples = samples
        self.processor = CLIPImageProcessor.from_pretrained(clip_name)
        self.augment = augment
        self.jpeg_quality = jpeg_quality  # set this (e.g. 30/50/70) for the robustness eval
        self._sbi_gen = None

    @property
    def sbi_gen(self):
        if self._sbi_gen is None:
            from sbi import SBIGenerator
            self._sbi_gen = SBIGenerator()
        return self._sbi_gen

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")

        if self.jpeg_quality is not None:
            # Robustness check: re-encode through JPEG at a given quality
            # before feeding the model, to test degradation under a common
            # real-world corruption instead of only clean images.
            import io
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        if self.augment:
            # 1. Random horizontal flip
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. Random JPEG recompression (forces model to ignore fragile high-frequency grid artifacts)
            if random.random() < 0.5:
                q = random.randint(35, 90)
                import io
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=q)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")

            # 3. Random Gaussian Blur (simulates post-processing / social media softening)
            if random.random() < 0.3:
                from PIL import ImageFilter
                radius = random.uniform(0.5, 1.8)
                img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        return img

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        if s.is_sbi:
            base_img = Image.open(s.path).convert("RGB")
            res = self.sbi_gen.generate(base_img)
            img = res[0] if res is not None else base_img
            if self.augment and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = self._load_image(s.path)

        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "real_fake_label": torch.tensor(s.real_fake_label, dtype=torch.long),
            "generator_label": torch.tensor(max(s.generator_label, 0), dtype=torch.long),
            "is_sbi": torch.tensor(s.is_sbi, dtype=torch.bool),
        }


def build_train_val_test(
    data_root: Path,
    val_fraction: float = 0.1,
    sbi_ratio: float = 0.0,
    seed: int = 42,
    clip_name: str = "openai/clip-vit-base-patch16",
):
    """
    Builds the three splits for the leave-one-out protocol:
      train / val  <- Environments A (label 0) + B (label 1), held-out val_fraction
      test (OOD)   <- Environment C entirely, generator label unused
    """
    data_root = Path(data_root)
    rng = random.Random(seed)

    env_a = build_environment_samples(data_root / "environment_a_stylegan", generator_label=0)
    env_b = build_environment_samples(data_root / "environment_b_diff", generator_label=1)
    env_c = build_environment_samples(data_root / "environment_c_faceforensics", generator_label=-1)

    if not env_a or not env_b:
        raise RuntimeError(
            "No images found in environment_a_stylegan/ or environment_b_diff/. "
            "See README.md for how to populate data/ before training."
        )
    if not env_c:
        raise RuntimeError(
            "No images found in environment_c_faceforensics/ — this is the OOD "
            "test set and training will run but you won't get a generalization "
            "gap number without it."
        )

    train_pool = env_a + env_b

    # Add Self-Blended Image (SBI) pseudo-fakes from training real images
    if sbi_ratio > 0.0:
        real_paths = [s.path for s in train_pool if s.real_fake_label == 0]
        n_sbi = int(len(train_pool) * sbi_ratio)
        sbi_samples = build_sbi_samples(real_paths, n_sbi)
        train_pool.extend(sbi_samples)
        print(f"      [SBI] Added {len(sbi_samples)} on-the-fly SBI pseudo-fakes to training pool.")

    rng.shuffle(train_pool)
    n_val = max(1, int(len(train_pool) * val_fraction))
    val_samples = train_pool[:n_val]
    train_samples = train_pool[n_val:]

    train_ds = DeepfakeEnvironmentDataset(train_samples, clip_name, augment=True)
    val_ds = DeepfakeEnvironmentDataset(val_samples, clip_name, augment=False)
    test_ds = DeepfakeEnvironmentDataset(env_c, clip_name, augment=False)

    return train_ds, val_ds, test_ds
