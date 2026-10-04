"""
Model: frozen CLIP ViT-B/16 encoder + two small heads.

  Head 1 (Real/Fake classifier)   — trained on ALL samples, normal gradient.
  Head 2 (Generator-ID classifier) — trained ONLY on fake-class samples,
                                      through a Gradient Reversal Layer.

Head 2's job is to correctly guess which generator made a fake image.
Because its gradient is reversed on the way back into the shared CLIP
features, training pushes those features to become features that a
classifier CANNOT use to tell generators apart — i.e. generator-invariant —
while Head 1 keeps them fully informative for the real/fake decision, since
Head 2 (and its reversed gradient) never sees real images at all.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from grl import GradientReversalLayer


@dataclass
class ModelConfig:
    clip_name: str = "openai/clip-vit-base-patch16"
    num_generators: int = 2          # e.g. {StyleGAN, DiFF-diffusion} seen at train time
    head_hidden_dim: int = 256
    dropout: float = 0.2
    freeze_backbone: bool = False
    use_lora: bool = True           # LoRA adapts CLIP to forensic signals without catastrophic forgetting
    lora_r: int = 8
    lora_alpha: int = 16


class SmallHead(nn.Module):
    """Two-layer MLP head used for both classification heads."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeneratorInvariantDetector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        base_backbone = CLIPVisionModel.from_pretrained(cfg.clip_name)
        feat_dim = base_backbone.config.hidden_size  # 768 for ViT-B/16

        if cfg.use_lora and not cfg.freeze_backbone:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            self.backbone = get_peft_model(base_backbone, lora_config)
        elif cfg.freeze_backbone:
            self.backbone = base_backbone
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        else:
            self.backbone = base_backbone

        self.real_fake_head = SmallHead(feat_dim, cfg.head_hidden_dim, 2, cfg.dropout)

        self.grl = GradientReversalLayer(lambda_=0.0)  # scheduled during training
        self.generator_id_head = SmallHead(
            feat_dim, cfg.head_hidden_dim, cfg.num_generators, cfg.dropout
        )

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract pooled CLIP features."""
        if self.cfg.freeze_backbone and not self.cfg.use_lora:
            with torch.no_grad():
                out = self.backbone(pixel_values=pixel_values)
        else:
            out = self.backbone(pixel_values=pixel_values)
        # pooler_output: CLS-token projection, shape (batch, hidden_size)
        return out.pooler_output

    def forward(self, pixel_values: torch.Tensor):
        features = self.encode(pixel_values)

        rf_logits = self.real_fake_head(features)

        reversed_features = self.grl(features)
        gen_logits = self.generator_id_head(reversed_features)

        return rf_logits, gen_logits

    def set_grl_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(lambda_)

    def trainable_parameters(self):
        """Returns trainable parameters (heads + LoRA parameters if enabled)."""
        params = list(self.real_fake_head.parameters()) + list(self.generator_id_head.parameters())
        if self.cfg.use_lora or not self.cfg.freeze_backbone:
            params += [p for p in self.backbone.parameters() if p.requires_grad]
        return params
