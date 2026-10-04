"""
Row 1 baseline: RGB + Frequency-domain Fusion CNN.

Architecture:
  - Two parallel ResNet-18 branches:
      Branch 1: raw RGB image (224 × 224 × 3)
      Branch 2: log-magnitude FFT spectrum (same spatial size, 3-channel replicated)
  - Concatenate 512-dim features from each branch → 1024-dim
  - Small MLP head → Real / Fake logits

Why this is the 'naïve' baseline:
  Frequency artifacts (grid patterns, spectral peaks) are highly
  generator-specific. A frequency branch learns to recognise THOSE
  specific artifacts — not forgery in general. The result is a model
  that performs well in-distribution (ID) but fails badly when the
  generator changes (large generalization gap), which is exactly what
  Rows 2 & 3 are designed to fix.

Reference:
  Frank et al. (2020), "Leveraging Frequency Analysis for Deep Fake
  Image Recognition." ICML 2020.
"""

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Frequency branch pre-processing
# ---------------------------------------------------------------------------

def rgb_to_log_fft(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of RGB images (B, 3, H, W) to a 3-channel log-magnitude
    FFT representation of the same spatial size, normalised to [0, 1].

    Steps:
      1. Luminance-weight the RGB channels into a single grayscale plane.
      2. Compute the 2-D FFT and shift zero-frequency to the centre.
      3. Take log(1 + |magnitude|) to compress the dynamic range.
      4. Replicate across 3 channels so the frequency branch ResNet-18
         receives the same tensor shape as the RGB branch.
      5. Per-image normalise to [0, 1].
    """
    # (B, H, W) — BT.601 luminance weights
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]

    fft = torch.fft.fft2(gray)
    magnitude = torch.abs(torch.fft.fftshift(fft))      # (B, H, W)
    log_mag = torch.log1p(magnitude)                     # compress dynamic range

    # Replicate to 3 channels: (B, H, W) → (B, 3, H, W)
    log_mag = log_mag.unsqueeze(1).expand(-1, 3, -1, -1)

    # Per-image normalise to [0, 1]
    mn = log_mag.flatten(1).min(dim=1).values[:, None, None, None]
    mx = log_mag.flatten(1).max(dim=1).values[:, None, None, None]
    log_mag = (log_mag - mn) / (mx - mn + 1e-8)

    return log_mag


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FusionBaseline(nn.Module):
    """
    Dual-branch ResNet-18 that fuses RGB + frequency-domain features.

    Args:
        pretrained (bool): Load ImageNet-pretrained weights for both branches.
        head_hidden_dim (int): Hidden layer size in the classifier MLP.
        dropout (float): Dropout probability in the classifier.
    """

    def __init__(
        self,
        pretrained: bool = True,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        feat_dim = 512  # ResNet-18 penultimate-layer feature dim

        # ---- RGB branch ----
        rgb_enc = models.resnet18(weights=weights)
        self.rgb_branch = nn.Sequential(*list(rgb_enc.children())[:-1])  # drop FC

        # ---- Frequency branch ----
        freq_enc = models.resnet18(weights=weights)
        self.freq_branch = nn.Sequential(*list(freq_enc.children())[:-1])

        # ---- Fusion head ----
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, head_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 2),   # Real (0) vs. Fake (1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) — RGB image tensor, values in whatever
               range your preprocessing uses (typically ImageNet-normalised).
        Returns:
            logits: (B, 2) — unnormalised Real / Fake logits.
        """
        freq = rgb_to_log_fft(x)                        # (B, 3, H, W)

        rgb_feat  = self.rgb_branch(x).flatten(1)       # (B, 512)
        freq_feat = self.freq_branch(freq).flatten(1)   # (B, 512)

        fused = torch.cat([rgb_feat, freq_feat], dim=1) # (B, 1024)
        return self.classifier(fused)                   # (B, 2)
