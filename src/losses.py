"""
Combined loss for the generator-invariant detector.

    L = L_cls(rf_logits, real_fake_labels)                          [all samples]
      + lambda_t * L_adv(gen_logits[fake_mask], generator_labels[fake_mask])   [fake-only]

The fake-only masking is the one piece that isn't just "standard DANN":
a normal domain-adversarial setup would apply L_adv to every sample, but
here domain (generator identity) is undefined for real images (a real
image wasn't made by any generator), and — more importantly — domain and
label are almost perfectly confounded in this problem, so reversing
gradients from an all-sample domain discriminator risks washing out the
real/fake signal along with the generator signal. Restricting L_adv to
fake samples only means the adversarial term can never push back on how
the model separates real from fake.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


from typing import Optional


@dataclass
class LossOutput:
    total: torch.Tensor
    cls_loss: torch.Tensor
    adv_loss: torch.Tensor
    lambda_used: float
    n_fake_in_batch: int


def compute_loss(
    rf_logits: torch.Tensor,
    gen_logits: torch.Tensor,
    real_fake_labels: torch.Tensor,   # 0 = real, 1 = fake — shape (B,)
    generator_labels: torch.Tensor,   # generator-class id, ignored for real rows — shape (B,)
    lambda_t: float,
    is_sbi: Optional[torch.Tensor] = None,
) -> LossOutput:
    cls_loss = F.cross_entropy(rf_logits, real_fake_labels)

    if is_sbi is not None:
        fake_mask = (real_fake_labels == 1) & (~is_sbi.bool())
    else:
        fake_mask = (real_fake_labels == 1)
    n_fake = int(fake_mask.sum().item())

    if n_fake > 0:
        adv_loss = F.cross_entropy(gen_logits[fake_mask], generator_labels[fake_mask])
    else:
        # No non-SBI fake samples in this batch
        adv_loss = torch.zeros((), device=rf_logits.device, dtype=rf_logits.dtype)

    total = cls_loss + lambda_t * adv_loss

    return LossOutput(
        total=total,
        cls_loss=cls_loss.detach(),
        adv_loss=adv_loss.detach(),
        lambda_used=lambda_t,
        n_fake_in_batch=n_fake,
    )
