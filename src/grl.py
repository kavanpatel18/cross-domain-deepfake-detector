"""
Gradient Reversal Layer (GRL).

Forward pass: identity (passes features through unchanged).
Backward pass: multiplies the incoming gradient by -lambda_.

This is the standard mechanism from Ganin & Lempitsky (2015), "Unsupervised
Domain Adaptation by Backpropagation". Used here so that training the
generator-identity head to correctly classify which generator produced a
fake image simultaneously pushes the shared CLIP features to become WORSE
at revealing generator identity — i.e. more generator-invariant — without
touching the real/fake decision boundary, because this layer (and the loss
that flows through it) is only ever applied to fake-class samples.
"""

import torch
import torch.nn as nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        # Identity in the forward direction.
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse (and scale) the gradient on the way back.
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    nn.Module wrapper around the GRL autograd function, with a mutable
    `lambda_` so the reversal strength can be scheduled/ramped during
    training (standard DANN practice) without rebuilding the module.
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def extra_repr(self) -> str:
        return f"lambda_={self.lambda_:.4f}"


def grl_lambda_schedule(progress: float, gamma: float = 10.0) -> float:
    """
    Standard DANN ramp-up schedule for the GRL strength, from Ganin et al.

    `progress` should go from 0.0 (start of training) to 1.0 (end of
    training). Starts at 0 (no adversarial signal while the real/fake head
    is still finding its feet) and ramps smoothly to 1 (full adversarial
    strength) by the end of training. Using this instead of a constant
    lambda from step one avoids the adversarial term dominating early
    training and destabilizing the (much easier) real/fake objective.
    """
    progress = min(max(progress, 0.0), 1.0)
    return 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * progress)).item()) - 1.0
