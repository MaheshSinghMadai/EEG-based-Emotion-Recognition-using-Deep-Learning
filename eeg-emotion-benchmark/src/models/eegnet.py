"""
EEGNet (Lawhern et al., 2018) — lightweight compact CNN baseline.
================================================================
Factorizes EEG feature learning into:
    1. temporal convolution        (learns frequency filters)
    2. depthwise spatial convolution (learns per-frequency spatial filters over channels)
    3. separable convolution        (learns how to mix temporal feature maps)

Input : raw windowed signal, tensor [B, C, T]  (C=14 channels, T=window_samples)
Output: class logits [B, n_classes]

Reference config here is EEGNet-8,2 (F1=8, D=2, F2=16), the variant used in the paper
for within-subject classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import make_builder


class _Constrained2d(nn.Conv2d):
    """Conv2d with a max-norm constraint on the kernel (as in the reference Keras impl).

    Implemented with a differentiable `torch.renorm` on a view of the weight each
    forward pass (no in-place mutation of the Parameter), which keeps the per-filter
    L2 norm <= `max_norm` along the output-channel axis.
    """

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        w = self.weight
        if self.max_norm is not None:
            w = torch.renorm(w, p=2, dim=0, maxnorm=self.max_norm)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 14,
        n_samples: int = 512,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int | None = None,
        kernel_length: int = 64,   # ~ half the sampling rate (128 Hz) -> ~0.5 s
        dropout: float = 0.25,
        pool1: int = 4,
        pool2: int = 8,
    ):
        super().__init__()
        F2 = F2 or F1 * D

        # Block 1: temporal conv + depthwise spatial conv
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            # depthwise over the channel axis -> spatial filtering per temporal feature map
            _Constrained2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )

        # Block 2: separable conv (depthwise temporal + pointwise mixing)
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )

        # Derive the flattened feature size from a dummy pass so odd/even kernel
        # padding never causes an off-by-one shape mismatch on non-512 windows.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat = self.separable(self.depthwise(self.temporal(dummy))).flatten(1).shape[1]
        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, 1, C, T]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


build = make_builder(
    EEGNet,
    default_shape=(14, 512),
    shape_kwargs=lambda s: {"n_channels": s[0], "n_samples": s[1]},
    params={"F1": 8, "D": 2, "kernel_length": 64, "dropout": 0.25},
)
