"""
CIT-EmotionNet — CNN Interactive Transformer network (Lu et al., arXiv 2023;
PeerJ Comput. Sci. 2024, doi:10.7717/peerj-cs.2610).

Reproduced from the paper (Sections 2.2-2.5, 3.1):
    * Input: spatial-spectral map SM in R^{32 x 32 x B}: per-band electrode values are
      placed on a 2-D scalp grid (B = number of frequency bands).
    * CNN branch: ResNet; Stage 0 (7x7 conv + BN + ReLU, resolution preserved) and
      Stages 1-4 with two BasicBlocks each (64, 128, 256, 512 channels).
    * Transformer branch: ViT with patch size 32 -> one token per band (+ class token),
      hidden dim 256, three encoder blocks per stage.
    * CIT module after Stages 1-3:
        L2G: Mean over H, W -> Linear -> Expand -> Concat as an extra ViT token (Eq. 11-15)
        G2L: first ViT token -> Unsqueeze/Expand to H x W -> Concat with CNN map ->
             1x1 conv back to the CNN width (Eq. 16-20)
    * Head: concat(global-average-pooled Stage-4 CNN map, Stage-4 class token) -> FC.
    * Recipe: Adam, lr 1e-5, batch size 32, dropout 0.2, cross-entropy.

Documented deviations:
    * The paper uses PSD band power from 62 SEED channels. Here the band feature is the
      differential entropy, which for band-limited Gaussian signals equals
      0.5*ln(2*pi*e) + 0.5*ln(band power), i.e. an affine function of log-PSD that is
      removed by per-feature z-scoring; values are interpolated from the 14 shared
      electrodes (9x9 10-20 grid, bilinear upsampling to 32x32).
    * The paper's headline accuracy (98.57% SEED) uses a RANDOM 6:4 window split.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...preprocessing import COMMON_CHANNELS
from .._common import make_builder

# Standard 9x9 10-20 grid positions (row: front -> back, col: left -> right).
GRID_9X9 = {
    "AF3": (1, 3), "AF4": (1, 5),
    "F7": (2, 0), "F3": (2, 2), "F4": (2, 6), "F8": (2, 8),
    "FC5": (3, 1), "FC6": (3, 7),
    "T7": (4, 0), "T8": (4, 8),
    "P7": (6, 0), "P8": (6, 8),
    "O1": (8, 3), "O2": (8, 5),
}


def spatial_spectral_map(x: torch.Tensor, channel_names: Sequence[str], size: int = 32) -> torch.Tensor:
    """[B, C, Bands] -> [B, Bands, size, size] scalp maps."""
    b, c, n_bands = x.shape
    grid = x.new_zeros(b, n_bands, 9, 9)
    for ci, name in enumerate(channel_names[:c]):
        r, col = GRID_9X9[name]
        grid[:, :, r, col] = x[:, ci, :]
    return F.interpolate(grid, size=(size, size), mode="bilinear", align_corners=True)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout),
        )
        self.skip = (nn.Identity() if stride == 1 and cin == cout else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x))


class L2G(nn.Module):
    def __init__(self, c_local: int, dim: int):
        super().__init__()
        self.linear = nn.Linear(c_local, dim)

    def forward(self, f_local: torch.Tensor, f_global: torch.Tensor) -> torch.Tensor:
        token = self.linear(f_local.mean(dim=(2, 3))).unsqueeze(1)    # Mean -> Linear -> Expand
        return torch.cat([token, f_global], dim=1)                     # Concat (new token)


class G2L(nn.Module):
    def __init__(self, c_local: int, dim: int):
        super().__init__()
        self.fuse = nn.Conv2d(c_local + dim, c_local, 1)

    def forward(self, f_global: torch.Tensor, f_local: torch.Tensor) -> torch.Tensor:
        g = f_global[:, 0][..., None, None].expand(-1, -1, *f_local.shape[2:])
        return self.fuse(torch.cat([g, f_local], dim=1))


class CITEmotionNet(nn.Module):
    def __init__(self, n_channels: int = 14, n_bands: int = 4, n_classes: int = 2,
                 widths: Sequence[int] = (64, 128, 256, 512), dim: int = 256, heads: int = 8,
                 blocks_per_stage: int = 3, map_size: int = 32, dropout: float = 0.2,
                 channel_names=None):
        super().__init__()
        self.channel_names = list(channel_names or COMMON_CHANNELS)[:n_channels]
        self.map_size = map_size
        self.stage0_cnn = nn.Sequential(
            nn.Conv2d(n_bands, widths[0], 7, 1, 3, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True),
        )
        self.cnn_stages = nn.ModuleList()
        cin = widths[0]
        for i, w in enumerate(widths):
            self.cnn_stages.append(nn.Sequential(BasicBlock(cin, w, 1 if i == 0 else 2), BasicBlock(w, w)))
            cin = w
        self.patch = nn.Linear(map_size * map_size, dim)            # one 32x32 patch per band
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n_bands + 1, dim) * 0.02)
        layer = lambda: nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True,
                                                   activation="gelu", norm_first=True)
        self.vit_stages = nn.ModuleList(
            nn.TransformerEncoder(layer(), blocks_per_stage, enable_nested_tensor=False) for _ in widths
        )
        self.l2g = nn.ModuleList(L2G(w, dim) for w in widths[:3])
        self.g2l = nn.ModuleList(G2L(w, dim) for w in widths[:3])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(widths[-1] + dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sm = spatial_spectral_map(x, self.channel_names, self.map_size)       # [B, Bands, H, W]
        f_local = self.stage0_cnn(sm)
        tokens = self.patch(sm.flatten(2))                                     # [B, Bands, dim]
        f_global = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1) + self.pos
        for k, (cnn, vit) in enumerate(zip(self.cnn_stages, self.vit_stages)):
            f_local, f_global = cnn(f_local), vit(f_global)
            if k < 3:                                                          # CIT module
                f_local, f_global = self.g2l[k](f_global, f_local), self.l2g[k](f_local, f_global)
        fused = torch.cat([f_local.mean(dim=(2, 3)), self.norm(f_global[:, 0])], dim=1)
        return self.head(fused)


build = make_builder(
    CITEmotionNet,
    default_shape=(14, 4),
    shape_kwargs=lambda s: {"n_channels": s[0], "n_bands": s[1]},
    params={"widths": (64, 128, 256, 512), "dim": 256, "heads": 8, "blocks_per_stage": 3,
            "map_size": 32, "dropout": 0.2},
)
