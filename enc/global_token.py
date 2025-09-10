# enc/global_token.py
# -----------------------------------------------------------------------------
# Per-agent [GLOBAL] token embedding for the war-sim.
#
# Inputs (per agent, real-time from the engine):
#   - pos_y, pos_x         : absolute integer grid coords (0..H-1, 0..W-1)
#   - hp                   : 0..hp_max
#   - red_points, blue_points : non-negative ints (unbounded → tamed via tanh scale)
#   - red_alive, blue_alive   : 0..team_size
#
# Outputs:
#   - global_token_embed : torch.FloatTensor[E] (or [B,E]) with E=32 by default
#
# Notes:
#   - No positional encoding is added here (this is a single token).
#   - We normalize everything:
#       pos → [0,1], hp → [0,1], alive → [0,1] by team_size,
#       points → tanh(points / points_scale), diffs similarly.
#   - Two derived signals help learning:
#       points_diff = tanh((red_points - blue_points)/points_scale)
#       alive_diff  = clamp((red_alive - blue_alive)/team_size, -1, 1)
#     You can disable via add_derived=False.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class GlobalTokenInput:
    """Container for per-agent global token inputs (scalars or 1-D batches)."""
    pos_y: ArrayLike            # 0..H-1
    pos_x: ArrayLike            # 0..W-1
    hp: ArrayLike               # 0..hp_max
    red_points: ArrayLike       # >= 0
    blue_points: ArrayLike      # >= 0
    red_alive: ArrayLike        # 0..team_size
    blue_alive: ArrayLike       # 0..team_size
    # Future optional fields: tick, team_id, vision_frac, cooldown, etc.


def _to1d(x: ArrayLike, device=None, dtype=torch.float32) -> torch.Tensor:
    """Accept numpy/torch scalars or arrays; return 1-D torch tensor."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        # python scalar
        t = torch.tensor(x)
    if device is not None and t.device != device:
        t = t.to(device)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if t.ndim == 0:
        t = t.view(1)
    return t.contiguous()


class GlobalTokenEmbedder(nn.Module):
    """
    Tiny MLP that maps normalized global features to E-dim embedding.

    Feature vector (order):
      [ y_norm, x_norm, hp_norm,
        red_pts_tanh, blue_pts_tanh,
        red_alive_norm, blue_alive_norm,
        (points_diff_tanh), (alive_diff) ]

    Toggle derived signals with add_derived.
    """

    def __init__(self,
                 embed_dim: int = 32,
                 hp_max: int = 3,
                 team_size: int = 100,
                 points_scale: float = 100.0,   # scale for tanh(points/scale)
                 add_derived: bool = True,
                 out_dtype: torch.dtype = torch.float32):
        super().__init__()
        assert embed_dim > 0
        assert hp_max > 0 and team_size > 0 and points_scale > 0.0

        self.embed_dim = embed_dim
        self.hp_max = float(hp_max)
        self.team_size = float(team_size)
        self.points_scale = float(points_scale)
        self.add_derived = add_derived
        self.out_dtype = out_dtype

        # Input feature count
        base_feats = 7  # y_norm, x_norm, hp_norm, red_pts, blue_pts, red_alive, blue_alive
        self.in_dim = base_feats + (2 if add_derived else 0)

        hidden = max(64, self.in_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def _normalize(self, g: GlobalTokenInput, H: int, W: int, device=None) -> torch.Tensor:
        y = _to1d(g.pos_y, device)
        x = _to1d(g.pos_x, device)
        hp = _to1d(g.hp, device)
        rp = _to1d(g.red_points, device)
        bp = _to1d(g.blue_points, device)
        ra = _to1d(g.red_alive, device)
        ba = _to1d(g.blue_alive, device)

        # Broadcast-safe checks
        B = max(y.shape[0], x.shape[0], hp.shape[0], rp.shape[0], bp.shape[0], ra.shape[0], ba.shape[0])

        # Normalize
        y_norm = torch.clamp(y / max(1.0, (H - 1)), 0.0, 1.0)
        x_norm = torch.clamp(x / max(1.0, (W - 1)), 0.0, 1.0)
        hp_norm = torch.clamp(hp / self.hp_max, 0.0, 1.0)

        red_pts = torch.tanh(rp / self.points_scale)
        blue_pts = torch.tanh(bp / self.points_scale)

        red_alive = torch.clamp(ra / self.team_size, 0.0, 1.0)
        blue_alive = torch.clamp(ba / self.team_size, 0.0, 1.0)

        feats = [y_norm, x_norm, hp_norm, red_pts, blue_pts, red_alive, blue_alive]

        if self.add_derived:
            pts_diff = torch.tanh((rp - bp) / self.points_scale)
            alive_diff = torch.clamp((ra - ba) / self.team_size, -1.0, 1.0)
            feats.extend([pts_diff, alive_diff])

        # Stack to [B, F]
        Fcat = torch.stack([f if f.ndim == 1 else f.view(-1) for f in feats], dim=1).to(torch.float32)
        return Fcat

    def forward(self,
                static_map,              # provides H, W
                g: GlobalTokenInput,
                device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns:
          - [E] if inputs are scalars
          - [B,E] if any input is a 1-D batch
        """
        H, W = int(static_map.H), int(static_map.W)
        feats = self._normalize(g, H, W, device=device)      # [B, F]
        out = self.mlp(feats)                                # [B, E]
        out = out.to(self.out_dtype)
        if out.shape[0] == 1:
            return out.view(self.embed_dim)                  # [E]
        return out                                           # [B, E]
