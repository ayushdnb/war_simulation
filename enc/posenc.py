# enc/posenc.py
# -----------------------------------------------------------------------------
# 2-D POSITIONAL ENCODING (general, fast, and resolution-agnostic)
#
# WHAT THIS MODULE DOES
# --------------------
# - Provides a general 2-D sinusoidal positional encoding (PE) tensor for any
#   H×W grid and any EVEN embedding dimension E (e.g., E=32).
# - The returned tensor has shape [H, W, E] and is meant to be **added**
#   element-wise to your per-cell CONTENT embeddings (from enc/embed_map.py).
#
# WHY SIN/COS 2-D PE?
# -------------------
# - It’s parameter-free, deterministic, and works across arbitrary map sizes.
# - Encodes absolute x/y location and a spectrum of spatial frequencies, which
#   helps the model reason about borders, corners, and global orientation.
#
# DESIGN CHOICES
# --------------
# - We split the E channels equally: E/2 channels for Y (rows) and E/2 for X
#   (cols). Each half is a standard 1-D Transformer PE (sin/cos pairs).
#   This requires E % 4 == 0 so each axis gets an even number of sin/cos pairs.
# - We keep PE separate from content embedding for clarity & profiling; you can
#   inspect how much PE changes behavior by toggling its addition on/off.
#
# HOW YOU'LL USE IT
# -----------------
#   from enc.posenc import sincos_posenc_2d, add_posenc_
#
#   pe = sincos_posenc_2d(H, W, E=32, device='cuda', dtype=torch.float16)
#   grid_content = embedder.build_full_grid_embed(static_map, dyn, device='cuda')  # [H,W,32]
#   grid_with_pe = add_posenc_(grid_content, pe)  # in-place add; also returns the tensor
#
# PERFORMANCE NOTES
# -----------------
# - Computation is tiny: O(H*W*E). For 128×128×32 it’s negligible on an RTX 3060.
# - You can precompute PE once per (H,W,E,device,dtype) and reuse it across ticks.
#
# TEST IDEAS (later)
# ------------------
# - Shape check and dtype/device alignment.
# - Corner vs center values differ (PE varies across the grid).
# - L2 norm map shows smooth spatial gradients.
# -----------------------------------------------------------------------------

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch


def _sincos_1d(length: int,
               dim_axis: int,
               base: float = 10_000.0,
               device: Optional[torch.device] = None,
               dtype: torch.dtype = torch.float32,
               normalize_positions: bool = False) -> torch.Tensor:
    """
    Standard 1-D Transformer sinusoidal PE for a single axis.

    Args
    ----
    length : int
        Number of positions along this axis (H for rows or W for cols).
    dim_axis : int
        Number of channels dedicated to this axis. Must be EVEN:
        half will be sin, half will be cos.
    base : float
        Frequency base for log-spaced harmonics (10k is the Transformer default).
    device, dtype :
        Torch placement and dtype for the result.
    normalize_positions : bool
        If True, positions are scaled to [0,1]. Usually keep False to match
        classic Transformer behavior.

    Returns
    -------
    pe_axis : torch.Tensor of shape [length, dim_axis]
    """
    assert dim_axis % 2 == 0, "dim_axis must be even (sin/cos pairing)."

    # Positions (0..length-1) or normalized [0,1]
    pos = torch.arange(length, device=device, dtype=torch.float32)
    if normalize_positions and length > 1:
        pos = pos / float(length - 1)

    # Build the exponent terms for log-spaced frequencies
    half = dim_axis // 2
    i = torch.arange(half, device=device, dtype=torch.float32)
    # angle = pos / (base ** (2i / dim_axis))
    div = torch.exp((2.0 * i / dim_axis) * (-np.log(base)))  # shape [half]

    angles = pos[:, None] * div[None, :]  # [length, half]
    pe_sin = torch.sin(angles)
    pe_cos = torch.cos(angles)
    pe_axis = torch.cat([pe_sin, pe_cos], dim=1).to(dtype)
    return pe_axis  # [length, dim_axis]


@lru_cache(maxsize=64)
def _cached_key(H: int, W: int, E: int, base: float, norm: bool) -> Tuple[int, int, int, int, bool]:
    # Helper to make @lru_cache keys explicit/readable
    return (H, W, E, int(base), norm)


@lru_cache(maxsize=64)
def _build_cached_cpu(H: int,
                      W: int,
                      E: int,
                      base: float,
                      normalize_positions: bool) -> torch.Tensor:
    """
    CPU cache builder for a given (H,W,E,base,normalize) combination.
    We keep a small CPU cache so moving to GPU is just a `.to(device, dtype)`.
    """
    assert E % 4 == 0, "E must be divisible by 4 (half for Y, half for X; each half needs sin/cos pairs)."
    Ey = E // 2
    Ex = E // 2

    # Build per-axis encodings on CPU float32
    pe_y = _sincos_1d(length=H, dim_axis=Ey, base=base, device='cpu',
                      dtype=torch.float32, normalize_positions=normalize_positions)   # [H,Ey]
    pe_x = _sincos_1d(length=W, dim_axis=Ex, base=base, device='cpu',
                      dtype=torch.float32, normalize_positions=normalize_positions)   # [W,Ex]

    # Broadcast to [H,W,E] by combining Y then X channels
    pe_y_full = pe_y[:, None, :].expand(H, W, Ey)  # [H,W,Ey]
    pe_x_full = pe_x[None, :, :].expand(H, W, Ex)  # [H,W,Ex]
    pe = torch.cat([pe_y_full, pe_x_full], dim=-1)  # [H,W,E] float32 CPU
    return pe


def sincos_posenc_2d(H: int,
                     W: int,
                     E: int = 32,
                     *,
                     base: float = 10_000.0,
                     normalize_positions: bool = False,
                     device: Optional[torch.device] = None,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Build a 2-D sinusoidal positional encoding for an H×W grid.

    Args
    ----
    H, W : int
        Grid height/width.
    E : int
        Encoding dimension. Must be divisible by 4 (half Y, half X, sin/cos pairs).
    base : float
        Frequency base (10_000.0 is the Transformer default).
    normalize_positions : bool
        If True, positions are scaled to [0,1] before applying frequencies.
        Leave False for classic absolute-index behavior.
    device, dtype :
        Placement for the returned tensor.

    Returns
    -------
    pe : torch.Tensor of shape [H, W, E]
    """
    # Build (or reuse) a small CPU cache, then move/cast as requested
    cpu_pe = _build_cached_cpu(H, W, E, base, normalize_positions)
    return cpu_pe.to(device=device if device is not None else cpu_pe.device,
                     dtype=dtype)


def add_posenc_(content: torch.Tensor,
                posenc: torch.Tensor) -> torch.Tensor:
    """
    In-place add of positional encoding to CONTENT embeddings.

    Args
    ----
    content : [H,W,E] Float tensor (e.g., from enc/embed_map.py)
    posenc  : [H,W,E] Float tensor (from sincos_posenc_2d)

    Returns
    -------
    content : the same tensor, after in-place addition (also returned for convenience)

    Notes
    -----
    - We assert matching shapes. Dtypes may differ (e.g., content fp16, PE fp32):
      PyTorch will upcast in the arithmetic; if you care about fp16 strictness,
      pass PE in the same dtype as content beforehand.
    """
    assert content.ndim == 3 and posenc.ndim == 3, "content and posenc must be [H,W,E]"
    assert content.shape == posenc.shape, f"shape mismatch: {tuple(content.shape)} vs {tuple(posenc.shape)}"
    content.add_(posenc)  # in-place
    return content
