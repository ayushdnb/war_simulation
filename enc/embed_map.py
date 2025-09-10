# enc/embed_map.py
# -----------------------------------------------------------------------------
# Full-grid CONTENT embedding (no positional encoding here) for the war-sim.
#
# WHAT THIS MODULE DOES
# --------------------
# For each tick, produce a tensor:
#     full_grid_embed_content : Float[H, W, E]   with E = 32 (configurable)
#
# "Content" = local per-cell information only:
#   - STATIC (precomputed once per map): tile_type + passable flag → small vector
#   - DYNAMIC (per tick): has_red, has_blue, hp_norm (and optional cooldown_norm)
#
# We DO NOT add positional encoding here (that happens in enc/posenc.py and a
# small utility that sums PE into this content embedding). Keeping PE separate
# makes debugging easier and lets us profile content vs. position impact.
#
# WHY THIS DESIGN
# ---------------
# - Arrays, not per-cell Python objects → vectorized, GPU-friendly, PPO-friendly.
# - Static base is cached (map rarely changes). Dynamic channels are cheap.
# - Small per-cell MLP ("1x1 conv" style) → fast even at 128x128.
# - Global scalars (scores, alive counts) are intentionally NOT repeated into
#   every cell; those will go into the [GLOBAL] token later.
#
# INPUTS (from other modules)
# ---------------------------
# - StaticMap (map/static_map.py):
#     .H, .W
#     .tile_type : np.int8[H,W]      (# 0 = PLAIN in v1)
#     .passable_lut : np.bool_[T]    (# passability of each tile type)
#
# - Dynamic snapshot (light holder from your engine; can be a dataclass or dict):
#     owner : int8[H,W]    (# 0 empty, 1 RED, 2 BLUE)
#     hp    : int8[H,W]    (# 0..HP_MAX; 0 means no agent or dead)
#     cooldown (optional) : int8[H,W] (# 0..CD_MAX, 0 means ready)
#
# OUTPUTS
# -------
# - full_grid_embed_content : torch.FloatTensor[H, W, E]  (default E=32)
#   (You will add positional encoding elsewhere: full_grid_embed = content + PE)
#
# USAGE (typical)
# ---------------
#   embedder = MapEmbedder(embed_dim=32, tile_type_vocab_size=static.passable_lut.size)
#   grid_emb = embedder.build_full_grid_embed(static_map=static, dynamic=dyn, device='cuda')
#
# PERF NOTES
# ----------
# - 128x128x32 fp16 is ~1.0 MB; recomputing per tick is fine on RTX 3060.
# - All agents will later slice 27x27 patches out of this cached grid emb.
#
# TEST IDEAS (later)
# ------------------
# - Static cache correctness: change map → cache invalidates.
# - has_red / has_blue derived correctly from owner.
# - hp_norm ∈ [0,1], zeros where owner==0.
# - Output shape [H,W,32]; finite; non-zero where agents exist.
# -----------------------------------------------------------------------------

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Types for clarity
ArrayLike = Union[np.ndarray, torch.Tensor]


# -------------------------
# Dynamic snapshot contract
# -------------------------
@dataclass(frozen=True)
class DynamicSnapshot:
    """
    Minimal per-tick state needed for embedding.
    You can pass this from your engine as-is, or adapt from your own holder.

    Shapes:
      owner:   int8[H,W]  (0=empty, 1=RED, 2=BLUE)
      hp:      int8[H,W]  (0..HP_MAX)
      cooldown (optional): int8[H,W] (0..CD_MAX)  # include if use_cooldown=True
    """
    owner: ArrayLike
    hp: ArrayLike
    cooldown: Optional[ArrayLike] = None


# -----------------------------------
# Helper: convert & move to torch
# -----------------------------------
def _to_tensor(x: ArrayLike,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Accepts numpy or torch; returns torch on desired device/dtype (no copy if possible).
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        raise TypeError(f"Unsupported array type: {type(x)}")
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype)
    if device is not None and t.device != device:
        t = t.to(device)
    return t


# -----------------------------------
# Helper: stable map key for caching
# -----------------------------------
def _map_cache_key(tile_type_np: np.ndarray,
                   passable_lut_np: np.ndarray,
                   embed_dim: int,
                   static_dim: int) -> str:
    """
    Build a stable hash key to cache the static base for a specific map + dims.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(tile_type_np.shape[0].to_bytes(4, "little"))
    h.update(tile_type_np.shape[1].to_bytes(4, "little"))
    h.update(tile_type_np.tobytes())  # small (H*W int8)
    h.update(passable_lut_np.tobytes())  # tiny
    h.update(embed_dim.to_bytes(2, "little"))
    h.update(static_dim.to_bytes(2, "little"))
    return h.hexdigest()


# -------------------------
# The main embedder module
# -------------------------
class MapEmbedder(nn.Module):
    """
    Per-cell CONTENT embedding for the static + dynamic world state.

    Architecture:
      1) Static base (cached):
         tile_type_emb(T->static_dim) + passable_flag(1) → Linear(static_dim)
      2) Dynamic channels per tick:
         has_red(1), has_blue(1), hp_norm(1) [+ cooldown_norm(1) if enabled]
      3) Fuse:
         concat(static_base, dynamic_channels) → MLP → embed_dim (E=32)

    Notes:
      - We compute in float32 for stability, return in requested dtype (fp16 ok).
      - No positional encoding here; add it outside after this returns.
    """

    def __init__(self,
                 embed_dim: int = 32,
                 tile_type_vocab_size: int = 1,
                 static_dim: int = 8,
                 use_cooldown: bool = False,
                 hp_max: int = 3,
                 cd_max: int = 3,
                 out_dtype: torch.dtype = torch.float32):
        super().__init__()
        assert embed_dim > 0 and static_dim > 0
        assert tile_type_vocab_size >= 1
        self.embed_dim = embed_dim
        self.static_dim = static_dim
        self.use_cooldown = use_cooldown
        self.hp_max = hp_max
        self.cd_max = cd_max
        self.out_dtype = out_dtype

        # 1) Static tile embedding (even if v1 has only PLAIN=0, keep API stable)
        self.tile_emb = nn.Embedding(num_embeddings=tile_type_vocab_size,
                                     embedding_dim=static_dim)

        # Passable flag (scalar) + tile_emb → static_dim via small linear
        # Input dim = static_dim (tile emb) + 1 (passable)
        self.static_proj = nn.Linear(static_dim + 1, static_dim)

        # 2) Dynamic channel count
        dyn_channels = 3 + (1 if use_cooldown else 0)  # has_red, has_blue, hp_norm [, cd_norm]

        # 3) Fuse MLP (lightweight, per-cell)
        fuse_in = static_dim + dyn_channels
        hidden = max(32, fuse_in)  # small hidden; you can tweak
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

        # ------------- cache for static base (per map) -------------
        self._cache_key: Optional[str] = None
        self._cached_static_base: Optional[torch.Tensor] = None  # [H,W,static_dim]

    # -------------------------
    # Public high-level API
    # -------------------------
    @torch.no_grad()
    def precompute_static_base(self,
                               H: int,
                               W: int,
                               tile_type: ArrayLike,
                               passable_lut: ArrayLike,
                               device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Build and cache the static base tensor: [H, W, static_dim].
        Call this once per map (or it will be invoked lazily by build_full_grid_embed).

        Static channels:
          - tile_type_emb  : learned table lookup
          - passable_flag  : passable_lut[tile_type] ∈ {0,1}
          - static_proj    : Linear([tile_emb || passable]) → static_dim
        """
        # Ensure numpy views for hashing + torch tensors for compute
        if isinstance(tile_type, torch.Tensor):
            tile_np = tile_type.detach().cpu().numpy().astype(np.int8, copy=False)
        else:
            tile_np = tile_type.astype(np.int8, copy=False)

        if isinstance(passable_lut, torch.Tensor):
            lut_np = passable_lut.detach().cpu().numpy().astype(np.bool_, copy=False)
        else:
            lut_np = passable_lut.astype(np.bool_, copy=False)

        key = _map_cache_key(tile_np, lut_np, self.embed_dim, self.static_dim)
        if key == self._cache_key and self._cached_static_base is not None:
            return self._cached_static_base  # already on correct device

        # Convert to torch on device
        tt = _to_tensor(tile_np, device=device, dtype=torch.long)     # for Embedding
        lut = _to_tensor(lut_np, device=device, dtype=torch.bool)     # LUT

        # passable flag per cell
        passable = lut[tt]                                            # bool[H,W]
        passable_f = passable.to(torch.float32)                       # float in {0,1}

        # tile type embedding
        tile_vec = self.tile_emb(tt)                                  # [H,W,static_dim]

        # concat + project
        static_in = torch.cat([tile_vec, passable_f.unsqueeze(-1)], dim=-1)  # [H,W,static_dim+1]
        static_base = self.static_proj(static_in)                     # [H,W,static_dim]

        self._cache_key = key
        self._cached_static_base = static_base
        return static_base

    def build_dynamic_channels(self,
                               owner: ArrayLike,
                               hp: ArrayLike,
                               cooldown: Optional[ArrayLike] = None,
                               device: Optional[torch.device] = None) -> Tuple[torch.Tensor, ...]:
        """
        Derive cheap per-cell dynamic channels (float32) from discrete grids.

        Returns a tuple of Float[H,W] tensors in this fixed order:
          (has_red, has_blue, hp_norm [, cd_norm])
        """
        own = _to_tensor(owner, device=device)
        hp_ = _to_tensor(hp, device=device)

        assert own.ndim == 2 and hp_.ndim == 2, "owner/hp must be [H,W]"

        # has_red / has_blue as {0,1} floats
        has_red = (own == 1).to(torch.float32)
        has_blue = (own == 2).to(torch.float32)

        # hp_norm in [0,1]
        hp_norm = hp_.to(torch.float32) / float(self.hp_max)
        hp_norm = torch.clamp(hp_norm, 0.0, 1.0)

        if self.use_cooldown:
            assert cooldown is not None, "cooldown grid required when use_cooldown=True"
            cd = _to_tensor(cooldown, device=device)
            cd_norm = cd.to(torch.float32) / float(self.cd_max)
            cd_norm = torch.clamp(cd_norm, 0.0, 1.0)
            return has_red, has_blue, hp_norm, cd_norm

        return has_red, has_blue, hp_norm

    def fuse_to_content_embed(self,
                              static_base: torch.Tensor,
                              dyn_channels: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Concatenate static_base and dynamic channels per cell,
        project through a tiny MLP to embed_dim (E=32).

        Shapes:
          static_base : [H,W,static_dim]
          dyn_channels: tuple of k tensors each [H,W]
          output      : [H,W,embed_dim]
        """
        H, W, _ = static_base.shape

        # Stack dynamic channels along the last dim: [H,W,k]
        dyn_stack = torch.stack(dyn_channels, dim=-1)  # [H,W,k]

        # Cat features: [H,W, static_dim + k]
        feats = torch.cat([static_base, dyn_stack], dim=-1)

        # Flatten for per-cell MLP
        flat = feats.view(H * W, -1).to(torch.float32)  # compute in fp32

        out = self.fuse_mlp(flat)  # [H*W, embed_dim]
        out = out.view(H, W, self.embed_dim).to(self.out_dtype)
        return out

    def build_full_grid_embed(self,
                              static_map,             # StaticMap
                              dynamic: DynamicSnapshot,
                              device: Optional[torch.device] = None) -> torch.Tensor:
        """
        One-stop call per tick:
          1) get cached static base
          2) derive dynamic channels
          3) fuse to per-cell content embeddings

        Returns:
          full_grid_embed_content : Float[H, W, embed_dim]  (dtype=self.out_dtype)
        """
        H, W = int(static_map.H), int(static_map.W)

        # 1) Static base (cached)
        static_base = self.precompute_static_base(
            H=H, W=W,
            tile_type=static_map.tile_type,
            passable_lut=static_map.passable_lut,
            device=device,
        )

        # 2) Dynamic channels
        if self.use_cooldown:
            dyn = self.build_dynamic_channels(
                owner=dynamic.owner,
                hp=dynamic.hp,
                cooldown=dynamic.cooldown,
                device=device,
            )
        else:
            dyn = self.build_dynamic_channels(
                owner=dynamic.owner,
                hp=dynamic.hp,
                cooldown=None,
                device=device,
            )

        # 3) Fuse to E-dim content
        content = self.fuse_to_content_embed(static_base, dyn)
        return content
