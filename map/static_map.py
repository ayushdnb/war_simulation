# map/static_map.py
# -----------------------------------------------------------------------------
# Static map representation for the war-sim.
#
# WHY THIS FILE EXISTS
# --------------------
# The world has two parts:
#   1) STATIC (this file): things that do NOT change per tick (tile types,
#      passability rules, topology = hard edges, no wrap)  [Blueprint: Grid, hard edges]
#      → Used by action masks & vision padding, and as an input to per-tick embedding.
#   2) DYNAMIC (other file): agents occupying cells (owner, hp, cooldown, positions)
#      → These change every tick and form the core of per-cell features in v1.
#
# In v1, your blueprint says per-cell features for learning are MINIMAL and DYNAMIC:
# has_red, has_blue, hp. Static terrain effects are deferred to later phases.
# So this module stays intentionally lightweight now and scalable later.
# [Refs] Grid + topology; minimal v1 features; embedding & tokens plan
#   - Grid: 128x128, hard edges (no wrap).            (L12-L15)        :contentReference[oaicite:7]{index=7}
#   - Per-cell v1 features are dynamic (red/blue/hp). (L60-L64)        :contentReference[oaicite:8]{index=8}
#   - Full-grid embedding once per tick, agents slice 27x27 patches.   :contentReference[oaicite:9]{index=9}
#   - Token stream is 1 [GLOBAL] + 27x27 = 730 tokens, E=128.          :contentReference[oaicite:10]{index=10}
#
# WHAT THIS MODULE PROVIDES
# -------------------------
# - TileType enum      : stable integer codes (0=PLAIN for v1)
# - StaticMap dataclass: holds tile_type grid (int8) and a passable LUT.
# - Factory: generate_plain(H, W) → StaticMap with all-PLAIN tiles (passable).
# - Utilities: passable_grid(), summary(), assert_invariants()
#
# HOW IT'S USED
# -------------
# - Action masks consult passability & hard edges to block illegal moves.
#   (Your v1 mask checks off-grid/occupied/etc.; terrain comes in Phase 3.)
#   [Action mask invalid reasons; Terrain later]        (L28-L31), (Phase 3)   :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
# - Vision padding uses hard-edge topology: out-of-bounds cells are invalid
#   and zero-padded in the 27x27 patch.                                 :contentReference[oaicite:13]{index=13}
# - Embed step merges STATIC (this module) + DYNAMIC (owner/hp) + posenc
#   to produce full_grid_embed[H, W, 128] once per tick.                 :contentReference[oaicite:14]{index=14}
#
# DESIGN NOTES
# ------------
# - We store arrays, not per-cell Python objects. This is GPU- and PPO-friendly
#   and keeps hot loops vectorized.
# - Passability is handled via a LUT so adding terrain types later is O(1).
# - Default H=W=128 matches the blueprint but callers can override.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple, Dict

import numpy as np


class TileType(IntEnum):
    """Stable numeric codes for tiles. v1 uses only PLAIN."""
    PLAIN = 0
    # Future: MOUNTAIN=1, WATER=2, FOREST=3, etc. (Phase 3 terrain)  # :contentReference[oaicite:15]{index=15}


@dataclass(frozen=True)
class StaticMap:
    """
    Immutable container for the static map.

    Attributes
    ----------
    H, W : int
        Grid height/width. Blueprint default is 128x128 with hard edges (no wrap).  # :contentReference[oaicite:16]{index=16}
    tile_type : np.ndarray[int8] of shape (H, W)
        Integer tile codes per cell (e.g., 0=PLAIN). In v1, all zeros.
    passable_lut : np.ndarray[bool] of shape (num_tile_types,)
        Lookup table mapping a tile type code → passable / not.
        v1: only PLAIN exists and is passable=True.
    topology : str
        'hard_edges' indicates out-of-bounds is invalid (no wrap). Used by vision and masks.
    """
    H: int
    W: int
    tile_type: np.ndarray
    passable_lut: np.ndarray
    topology: str = "hard_edges"

    # --------------------------
    # Derived helpers
    # --------------------------
    def passable_grid(self) -> np.ndarray:
        """
        Boolean grid [H,W] derived from tile_type and passable_lut.

        Used by:
          - action masking (block moves into impassable cells when terrain arrives)
          - path/placement checks

        v1 note: entire grid is passable, returns all True.
        """
        return self.passable_lut[self.tile_type]

    def summary(self) -> Dict[str, object]:
        """Small dict useful for logs/debugging."""
        unique, counts = np.unique(self.tile_type, return_counts=True)
        return {
            "shape": (self.H, self.W),
            "topology": self.topology,
            "tile_hist": dict(zip([int(u) for u in unique], [int(c) for c in counts])),
            "num_tile_types": int(self.passable_lut.shape[0]),
        }

    def assert_invariants(self) -> None:
        """
        Sanity checks: shapes/dtypes/topology.
        Keep failures loud & early to prevent silent bugs in the step loop.
        """
        assert isinstance(self.H, int) and isinstance(self.W, int), "H,W must be ints"
        assert self.tile_type.shape == (self.H, self.W), "tile_type shape mismatch"
        assert self.tile_type.dtype == np.int8, "tile_type must be int8"
        assert self.passable_lut.ndim == 1, "passable_lut must be 1D"
        assert self.passable_lut.dtype == np.bool_, "passable_lut must be bool dtype"
        assert self.topology in ("hard_edges",), "unsupported topology"
        # Blueprint: hard edges = OOB cells are invalid; vision pads OOB to zero.  # :contentReference[oaicite:17]{index=17}


def generate_plain(H: int = 128, W: int = 128) -> StaticMap:
    """
    Factory: build an all-PLAIN, fully passable map.

    Parameters
    ----------
    H, W : int
        Grid size. Blueprint default 128x128.                             # :contentReference[oaicite:18]{index=18}

    Returns
    -------
    StaticMap
        - tile_type[H,W] = 0 (TileType.PLAIN)
        - passable_lut   = [True]  (index 0 → passable)
        - topology       = 'hard_edges'  (no wrap; OOB invalid)            # :contentReference[oaicite:19]{index=19}

    Why this shape/design?
    ----------------------
    - v1 per-cell learning features are dynamic (red/blue/hp) and will be added
      at embed time, not stored here.                                      # :contentReference[oaicite:20]{index=20}
    - Full-grid embedding is computed once per tick from STATIC + DYNAMIC + posenc,
      then agents slice 27x27 patches to form their 730-token input.       # :contentReference[oaicite:21]{index=21} :contentReference[oaicite:22]{index=22}
    """
    tile_type = np.zeros((H, W), dtype=np.int8)  # all PLAIN
    passable_lut = np.array([True], dtype=np.bool_)  # type 0 (PLAIN) is passable

    m = StaticMap(H=H, W=W, tile_type=tile_type, passable_lut=passable_lut, topology="hard_edges")
    m.assert_invariants()
    return m
