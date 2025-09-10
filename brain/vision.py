# brain/vision.py
# =============================================================================
# Agent-centric vision tokenizer:
#   • Input:  full-grid embeddings WITH positional encoding  [H, W, E]
#   • Input:  agent positions                                [B, 2]  (y, x)
#   • Input:  per-agent global tokens (already embedded)     [B, E]  or [E]
#   • Output: per-agent token stream                         [B, 730, E]
#         -> 729 = 27×27 local vision (with OOB filled by learnable E_OOB)
#         ->   1 = global token (appended as the last token; never masked)
#
# Design choices (frozen contracts):
#   • We DO NOT multiply embeddings by a vision mask. OOB cells are represented
#     by a semantic, learnable vector E_OOB (shape [E]) so the model can learn
#     "edge-awareness" rather than receiving pure zeros. (Think: BERT [PAD] token.)
#   • We STILL return a boolean mask for the 729 grid tokens (True=in-bounds).
#     This is *not* applied to embeddings here; it's for optional attention/loss.
#   • Global token is appended as-is and is never masked by vision rules.
#   • The grid passed here is assumed to be *already* content+PE (enc/embed_map
#     + enc/posenc). We do not add PE in this module; we only slice/inject E_OOB.
#
# Performance:
#   • Fully batched/vectorized using advanced indexing; no Python loops over B.
#   • Memory: 27×27×E per agent. With E=32 and B ~ 1000, it's tiny on RTX 3060.
#
# Invariants:
#   • E (embed_dim) must match the embedding dimension used across the stack.
#   • Vision window (27) and radius (13) are taken from engine.rules.CFG to keep
#     the entire codebase consistent (one source of truth).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Import canonical config (window/radius) from engine
from engine.rules import CFG  # vision_window_size=27, vision_radius_cheb=13


@dataclass(frozen=True)
class VisionSpec:
    """
    Static spec for the vision tokenizer.
    We read values from CFG to avoid divergent constants across files.
    """
    window: int = CFG.vision_window_size           # 27
    radius: int = CFG.vision_radius_cheb           # 13
    embed_dim: int = 32                            # E; keep consistent with enc stack

    # Safety: assert window matches radius*2+1
    def __post_init__(self):
        if self.window != self.radius * 2 + 1:
            raise ValueError(
                f"VisionSpec invalid: window={self.window} vs radius={self.radius} "
                f"(expected window == radius*2 + 1)."
            )


class VisionTokenizer(nn.Module):
    """
    Build an agent's token stream from the global grid embedding.

    Core responsibilities:
      1) Slice a 27×27 patch centered on each agent.
      2) Replace out-of-bounds (OOB) cells with a *learnable* E_OOB vector.
      3) Return both:
           • vision patch as tokens [B, 729, E]
           • vision mask           [B, 729]  (True=in-bounds)
      4) Concatenate the provided per-agent global token [B, E] to form:
           • tokens_with_global   [B, 730, E]
         (Global token is not masked here; vision mask covers only the 729 grid tokens.)

    IMPORTANT:
      • We DO NOT multiply embeddings by the mask. The semantics of OOB are carried
        by E_OOB (learnable). If you later run attention or compute loss, you can
        pass the returned mask to those modules.
    """

    def __init__(self, spec: Optional[VisionSpec] = None, *, embed_dim: int = 32):
        """
        Args:
          spec: optional VisionSpec; if None, uses CFG-derived defaults.
          embed_dim: kept for convenience/clarity; must equal spec.embed_dim.
        """
        super().__init__()
        self.spec = spec or VisionSpec(embed_dim=embed_dim)
        if self.spec.embed_dim != embed_dim:
            raise ValueError(f"embed_dim mismatch: spec={self.spec.embed_dim} vs arg={embed_dim}")

        # A single learnable vector used for all out-of-bounds cells.
        # Small init keeps it quiet until learning shapes it.
        self.e_oob = nn.Parameter(torch.empty(self.spec.embed_dim))
        nn.init.normal_(self.e_oob, mean=0.0, std=0.01)

        # Precompute the relative offset grid (dy, dx) for the 27×27 window.
        # Shape: [win, win] each. These are CPU tensors; moved/cast at forward.
        r = self.spec.radius
        ys = torch.arange(-r, r + 1, dtype=torch.int32)  # [-13..+13]
        xs = torch.arange(-r, r + 1, dtype=torch.int32)  # [-13..+13]
        self.register_buffer("_dy", ys.view(-1, 1).expand(self.spec.window, self.spec.window), persistent=False)
        self.register_buffer("_dx", xs.view(1, -1).expand(self.spec.window, self.spec.window), persistent=False)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def extract_patch(
        self,
        grid_with_pe: torch.Tensor,   # [H, W, E]  (already content+PE)
        pos: torch.Tensor,            # [B, 2]     (y, x) int tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slice the per-agent 27×27 patch and inject E_OOB for out-of-bounds cells.

        Args:
          grid_with_pe : Float tensor [H, W, E]. This MUST already include
                         positional encodings (enc.posenc.add_posenc_ was applied).
          pos          : Int tensor [B, 2] with (y, x) per agent (row-major).

        Returns:
          patch_tokens : Float tensor [B, 729, E].
                         Row-major flatten of the 27×27 window. For in-bounds
                         cells, values come from grid_with_pe[y, x]. For OOB,
                         the token is the learnable E_OOB vector.
          patch_mask   : Bool tensor [B, 729]. True where token is in-bounds,
                         False where token is OOB (i.e., came from E_OOB).

        Notes:
          • No masks are *applied* to embeddings here. We only *return* the mask.
          • Device/dtype follow the input grid (E_OOB is cast to match).
        """
        H, W, E = grid_with_pe.shape
        if E != self.spec.embed_dim:
            raise ValueError(f"Embedding dim mismatch: grid E={E} vs spec E={self.spec.embed_dim}")

        device = grid_with_pe.device
        dtype = grid_with_pe.dtype

        # Positions -> broadcast to window
        # pos_y/x: [B] -> [B, 1, 1] so we can add dy/dx
        pos = pos.to(torch.int32).to(device)
        y0 = pos[:, 0].view(-1, 1, 1)  # [B,1,1]
        x0 = pos[:, 1].view(-1, 1, 1)  # [B,1,1]

        # Relative offsets to absolute candidate coords for each window cell.
        # y_idx/x_idx: [B, win, win]
        dy = self._dy.to(device=device)  # [win,win]
        dx = self._dx.to(device=device)
        y_idx = y0 + dy  # [B,win,win]
        x_idx = x0 + dx  # [B,win,win]

        # Compute OOB mask under HARD_EDGES topology.
        oob = (y_idx < 0) | (y_idx >= H) | (x_idx < 0) | (x_idx >= W)  # [B,win,win]
        inb = ~oob

        # For safe fancy-indexing, clamp coordinates to the grid.
        # We'll later overwrite OOB positions with E_OOB, so clamping is harmless.
        y_clamped = y_idx.clamp_(0, H - 1).to(torch.long)
        x_clamped = x_idx.clamp_(0, W - 1).to(torch.long)

        # Fancy indexing: grid_with_pe[y_clamped, x_clamped] → [B,win,win,E]
        # (PyTorch broadcasts last dim automatically for advanced indexing.)
        patch = grid_with_pe[y_clamped, x_clamped]  # [B,win,win,E]

        # Replace OOB cells by E_OOB (learnable), cast to grid dtype/device.
        e = self.e_oob.to(device=device, dtype=dtype).view(1, 1, 1, E)  # [1,1,1,E]
        patch = torch.where(oob.unsqueeze(-1), e, patch)  # [B,win,win,E]

        # Flatten to tokens: [B, 729, E]; and a flat mask [B, 729]
        B = pos.shape[0]
        win = self.spec.window
        patch_tokens = patch.view(B, win * win, E)
        patch_mask = inb.view(B, win * win)  # True=in-bounds, False=OOB

        return patch_tokens, patch_mask

    @torch.no_grad()
    def build_agent_tokens(
        self,
        grid_with_pe: torch.Tensor,  # [H, W, E]
        pos: torch.Tensor,           # [B, 2] int (y, x)
        global_token: torch.Tensor,  # [B, E] or [E]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience API:
          Extract the 27×27 vision patch and append the provided global token.

        Args:
          grid_with_pe : [H, W, E] float (content+PE)
          pos          : [B, 2] int (y, x) per agent
          global_token : [B, E] or [E]. If [E], it will be expanded to [B, E].

        Returns:
          tokens_730   : [B, 730, E]  (729 grid + 1 global, last index is global)
          patch_mask   : [B, 729] bool  (True=in-bounds grid cell, False=OOB)

        Notes:
          • Global token is appended as the *final* token and is never masked
            by vision rules. If you later need a 730-length mask for attention,
            append a column of ones to patch_mask at call-site.
        """
        patch_tokens, patch_mask = self.extract_patch(grid_with_pe, pos)  # [B,729,E], [B,729]

        # Normalize the global token shape to [B, E]
        E = patch_tokens.shape[-1]
        device = patch_tokens.device
        dtype = patch_tokens.dtype

        if global_token.ndim == 1:
            # Single token shared across batch
            global_token = global_token.view(1, E).to(device=device, dtype=dtype)
            global_token = global_token.expand(patch_tokens.shape[0], E)  # [B,E]
        elif global_token.ndim == 2:
            if global_token.shape != (patch_tokens.shape[0], E):
                raise ValueError(f"global_token shape {tuple(global_token.shape)} "
                                 f"does not match (B,E)=({patch_tokens.shape[0]},{E})")
            global_token = global_token.to(device=device, dtype=dtype)
        else:
            raise ValueError("global_token must be [E] or [B,E].")

        # Concatenate along the token axis: [B, 729 + 1, E] = [B, 730, E]
        tokens_730 = torch.cat([patch_tokens, global_token.unsqueeze(1)], dim=1)  # [B,730,E]
        return tokens_730, patch_mask
