# mechanics/spawn.py
# =============================================================================
# Periodic agent spawning for an infinite-match war simulation.
# -----------------------------------------------------------------------------
# What this module does
# ---------------------
# • When called (typically every CFG.spawn.interval_ticks), it adds a SMALL,
#   controlled number of new agents to each team, up to a per-team alive cap.
# • Spawn locations are chosen among EMPTY cells inside the most populous
#   friendly zones (dense regions of same-team presence) to create stable fronts.
# • The number of agents to spawn is inversely proportional to the team's
#   current alive population (more dead → more spawns), with tight per-wave caps.
# • Each spawned agent is assigned a brain variant for its team (uniform by
#   default; swap the sampler if you want share/EMA-driven allocation).
#
# What this module does NOT do
# ----------------------------
# • It does NOT decide when to spawn; the tick orchestrator decides and calls
#   this module (e.g., every 100 ticks).
# • It does NOT create brain variants or mutate networks (see evolution manager).
# • It does NOT handle any terrain/passability beyond "cell must be empty" —
#   v1 map is fully passable. Add terrain rules later if needed.
#
# Core invariants
# ---------------
# • One agent per cell: we only place on occupancy == EMPTY (no stacking).
# • Alive cap per team is enforced strictly (CFG.spawn.max_agents_per_team).
# • If no suitable empty cells exist (rare), we gracefully spawn fewer agents.
#
# Dependencies & expected state
# -----------------------------
# • We expect a `state` object with fields:
#     H, W                  : ints
#     pos[N,2]              : int16 (y, x)
#     team[N]               : int8  (0=RED, 1=BLUE)
#     brain_idx[N]          : int8  (0..3)
#     hp[N]                 : int16
#     alive[N]              : bool
#     cooldown_a[N]         : int8
#     occupancy[H,W]        : int32 (-1 empty else agent_id)
#     team_points[2]        : float32 (unused here; present for symmetry)
#     alive_count[2]        : int32
#     tick                  : int
# • We expect `CFG` from engine.rules and `Team` enum (RED=0, BLUE=1).
# • We expect a brain-pool-ish container (e.g., mechanics/tick.BrainPool) with
#   lists `variants_red` and `variants_blue`. We only need their lengths to pick
#   a variant index; the models themselves are used elsewhere.
#
# Performance notes
# -----------------
# • Density is computed via a small 2-D convolution on the team presence map.
# • All math runs on the state's device (CUDA if available).
# • We add a tiny jitter to break ties fairly among equally dense cells.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from engine.rules import CFG, Team


# -----------------------------------------------------------------------------
# Tunable policy knobs (kept local to keep CFG stable & focused on invariants)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SpawnPolicy:
    """
    SpawnPolicy holds gentle, non-critical knobs that shape spawn behavior.

    You can adjust these without touching core invariants in CFG:
      • kernel_size       : spatial window for "friend density" (odd int, e.g., 5 or 7)
      • rate_per_room     : fraction of free capacity to attempt per wave
      • min_per_wave      : lower bound on wave size (if room > 0)
      • max_per_wave      : safe upper bound on wave size
      • tie_break_jitter  : small noise added to scores to avoid ties
    """
    kernel_size: int = 7              # odd; typical 5 or 7
    rate_per_room: float = 0.05       # 5% of remaining room per wave
    min_per_wave: int = 1
    max_per_wave: int = 25
    tie_break_jitter: float = 1e-4    # tiny; breaks top-k ties


# Default policy instance (importers can pass their own if needed)
DEFAULT_POLICY = SpawnPolicy()


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
@torch.no_grad()
def spawn_wave(
    state,
    brain_pool,
    *,
    policy: SpawnPolicy = DEFAULT_POLICY,
    variant_sampler: Optional[Callable[[int, int], torch.Tensor]] = None,
) -> int:
    """
    Spawn a small number of agents for each team, up to the alive-cap.

    Args
    ----
    state : object with dynamic world tensors (see header "Dependencies").
    brain_pool : object with per-team variant lists (we read their lengths).
    policy : SpawnPolicy
        Non-critical knobs to shape the wave; invariants remain under CFG.
    variant_sampler : Optional[callable(team_id:int, count:int) -> LongTensor[count]]
        If provided, returns a tensor of variant indices for the team. If None,
        we assign variants uniformly among the team's available variants.

    Returns
    -------
    total_spawned : int
        Count of newly spawned agents across both teams.

    Notes
    -----
    • This function mutates `state` in place (pos/team/hp/alive/brain_idx/occupancy/alive_count).
    • Callers should check cadence (e.g., `if tick % CFG.spawn.interval_ticks == 0`).
    """
    device = state.occupancy.device
    total_new = 0

    for team_id in (Team.RED, Team.BLUE):
        # 1) How many can/should we spawn for this team?
        alive_now = int(torch.count_nonzero((state.team == team_id) & state.alive))
        room = max(0, CFG.spawn.max_agents_per_team - alive_now)
        if room <= 0:
            continue

        to_spawn = _compute_wave_size(room, policy)
        if to_spawn <= 0:
            continue

        # 2) Choose spawn positions in dense friendly zones (EMPTY cells only).
        yx = _choose_spawn_positions_dense(state, team_id, to_spawn, policy)
        if yx is None or yx.shape[0] == 0:
            continue

        # 3) Assign variants for the new agents.
        v_idx = _assign_variants(team_id, yx.shape[0], brain_pool, device, sampler=variant_sampler)

        # 4) Materialize new agents into state.
        _inject_agents(state, team_id, yx, v_idx)

        total_new += int(yx.shape[0])

    return total_new


# -----------------------------------------------------------------------------
# Wave size logic (inverse to population via "room")
# -----------------------------------------------------------------------------
def _compute_wave_size(room: int, policy: SpawnPolicy) -> int:
    """
    Convert available room (cap - alive) to a small wave size.

    A simple, robust policy:
      wave = clamp(int(room * rate_per_room), min_per_wave, max_per_wave)

    • When room is large, we add only a small slice (stability).
    • When room is small, we still try to add at least 1 (if room>0).
    • Hard-capped to avoid sudden population shocks.
    """
    if room <= 0:
        return 0
    wave = int(room * policy.rate_per_room)
    wave = max(policy.min_per_wave, min(policy.max_per_wave, wave))
    # If room itself is smaller than the wave, respect room strictly:
    return min(wave, room)


# -----------------------------------------------------------------------------
# Dense-zone placement (empty cells near friendly agents)
# -----------------------------------------------------------------------------
def _choose_spawn_positions_dense(state, team_id: int, count: int, policy: SpawnPolicy) -> Optional[torch.Tensor]:
    """
    Pick up to `count` empty cells that lie inside the densest friendly regions.

    Returns:
      yx : LongTensor[K, 2]  (K ≤ count), each row = (y, x)
      or None if no empty cells exist.

    Method:
      1) Build a binary presence map for this team: presence[y,x] = 1 if a
         friendly agent occupies (y,x), else 0.
      2) Convolve with a K×K ones kernel (K = policy.kernel_size) to get a
         density score per cell (how many friendlies nearby).
      3) Mask out non-empty cells; add tiny jitter to break ties fairly.
      4) Take top-K by score.

    Notes:
      • Edges are zero-padded in the convolution, which naturally biases
        centers slightly—acceptable and often desirable as a "home turf" bias.
      • If there are fewer empty cells than requested, we return as many as we can.
    """
    H, W = state.H, state.W
    device = state.occupancy.device
    occ = state.occupancy  # [H,W] int32

    # 1) Friendly presence map: 1 where this team occupies a cell; else 0.
    presence = torch.zeros((H, W), dtype=torch.float32, device=device)
    occ_valid = occ >= 0
    if occ_valid.any():
        ids = occ[occ_valid].to(torch.long)
        friendly = (state.team[ids] == team_id)
        presence[occ_valid] = friendly.to(torch.float32)

    # 2) Convolution with a K×K ones kernel (box filter) to get local friend count.
    k = policy.kernel_size
    assert k % 2 == 1 and k >= 3, "kernel_size must be odd and ≥ 3"
    kernel = torch.ones((1, 1, k, k), dtype=torch.float32, device=device)
    dens = F.conv2d(
        presence.view(1, 1, H, W), kernel, padding=k // 2
    ).view(H, W)  # [H,W] float32

    # 3) Mask: only empty cells are eligible. Add a tiny jitter for fair tie-breaking.
    empty_mask = (occ == CFG.tensors.occupancy_empty)
    if not torch.any(empty_mask):
        return None

    # Convert scores for empty cells; -inf for non-empty.
    scores = torch.where(empty_mask, dens, torch.full_like(dens, fill_value=-1e9))
    if policy.tie_break_jitter > 0:
        # Small noise (uniform in [-j, j]) to make top-k stable when equal.
        j = policy.tie_break_jitter
        scores = scores + (torch.rand_like(scores) * 2.0 - 1.0) * j

    # Flatten and select top-K empty cells by score.
    flat = scores.view(-1)                     # [H*W]
    k_req = min(count, int(torch.count_nonzero(empty_mask)))
    if k_req <= 0:
        return None

    topk = torch.topk(flat, k=k_req, largest=True, sorted=False)
    flat_idx = topk.indices  # [k_req]

    # Map flat indices back to (y, x)
    y = (flat_idx // W).to(torch.long)
    x = (flat_idx %  W).to(torch.long)
    yx = torch.stack([y, x], dim=1)  # [k_req,2]
    return yx


# -----------------------------------------------------------------------------
# Variant assignment
# -----------------------------------------------------------------------------
def _assign_variants(
    team_id: int,
    count: int,
    brain_pool,
    device: torch.device,
    sampler: Optional[Callable[[int, int], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Choose a controlling brain variant index for each new agent of `team_id`.

    Returns:
      LongTensor[count] with values in [0 .. num_variants-1].

    Default policy:
      • Uniform among existing variants for that team.
      • If no variants exist (shouldn't happen), returns zeros.
    """
    if count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    if sampler is not None:
        out = sampler(int(team_id), count)
        return out.to(device=device, dtype=torch.long)

    # Uniform fallback
    variants = brain_pool.variants_red if team_id == Team.RED else brain_pool.variants_blue
    n = max(1, len(variants))
    return torch.randint(low=0, high=n, size=(count,), device=device, dtype=torch.long)


# -----------------------------------------------------------------------------
# State mutation: append new agents and update occupancy/alive_count
# -----------------------------------------------------------------------------
def _inject_agents(
    state,
    team_id: int,
    yx: torch.Tensor,          # [K,2] long
    variant_idx: torch.Tensor, # [K] long
) -> None:
    """
    Append K new agents to state and update occupancy/alive_count.

    New agents start with:
      • hp       = CFG.combat.hp_max
      • alive    = True
      • cooldown = 0
      • team     = team_id
      • brain    = variant_idx[k]

    Notes:
      • We append along the agent dimension (cat). This is simple and fine
        for v1; you can move to preallocation/pooling later if needed.
      • Occupancy is updated immediately (one agent per empty cell).
    """
    device = state.occupancy.device
    K = int(yx.shape[0])
    if K == 0:
        return

    # 1) Build new agent attribute tensors.
    pos_new = yx.to(dtype=CFG.tensors.pos_dtype, device=device)  # [K,2]
    team_new = torch.full((K,), int(team_id), dtype=CFG.tensors.team_dtype, device=device)
    brain_new = variant_idx.to(dtype=torch.int8, device=device)  # fits 0..3
    hp_new = torch.full((K,), CFG.combat.hp_max, dtype=CFG.tensors.hp_dtype, device=device)
    alive_new = torch.ones((K,), dtype=CFG.tensors.alive_dtype, device=device)
    cd_new = torch.zeros((K,), dtype=CFG.tensors.cooldown_dtype, device=device)

    # 2) Append to state arrays.
    state.pos = torch.cat([state.pos, pos_new], dim=0)
    state.team = torch.cat([state.team, team_new], dim=0)
    state.brain_idx = torch.cat([state.brain_idx, brain_new], dim=0)
    state.hp = torch.cat([state.hp, hp_new], dim=0)
    state.alive = torch.cat([state.alive, alive_new], dim=0)
    state.cooldown_a = torch.cat([state.cooldown_a, cd_new], dim=0)

    # 3) Update occupancy with new agent_ids.
    #    New agent_ids are contiguous after old N.
    N_old = state.occupancy.max().item() + 1  # WARNING: not reliable if agents died & ids freed
    # Instead, derive N_old from previous length of state.team (before cat).
    # We can compute it from shapes:
    N_old = state.pos.shape[0] - K

    # Set occupancy[y,x] = agent_id for each new agent.
    # yx rows correspond to new agents in the same order we appended them.
    ys = yx[:, 0].to(torch.long)
    xs = yx[:, 1].to(torch.long)
    agent_ids = torch.arange(N_old, N_old + K, device=device, dtype=torch.long)
    state.occupancy[ys, xs] = agent_ids.to(CFG.tensors.occupancy_dtype)

    # 4) Refresh alive counts.
    state.alive_count[Team.RED]  = torch.count_nonzero((state.team == Team.RED)  & state.alive)
    state.alive_count[Team.BLUE] = torch.count_nonzero((state.team == Team.BLUE) & state.alive)
