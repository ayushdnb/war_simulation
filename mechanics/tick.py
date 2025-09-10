# mechanics/tick.py
# =============================================================================
# One-tick orchestrator for the war simulation.
# -----------------------------------------------------------------------------
# Responsibilities (in this exact order):
#   1) Build full-grid embeddings once per tick (content + PE).
#   2) For each team/variant bucket:
#        - Extract 27×27 vision tokens (E_OOB for out-of-bounds),
#        - Build legal action masks,
#        - Run policy to get actions (logits → masked → sample or argmax).
#   3) Resolve movement (simultaneous), enforcing NO STACKING invariant.
#   4) Resolve attacks (simultaneous damage), update hp/deaths/occupancy.
#   5) Compute instantaneous rewards + update team points / stats.
#   6) Periodic mechanics:
#        - Spawner every 100 ticks (respecting caps & placement policy),
#        - Evolution every 300 ticks (≤ 4 variants; 10% share seed).
#
# What this file DOES NOT do:
#   • Implement movement/attacks/masks (engine modules do that).
#   • Implement PPO (trainer wraps this with rollouts/updates).
#
# Design choices:
#   • Pure orchestration; no model weights or training logic here.
#   • Contracts are explicit; failures are early and loud.
#   • Batch-first, GPU-friendly; no Python loops over agents in hot paths.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Engine contracts & constants
from engine.rules import CFG, Team

# Encoders
from enc.embed_map import MapEmbedder, DynamicSnapshot
from enc.posenc import sincos_posenc_2d, add_posenc_
from enc.global_token import GlobalTokenEmbedder, GlobalTokenInput

# Agent-side vision tokenizer (injects E_OOB; returns [B,729,E] + mask)
from brain.vision import VisionTokenizer

# Mechanics to be implemented in engine/*
import engine.action_mask as action_mask_mod
import engine.movement as movement_mod
import engine.attacks as attacks_mod


# -----------------------------------------------------------------------------
# State contracts (dynamic world state)
# -----------------------------------------------------------------------------
@dataclass
class WorldState:
    """
    Central dynamic tensors for the world. Kept on GPU for speed.

    Shapes & dtypes:
      N: number of agents in the world (alive or dead)
      H, W: map dimensions

    Agents (row-major agent_id = index):
      pos        : [N,2]  int16   (y, x)
      team       : [N]    int8    (0=RED, 1=BLUE)
      brain_idx  : [N]    int8    (0..3; which team variant controls agent)
      hp         : [N]    int16
      alive      : [N]    bool
      cooldown_a : [N]    int8    (optional; 0 in v1)

    Grids:
      occupancy  : [H,W]  int32   (=-1 empty; else agent_id)

    Team stats (updated each tick):
      team_points : [2]  float32  (cumulative; infinite match)
      alive_count : [2]  int32    (derived from `alive`)

    Tick counter:
      tick : int
    """
    H: int
    W: int
    pos: torch.Tensor
    team: torch.Tensor
    brain_idx: torch.Tensor
    hp: torch.Tensor
    alive: torch.Tensor
    cooldown_a: torch.Tensor
    occupancy: torch.Tensor
    team_points: torch.Tensor
    alive_count: torch.Tensor
    tick: int


@dataclass
class BrainPool:
    """
    Per-team registry of controlling brains (≤ 4 variants).

    Invariants:
      • len(variants[t]) <= CFG.evolution.max_variants_per_team
      • Each TeamBrain maps [B,730,E] → logits [B,17]
    """
    variants_red: List[nn.Module]
    variants_blue: List[nn.Module]


@dataclass
class TickHooks:
    """
    Optional callbacks triggered on schedule.

    All hooks are called AFTER combat & rewards for the tick complete.

    spawner(state): mutates state to add agents up to caps.
    evolution(state, pool): may mutate brain pool (add/retire variants).
    """
    spawner: Optional[callable] = None
    evolution: Optional[callable] = None


@dataclass
class TickIO:
    """
    Immutable inputs the tick runner needs beyond state itself.
    """
    static_map: object  # StaticMap (H,W,tile_type,passable_lut,topology='hard_edges')
    embedder: MapEmbedder
    global_token: GlobalTokenEmbedder
    vision: VisionTokenizer
    brain_pool: BrainPool

    # Device/dtype preferences for embeddings
    device: torch.device
    emb_dtype: torch.dtype = torch.float32


# -----------------------------------------------------------------------------
# Helper utilities (local to this module)
# -----------------------------------------------------------------------------
def _assert_state_shapes(state: WorldState) -> None:
    N = state.pos.shape[0]
    assert state.team.shape == (N,)
    assert state.brain_idx.shape == (N,)
    assert state.hp.shape == (N,)
    assert state.alive.shape == (N,)
    assert state.cooldown_a.shape == (N,)
    assert state.occupancy.shape == (state.H, state.W)
    assert state.team_points.shape == (2,)
    assert state.alive_count.shape == (2,)


def _build_dynamic_snapshot_from_state(state: WorldState) -> DynamicSnapshot:
    """
    Create the per-cell dynamic grids expected by the content embedder.
    owner: int8[H,W]  (0 empty, 1 RED, 2 BLUE)
    hp   : int8[H,W]  (0..hp_max; 0 if no agent)
    cooldown(optional): int8[H,W]
    """
    H, W = state.H, state.W
    device = state.occupancy.device

    owner = torch.zeros((H, W), dtype=torch.int8, device=device)
    hp_grid = torch.zeros((H, W), dtype=torch.int8, device=device)
    # cooldown omitted in v1; keep path open
    # cd_grid = torch.zeros((H, W), dtype=torch.int8, device=device)

    occ = state.occupancy  # [H,W] int32 (-1 empty else agent_id)
    valid = occ >= 0
    if valid.any():
        ag_ids = occ[valid].to(torch.long)
        # Map team 0/1 → owner 1/2
        owner_vals = (state.team[ag_ids] + 1).to(torch.int8)
        owner[valid] = owner_vals
        # Clamp hp to config hp_max
        hp_vals = torch.clamp(state.hp[ag_ids], 0, CFG.combat.hp_max).to(torch.int8)
        hp_grid[valid] = hp_vals

    return DynamicSnapshot(owner=owner, hp=hp_grid, cooldown=None)


def _team_indices(state: WorldState, team_id: int) -> torch.Tensor:
    """Return indices of agents that are alive and belong to team_id."""
    return torch.nonzero((state.team == team_id) & state.alive, as_tuple=False).view(-1)


def _split_by_variant(idx: torch.Tensor, brain_idx: torch.Tensor, max_k: int) -> List[torch.Tensor]:
    """
    Split indices into buckets by brain_idx ∈ [0..max_k-1].
    Returns list of index tensors (possibly empty).
    """
    buckets: List[torch.Tensor] = []
    for k in range(max_k):
        sel = idx[(brain_idx[idx] == k)]
        buckets.append(sel)
    return buckets


def _build_global_token_inputs(state: WorldState, idx: torch.Tensor) -> GlobalTokenInput:
    """Collect real-time scalars for the subset `idx` of agents."""
    y = state.pos[idx, 0].to(torch.int32)
    x = state.pos[idx, 1].to(torch.int32)
    hp = state.hp[idx].to(torch.int32)

    # Team points / alive counts (same scalar for all in the batch)
    red_pts = state.team_points[Team.RED].repeat(len(idx))
    blue_pts = state.team_points[Team.BLUE].repeat(len(idx))
    red_alive = state.alive_count[Team.RED].repeat(len(idx))
    blue_alive = state.alive_count[Team.BLUE].repeat(len(idx))

    return GlobalTokenInput(
        pos_y=y, pos_x=x, hp=hp,
        red_points=red_pts, blue_points=blue_pts,
        red_alive=red_alive, blue_alive=blue_alive,
    )


# -----------------------------------------------------------------------------
# Tick runner (one step)
# -----------------------------------------------------------------------------
class TickRunner:
    """
    Single-tick orchestrator. Construct once; call .step(state, io, hooks) each tick.

    The trainer (outside this file) wraps this in a rollout loop and handles:
      • storing transitions (obs, actions, logprobs, rewards),
      • PPO updates for each team variant,
      • logging/metrics/visualization.
    """

    def __init__(self):
        # Internal scratch to avoid reallocating PE every tick for same (H,W,E,device,dtype)
        self._last_pe_key: Optional[Tuple[int, int, int, torch.device, torch.dtype]] = None
        self._last_pe: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _full_grid_with_pe(
        self,
        state: WorldState,
        io: TickIO,
    ) -> torch.Tensor:
        """
        Compute full-grid content embedding and add positional encoding.

        Returns:
          grid_with_pe: [H, W, E] on io.device & io.emb_dtype
        """
        dyn = _build_dynamic_snapshot_from_state(state)
        grid_content = io.embedder.build_full_grid_embed(
            static_map=io.static_map,
            dynamic=dyn,
            device=io.device,
        )  # [H,W,E] dtype=io.embedder.out_dtype

        # Build/cached PE on demand
        H, W = state.H, state.W
        E = grid_content.shape[-1]
        pe_key = (H, W, E, io.device, io.emb_dtype)
        if self._last_pe_key != pe_key:
            pe_cpu = sincos_posenc_2d(H, W, E, device='cpu', dtype=torch.float32)
            self._last_pe = pe_cpu.to(device=io.device, dtype=io.emb_dtype)
            self._last_pe_key = pe_key

        grid_with_pe = grid_content.to(dtype=io.emb_dtype, device=io.device)
        add_posenc_(grid_with_pe, self._last_pe)  # in-place add
        return grid_with_pe

    @torch.no_grad()
    def _policy_phase(
        self,
        state: WorldState,
        io: TickIO,
        grid_with_pe: torch.Tensor,  # [H,W,E]
    ) -> torch.Tensor:
        """
        Run action selection for all alive agents.

        Returns:
          actions [N] int64 (0..16). Dead agents receive Action.MOVE_STAY (0).
        """
        N = state.pos.shape[0]
        device = io.device
        actions = torch.zeros((N,), dtype=torch.int64, device=device)

        # For each team separately (policies are independent)
        for team_id, variants in ((Team.RED, io.brain_pool.variants_red),
                                  (Team.BLUE, io.brain_pool.variants_blue)):
            idx_team = _team_indices(state, team_id)
            if idx_team.numel() == 0:
                continue

            # Build legal action mask for this subset
            legal_mask = action_mask_mod.build_legal_action_mask(
                state=state, subset_idx=idx_team
            )  # [B_t, 17] bool

            # Build 730-token inputs via vision + global token
            pos_sub = state.pos[idx_team]  # [B_t,2]
            patch_tokens, _patch_mask = io.vision.extract_patch(grid_with_pe, pos_sub)  # [B_t,729,E], [B_t,729]

            g_inputs = _build_global_token_inputs(state, idx_team)
            g_tok = io.global_token(io.static_map, g_inputs, device=device)  # [B_t,E]

            tokens_730, _ = io.vision.build_agent_tokens(
                grid_with_pe, pos_sub, g_tok
            )  # [B_t,730,E], [B_t,729] (mask not used here)

            # Split by variant id (0..3) and run the corresponding brain
            buckets = _split_by_variant(idx_team, state.brain_idx, CFG.evolution.max_variants_per_team)
            for k, ids in enumerate(buckets):
                if ids.numel() == 0:
                    continue
                if k >= len(variants):
                    # Variant slot unused in this team
                    continue

                # Select this bucket’s slice
                bsel = (idx_team.unsqueeze(0) == ids.view(-1, 1)).any(dim=0)  # mask over idx_team
                tok_slice = tokens_730[bsel]           # [B_k,730,E]
                legal_slice = legal_mask[bsel]         # [B_k,17]

                # Your current TeamBrain expects flattened input; reshape here.
                # If you later move to a transformer brain, change only this block.
                flat = tok_slice.view(tok_slice.size(0), -1)  # [B_k, 730*E]
                logits = variants[k](flat, action_mask=None)  # [B_k,17]

                # Impose legality (hard mask)
                logits = logits.masked_fill(~legal_slice.bool(), -1e9)

                # Action selection policy: argmax for now (replace with sampling if desired)
                chosen = torch.argmax(logits, dim=-1)  # [B_k]

                # Scatter into global actions array
                actions[ids] = chosen.to(torch.int64)

        # Dead agents: keep them at STAY (0)
        actions = torch.where(state.alive, actions, torch.zeros_like(actions))
        return actions

    @torch.no_grad()
    def _movement_phase(self, state: WorldState, actions: torch.Tensor) -> None:
        """
        Resolve simultaneous movement; enforce no stacking invariant.

        Mutates state.pos and state.occupancy in place.
        """
        movement_mod.resolve_moves(state=state, actions=actions)

    @torch.no_grad()
    def _attack_phase(self, state: WorldState, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Resolve simultaneous attacks & damage.

        Mutates state.hp, state.alive, state.occupancy.
        Returns a small dict of per-agent logs to support reward calc.
          expected keys (engine decides exact names):
            - 'damage_given'[N] float32
            - 'damage_taken'[N] float32
            - 'kill_flag'[N]    float32 (1 for killers this tick else 0)
            - 'assist_flag'[N]  float32 (1 for assisters this tick else 0)
            - 'death_flag'[N]   float32 (1 if died this tick else 0)
        """
        logs = attacks_mod.resolve_attacks(state=state, actions=actions)
        return logs

    @torch.no_grad()
    def _rewards_and_stats(
        self,
        state: WorldState,
        attack_logs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute instantaneous reward per agent based on engine CFG.rewards,
        update team points and alive_count. Returns reward[N] float32.
        """
        r_cfg = CFG.rewards
        device = state.pos.device

        dmg_g = attack_logs.get("damage_given", torch.zeros_like(state.hp, dtype=torch.float32, device=device))
        dmg_t = attack_logs.get("damage_taken", torch.zeros_like(state.hp, dtype=torch.float32, device=device))
        kill  = attack_logs.get("kill_flag",   torch.zeros_like(state.hp, dtype=torch.float32, device=device))
        asst  = attack_logs.get("assist_flag", torch.zeros_like(state.hp, dtype=torch.float32, device=device))
        death = attack_logs.get("death_flag",  torch.zeros_like(state.hp, dtype=torch.float32, device=device))

        reward = (
            r_cfg.damage_given * dmg_g +
            r_cfg.damage_taken * dmg_t +
            r_cfg.kill         * kill  +
            r_cfg.assist       * asst  +
            r_cfg.death        * death
        ).to(torch.float32)

        # Aggregate to team points (simple sum of rewards; adjust if you want only positives)
        for t in (Team.RED, Team.BLUE):
            team_mask = (state.team == t)
            state.team_points[t] += reward[team_mask].sum()

        # Refresh alive counts
        state.alive_count[Team.RED]  = torch.count_nonzero((state.team == Team.RED)  & state.alive)
        state.alive_count[Team.BLUE] = torch.count_nonzero((state.team == Team.BLUE) & state.alive)

        return reward

    @torch.no_grad()
    def step(
        self,
        state: WorldState,
        io: TickIO,
        hooks: Optional[TickHooks] = None,
    ) -> Dict[str, object]:
        """
        Execute one full simulation tick. Mutates `state` in-place.

        Returns a small dict of scalars/tensors needed for logging or training:
          {
            'tick': int,
            'team_points': [2] float,
            'alive_count': [2] int,
            'actions': [N] int64,
            'reward': [N] float32,
          }
        """
        _assert_state_shapes(state)
        hooks = hooks or TickHooks()

        # 1) Build full-grid content+PE once per tick
        grid_with_pe = self._full_grid_with_pe(state, io)  # [H,W,E]

        # 2) Policy phase → actions[N]
        actions = self._policy_phase(state, io, grid_with_pe)

        # 3) Movement phase (simultaneous resolution; no stacking)
        self._movement_phase(state, actions)

        # 4) Attack phase (simultaneous damage & deaths)
        logs = self._attack_phase(state, actions)

        # 5) Rewards & stats (instantaneous, infinite match)
        reward = self._rewards_and_stats(state, logs)

        # 6) Scheduled mechanics (after combat & rewards)
        state.tick += 1
        if CFG.spawn.interval_ticks > 0 and (state.tick % CFG.spawn.interval_ticks == 0):
            if hooks.spawner is not None:
                hooks.spawner(state)

        if CFG.evolution.interval_ticks > 0 and (state.tick % CFG.evolution.interval_ticks == 0):
            if hooks.evolution is not None:
                hooks.evolution(state, io.brain_pool)

        return {
            "tick": state.tick,
            "team_points": state.team_points.clone(),
            "alive_count": state.alive_count.clone(),
            "actions": actions,           # keep on device
            "reward": reward,             # keep on device
        }
