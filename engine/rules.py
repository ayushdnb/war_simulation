    # rules.py
# =============================================================================
# Engine contracts, enums, constants, and long-lived configuration.
# -----------------------------------------------------------------------------
# This module is the SINGLE SOURCE OF TRUTH for:
#   • Action space (indices, names, movement/attack split)
#   • Direction vectors, vision geometry, and edge topology
#   • Combat parameters (range, damage, cooldowns) and deterministic tie-breaks
#   • Occupancy/stacking policy (one agent per cell; no stacking)
#   • Infinite-match lifecycle knobs: spawning and evolution/mutation
#   • Canonical tensor dtypes + sentinel values used across the engine
#
# Design goals (5-year horizon):
#   • Stability: explicit enums, frozen dataclasses, zero hidden “magic numbers”
#   • Determinism: collisions, swaps, and multi-claims resolved by written rules
#   • Import safety: no device allocations; tiny CPU tensors for tables/stencils
#   • Readability: heavy comments; this file is where new contributors start
#
# NOTE: This version reflects an “infinite match” world. There is NO team-win
#       terminal reward. Agents optimize instantaneous reward every tick.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, Enum
from typing import Final, Tuple

import torch


# -----------------------------------------------------------------------------
# Versioning (bump on any breaking contract change)
# -----------------------------------------------------------------------------
RULES_VERSION: Final[str] = "1.1.0"  # 1.0.0 was finite match w/ team_win; now infinite


# -----------------------------------------------------------------------------
# Teams (stable IDs used everywhere: logs, tensors, file formats)
# -----------------------------------------------------------------------------
class Team(IntEnum):
    RED = 0
    BLUE = 1


TEAM_NAMES: Final[Tuple[str, str]] = ("RED", "BLUE")


# -----------------------------------------------------------------------------
# Topology: how the board handles edges
# -----------------------------------------------------------------------------
class Topology(str, Enum):
    HARD_EDGES = "hard_edges"  # out-of-bounds is invalid; no wrap
    WRAP = "wrap"              # torus (NOT used in v1; reserved for future)


# -----------------------------------------------------------------------------
# Directions and ACTION SPACE (17 logits total)
# -----------------------------------------------------------------------------
# Canonical 8-way order (NEVER change without bumping RULES_VERSION)
DIR_NAMES: Final[Tuple[str, ...]] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

# (dy, dx) in row-major coordinates:
#   dy: -1 up, +1 down
#   dx: -1 left, +1 right
DELTA_8: Final[Tuple[Tuple[int, int], ...]] = (
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
)

class Action(IntEnum):
    """
    Stable action indices (17). Do NOT renumber without bumping RULES_VERSION.
    Movement: 9 actions (including STAY)
    Attacks : 8 actions (directional melee)
    """
    # Movement (0..8)
    MOVE_STAY = 0
    MOVE_N = 1
    MOVE_NE = 2
    MOVE_E = 3
    MOVE_SE = 4
    MOVE_S = 5
    MOVE_SW = 6
    MOVE_W = 7
    MOVE_NW = 8

    # Attack (9..16)
    ATTACK_N = 9
    ATTACK_NE = 10
    ATTACK_E = 11
    ATTACK_SE = 12
    ATTACK_S = 13
    ATTACK_SW = 14
    ATTACK_W = 15
    ATTACK_NW = 16


ACTION_NAMES: Final[Tuple[str, ...]] = (
    "STAY",
    "MOVE_N", "MOVE_NE", "MOVE_E", "MOVE_SE", "MOVE_S", "MOVE_SW", "MOVE_W", "MOVE_NW",
    "ATTACK_N", "ATTACK_NE", "ATTACK_E", "ATTACK_SE", "ATTACK_S", "ATTACK_SW", "ATTACK_W", "ATTACK_NW",
)

# Fast boolean partitions (index-aligned)
ACTION_IS_MOVE: Final[Tuple[bool, ...]]   = tuple([True] * 9 + [False] * 8)
ACTION_IS_ATTACK: Final[Tuple[bool, ...]] = tuple([False] * 9 + [True] * 8)

# Movement action → direction index (0..7) or -1 for STAY
MOVE_ACT_TO_DIRIDX: Final[Tuple[int, ...]] = (
    -1,  # STAY (no direction)
    0, 1, 2, 3, 4, 5, 6, 7,  # MOVE_* → DIR 0..7
)

# Attack action → direction index (0..7); non-attack → -1
ATTACK_ACT_TO_DIRIDX: Final[Tuple[int, ...]] = tuple([-1] * 9 + list(range(8)))


# -----------------------------------------------------------------------------
# Vision geometry (Chebyshev disk → 27×27 crop with agent centered)
# -----------------------------------------------------------------------------
VISION_RADIUS_CHEB: Final[int] = 13
VISION_WINDOW_SIZE: Final[int] = VISION_RADIUS_CHEB * 2 + 1  # 27

def build_chebyshev_stencil(window_size: int) -> torch.Tensor:
    """
    Build a [window_size, window_size] boolean mask where (dy, dx) is True iff
    max(|dy|, |dx|) ≤ radius. Center is True; corners are True.
    """
    assert window_size % 2 == 1, "Vision window must be odd (agent centered)."
    r = window_size // 2
    ys = torch.arange(-r, r + 1)
    xs = torch.arange(-r, r + 1)
    dy = ys.view(-r*0 + window_size, 1).expand(window_size, window_size)  # expand avoids alloc per call
    dx = xs.view(1, -r*0 + window_size).expand(window_size, window_size)
    cheb = torch.maximum(dy.abs(), dx.abs())
    return (cheb <= r)

# Precomputed stencil (CPU, bool). Move/cast on demand at call sites.
VISION_STENCIL_CHEB: Final[torch.Tensor] = build_chebyshev_stencil(VISION_WINDOW_SIZE)


# -----------------------------------------------------------------------------
# Combat, timing, and collision policies
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CombatConfig:
    """
    Static combat knobs. All damage is applied SIMULTANEOUSLY each tick.
    Range is Chebyshev distance (range=1 = 8 neighbors).
    """
    hp_max: int = 100
    attack_range: int = 1
    attack_damage: int = 20
    attack_cooldown_ticks: int = 0  # 0 = may attack every tick
    move_cooldown_ticks: int = 0    # 0 = may move every tick


class KillCreditRule(str, Enum):
    """
    How to assign 'kill' reward when multiple attackers damage a victim in the
    SAME tick.
    """
    SPLIT_EVEN_THIS_TICK = "split_even_this_tick"  # even split among all attackers this tick
    LAST_HIT_ONLY = "last_hit_only"                # credit only the lethal-hit attacker


@dataclass(frozen=True)
class TieBreakPolicy:
    """
    Deterministic conflict resolution for movement intents.
    • No stacking is allowed (global occupancy invariant).
    • Perfect swaps (A↔B) may be allowed.
    • Multi-claim to same empty cell is resolved by a stable rule.
    """
    allow_perfect_swap: bool = True                # A->B & B->A allowed as a swap
    multi_claim: str = "lowest_agent_id"          # winner = smallest agent_id; others blocked

    # Explicit stacking policy (documented for clarity; enforced by engine):
    stacking_allowed: bool = False                # MUST remain False (one agent per cell)


# -----------------------------------------------------------------------------
# Reward scheme (infinite match ⇒ NO team_win)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardScheme:
    """
    Per-agent scalar coefficients used by the reward calculator.
    These are instantaneous (per tick) rewards; there is NO terminal team win.
    """
    kill: float = 1.0
    assist: float = 0.4
    damage_given: float = 0.1
    death: float = -1.0
    damage_taken: float = -0.05


# -----------------------------------------------------------------------------
# Infinite-match lifecycle knobs: spawning & evolution/mutation
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SpawnConfig:
    """
    Periodic agent spawning (births) for an infinite match.
    • Triggers every `interval_ticks`.
    • Spawns only for a team below its cap.
    • Placement strategy aims for "most populous friendly area" to create fronts.
    Implementation details (counts per wave, local neighborhood size) live in
    the spawn manager; this file only encodes policy + invariants/caps.
    """
    interval_ticks: int = 100
    max_agents_per_team: int = 1000
    placement_strategy: str = "most_populous_friendly_zone"  # policy name used by spawn manager


@dataclass(frozen=True)
class EvolutionConfig:
    """
    Structural mutation policy for team brains (architectural diversity).
    • Every `interval_ticks`, introduce a new VARIANT by mutating an existing one.
    • Mutation assigns the new variant to `population_fraction` (per team).
    • Each team may have at most `max_variants_per_team` live variants at once.
    • Underperforming variants naturally die out (selection handled by trainer).
    """
    interval_ticks: int = 300
    population_fraction: float = 0.10                # 10% of the team's current population
    max_variants_per_team: int = 4
    mutation_operator: str = "insert_single_hidden_layer_random_position"
    selection_rule: str = "natural_selection_reward_driven"  # trainer decides retire/expand


# -----------------------------------------------------------------------------
# Canonical tensor spec (shared dtypes + sentinel values)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorSpec:
    """
    Canonical dtypes & sentinels. Keep these consistent across all modules.
    """
    # Per-agent attributes
    pos_dtype: torch.dtype = torch.int16        # (y, x), fits maps up to ~32k
    team_dtype: torch.dtype = torch.int8        # 0/1
    hp_dtype: torch.dtype = torch.int16
    alive_dtype: torch.dtype = torch.bool
    cooldown_dtype: torch.dtype = torch.int8

    # Grid occupancy (agent_id or empty sentinel)
    occupancy_dtype: torch.dtype = torch.int32
    occupancy_empty: int = -1                   # “no agent in this cell”


# -----------------------------------------------------------------------------
# Global engine configuration (imported as CFG)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineConfig:
    """
    Frozen, import-only configuration object. Treat as read-only global.
    """
    version: str = RULES_VERSION
    topology: Topology = Topology.HARD_EDGES

    # Geometry
    vision_radius_cheb: int = VISION_RADIUS_CHEB
    vision_window_size: int = VISION_WINDOW_SIZE

    # Sub-configs
    combat: CombatConfig = CombatConfig()
    rewards: RewardScheme = RewardScheme()
    tiebreak: TieBreakPolicy = TieBreakPolicy()
    tensors: TensorSpec = TensorSpec()

    # Lifecycle (infinite match)
    spawn: SpawnConfig = SpawnConfig()
    evolution: EvolutionConfig = EvolutionConfig()

    # Kill credit policy (documented above)
    kill_credit_rule: KillCreditRule = KillCreditRule.SPLIT_EVEN_THIS_TICK


# Single canonical instance
CFG: Final[EngineConfig] = EngineConfig()


# -----------------------------------------------------------------------------
# Helper utilities (pure, side-effect-free)
# -----------------------------------------------------------------------------
def apply_topology(y: int, x: int, H: int, W: int, topology: Topology) -> Tuple[int, int, bool]:
    """
    Apply grid topology to (y, x).

    Returns:
        (y_adj, x_adj, out_of_bounds)
        • HARD_EDGES: oob=True when (y,x) is outside [0..H-1]×[0..W-1]; (y_adj,x_adj) unchanged
        • WRAP      : (y_adj, x_adj) wrapped mod H/W; oob=False
    """
    if topology == Topology.HARD_EDGES:
        oob = not (0 <= y < H and 0 <= x < W)
        return y, x, oob
    elif topology == Topology.WRAP:
        return (y % H), (x % W), False
    else:
        raise ValueError(f"Unknown topology: {topology}")


def is_move_action(a: int) -> bool:
    """True iff `a` is a movement action (STAY or directional move)."""
    return ACTION_IS_MOVE[a]


def is_attack_action(a: int) -> bool:
    """True iff `a` is an attack action."""
    return ACTION_IS_ATTACK[a]


def move_action_to_delta(a: int) -> Tuple[int, int]:
    """
    Convert a movement action to (dy, dx).
    STAY raises ValueError because it has no direction; handle at call site.
    """
    dir_idx = MOVE_ACT_TO_DIRIDX[a]
    if dir_idx == -1:
        raise ValueError("MOVE_STAY has no direction; do not call move_action_to_delta for STAY.")
    return DELTA_8[dir_idx]


def attack_action_to_delta(a: int) -> Tuple[int, int]:
    """
    Convert an attack action to (dy, dx). Raises if `a` is not an attack.
    """
    dir_idx = ATTACK_ACT_TO_DIRIDX[a]
    if dir_idx == -1:
        raise ValueError(f"Action {a} is not ATTACK_*")
    return DELTA_8[dir_idx]


# -----------------------------------------------------------------------------
# Lightweight self-checks (run at import)
# -----------------------------------------------------------------------------
def _run_self_checks() -> None:
    # Action table sizes
    assert len(ACTION_NAMES) == 17
    assert len(ACTION_IS_MOVE) == 17 and len(ACTION_IS_ATTACK) == 17

    # Partitions are disjoint and cover all indices
    for i in range(17):
        assert ACTION_IS_MOVE[i] ^ ACTION_IS_ATTACK[i], "Each action must be exactly one of {move, attack}"

    # Movement & attack mapping correctness
    assert MOVE_ACT_TO_DIRIDX[0] == -1
    assert tuple(MOVE_ACT_TO_DIRIDX[1:9]) == tuple(range(8))
    assert tuple(ATTACK_ACT_TO_DIRIDX[9:17]) == tuple(range(8))

    # Vision stencil shape & obvious truths
    assert VISION_STENCIL_CHEB.shape == (VISION_WINDOW_SIZE, VISION_WINDOW_SIZE)
    r = VISION_WINDOW_SIZE // 2
    assert bool(VISION_STENCIL_CHEB[r, r])          # center True
    assert bool(VISION_STENCIL_CHEB[0, 0])          # corner True
    assert bool(VISION_STENCIL_CHEB[0, r])          # edge True

    # Occupancy invariants (documented, enforced by engine)
    assert CFG.tiebreak.stacking_allowed is False, "Stacking MUST remain disabled (one agent per cell)."

    # Tensor dtypes & sentinel sanity
    spec = CFG.tensors
    assert spec.occupancy_empty == -1
    assert spec.pos_dtype == torch.int16
    assert spec.team_dtype == torch.int8
    assert spec.hp_dtype == torch.int16
    assert spec.occupancy_dtype == torch.int32

_run_self_checks()


# -----------------------------------------------------------------------------
# Public exports
# -----------------------------------------------------------------------------
__all__ = [
    # Version
    "RULES_VERSION",

    # Teams & topology
    "Team", "TEAM_NAMES", "Topology",

    # Directions & actions
    "DIR_NAMES", "DELTA_8",
    "Action", "ACTION_NAMES", "ACTION_IS_MOVE", "ACTION_IS_ATTACK",
    "MOVE_ACT_TO_DIRIDX", "ATTACK_ACT_TO_DIRIDX",

    # Vision
    "VISION_RADIUS_CHEB", "VISION_WINDOW_SIZE", "VISION_STENCIL_CHEB",

    # Configs
    "CombatConfig", "RewardScheme", "TieBreakPolicy", "TensorSpec",
    "SpawnConfig", "EvolutionConfig", "KillCreditRule",
    "EngineConfig", "CFG",

    # Helpers
    "apply_topology", "is_move_action", "is_attack_action",
    "move_action_to_delta", "attack_action_to_delta",
]
