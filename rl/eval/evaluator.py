"""
PPO Evaluator.

evaluate_vs_pool()   – pit a PPOAgent against a list of opponents (or RandomOpponent)
                       and return win% + BB/100.

run_tournament()     – head-to-head round-robin between any list of agents that
                       implement .get_action(game, player_id) -> int.
                       Accepts mixed pools: PPOAgent, AgentPlayer (evolution),
                       RandomOpponent, etc.

Both functions are engine-agnostic — they use engine.PokerGame directly and
do not depend on either training/ or rl/ppo/.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from engine import PokerGame
from training.fitness import abstract_action_to_engine_action


def _safe_mask(game, pid: int) -> "np.ndarray":
    """Return 6-slot abstract action mask for pid."""
    from rl.poker_env import get_abstract_action_mask
    return get_abstract_action_mask(game, pid)


# ---------------------------------------------------------------------------
# Low-level hand runner
# ---------------------------------------------------------------------------

def _play_one_hand(
    game: PokerGame,
    agents: List[Any],       # one per seat, .get_action(game, pid) -> int
    big_blind: int,
) -> Dict[int, int]:
    """
    Play a single poker hand to completion.

    Returns chip_deltas: {player_id: chip_delta_from_start}
    """
    start_stacks = {p.player_id: p.stack for p in game.players}

    while game.state.betting_round != "showdown":
        active = [p for p in game.players if not p.has_folded]
        if len(active) <= 1:
            break

        pid = game.state.current_player
        if pid is None:
            break

        # Mask-safe agent call
        mask  = _safe_mask(game, pid)
        agent = agents[pid]
        try:
            action_idx = agent.get_action(game, pid)
        except Exception:
            action_idx = 1   # fallback: check/call

        if not mask[action_idx]:
            legal = np.where(mask)[0]
            action_idx = int(legal[0]) if len(legal) else 1

        eng_action = abstract_action_to_engine_action(action_idx, game, pid)
        game.apply_action(pid, eng_action)

    # Resolve showdown if round reached it
    if game.state.betting_round == "showdown":
        from engine import resolve_showdown
        resolve_showdown(
            game.players,
            game.state.community_cards,
            game.state.pot,
            game.state.button,
        )

    final_stacks = {p.player_id: p.stack for p in game.players}
    return {pid: final_stacks[pid] - start_stacks[pid] for pid in start_stacks}


# ---------------------------------------------------------------------------
# evaluate_vs_pool
# ---------------------------------------------------------------------------

def evaluate_vs_pool(
    agent:      Any,
    opponents:  Optional[List[Any]],
    num_hands:  int              = 500,
    cfg:        Optional[Any]    = None,
    num_players:    int          = 2,
    starting_stack: int          = 1_000,
    small_blind:    int          = 5,
    big_blind:      int          = 10,
    seed:           Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate a single agent against a pool of opponents.

    The agent always occupies seat 0.  Opponents are sampled round-robin
    (or randomly if the pool has multiple entries) for each hand.

    Parameters
    ----------
    agent:       Any object with .get_action(game, pid) -> int.
    opponents:   List of opponent objects.  None → uses RandomOpponent.
    num_hands:   Hands to play per evaluation.
    cfg:         Optional PPOConfig (to read num_players etc.; overrides kwargs).
    num_players, starting_stack, small_blind, big_blind, seed: env params.

    Returns
    -------
    dict with keys: win_pct, bb_per_100, total_hands, chip_delta
    """
    from rl.agents import RandomOpponent

    # Parse config if provided
    if cfg is not None:
        num_players    = cfg.num_players
        starting_stack = cfg.starting_stack
        small_blind    = cfg.small_blind
        big_blind      = cfg.big_blind

    if opponents is None or len(opponents) == 0:
        opponents = [RandomOpponent() for _ in range(num_players - 1)]

    rng          = np.random.default_rng(seed)
    chip_total   = 0
    wins         = 0

    for hand_idx in range(num_hands):
        # Sample opponents for this hand
        sampled_opps = []
        for _ in range(num_players - 1):
            idx = int(rng.integers(0, len(opponents)))
            sampled_opps.append(opponents[idx])

        seats = [agent] + sampled_opps
        stacks = [starting_stack] * num_players
        hand_seed = int(rng.integers(0, 2**31))

        game = PokerGame(
            player_stacks = stacks,
            small_blind   = small_blind,
            big_blind     = big_blind,
            seed          = hand_seed,
            enable_history = False,
        )

        deltas = _play_one_hand(game, seats, big_blind)
        agent_delta = deltas.get(0, 0)
        chip_total += agent_delta
        if agent_delta > 0:
            wins += 1

    return {
        "win_pct":    100.0 * wins / num_hands,
        "bb_per_100": (chip_total / big_blind) / num_hands * 100,
        "total_hands": num_hands,
        "chip_delta":  chip_total,
    }


# ---------------------------------------------------------------------------
# run_tournament (round-robin over any agent list)
# ---------------------------------------------------------------------------

def run_tournament(
    agents:         Dict[str, Any],
    num_hands:      int   = 200,
    num_players:    int   = 2,
    starting_stack: int   = 1_000,
    small_blind:    int   = 5,
    big_blind:      int   = 10,
    seed:           Optional[int] = None,
    verbose:        bool  = True,
) -> List[Dict]:
    """
    Head-to-head round-robin tournament.

    Parameters
    ----------
    agents:  Dict mapping name → agent object (any .get_action interface).
    Returns list of result dicts sorted by bb_per_100 descending.

    Example
    -------
    >>> from rl.eval.evaluator import run_tournament
    >>> results = run_tournament({
    ...     "ppo_v1":  ppo_agent,
    ...     "evo_hof": evo_agent,
    ...     "random":  RandomOpponent(),
    ... })
    """
    names = list(agents.keys())
    n     = len(names)

    # Accumulate bb/100 over all matchups for each agent
    totals: Dict[str, Dict] = {
        name: {"chip_delta": 0, "total_hands": 0, "wins": 0}
        for name in names
    }

    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if verbose:
        print(f"[Tournament] {n} agents, {len(pairs)} matchups × {num_hands} hands each")

    for idx, (ai, bi) in enumerate(pairs):
        name_a, name_b = names[ai], names[bi]
        agent_a = agents[name_a]
        agent_b = agents[name_b]

        res = evaluate_vs_pool(
            agent       = agent_a,
            opponents   = [agent_b],
            num_hands   = num_hands,
            num_players = num_players,
            starting_stack = starting_stack,
            small_blind    = small_blind,
            big_blind      = big_blind,
            seed           = (seed + idx) if seed is not None else None,
        )

        totals[name_a]["chip_delta"]  += res["chip_delta"]
        totals[name_a]["total_hands"] += res["total_hands"]
        totals[name_a]["wins"]        += int(res["win_pct"] > 50)

        if verbose:
            print(f"  {name_a:30s} vs {name_b:30s}  BB/100={res['bb_per_100']:+.2f}  win%={res['win_pct']:.1f}")

    # Build final leaderboard
    results = []
    for name in names:
        t       = totals[name]
        hands   = max(t["total_hands"], 1)
        bb100   = (t["chip_delta"] / big_blind) / hands * 100
        results.append({
            "name":        name,
            "bb_per_100":  bb100,
            "total_hands": t["total_hands"],
            "matchup_wins": t["wins"],
        })

    results.sort(key=lambda x: x["bb_per_100"], reverse=True)

    if verbose:
        print("\n── Leaderboard ──────────────────────────────────────")
        for rank, r in enumerate(results, 1):
            print(f"  #{rank:2d}  {r['name']:30s}  BB/100={r['bb_per_100']:+7.2f}")

    return results
