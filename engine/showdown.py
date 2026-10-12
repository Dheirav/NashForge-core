from typing import List, Dict
from .hand_eval import evaluate_hand, compare_hands
from .hand_eval_fast import compare_hands_fast
from .state import PlayerState
from .pot import Pot


def _winner_indices(candidates: List[PlayerState], community_cards: List) -> List[int]:
    """
    Indices (within `candidates`) of the player(s) holding the best hand.

    A lone candidate wins by default and is never evaluated.  This matters
    because a hand won by folding can end before the board is complete, and the
    five-card comparator raises on fewer than five cards — so comparing a single
    survivor against nobody used to crash with
    "Need at least 5 cards, got 2" instead of simply paying them.
    """
    if len(candidates) == 1:
        return [0]
    return compare_hands_fast([p.hole_cards + community_cards for p in candidates])


def resolve_showdown(players: List[PlayerState], community_cards: List, pot: Pot, button: int = 0) -> Dict[int, int]:
    """
    Given a list of PlayerState, community cards, and the pot,
    returns a dict mapping player_id to amount won.
    Handles split pots and side pots.
    Odd chips go to first player clockwise from button.
    """
    winnings = {p.player_id: 0 for p in players}
    # Only active (not folded) players can win
    active_players = [p for p in players if not p.has_folded]
    if not active_players:
        return winnings
    
    # Build position order clockwise from button for odd chip assignment
    num_players = len(players)
    position_order = [(button + 1 + i) % num_players for i in range(num_players)]
    # For each side pot, determine eligible winners
    if pot.side_pots:
        for side in pot.side_pots:
            eligible = [p for p in active_players if p.player_id in side['eligible']]
            if not eligible:
                # No eligible players for this pot (all folded) - award to remaining active players
                # Fall back to awarding to all active players proportionally
                if active_players:
                    # Award to player(s) with best hand among all active
                    winner_idxs = _winner_indices(active_players, community_cards)
                    share = side['amount'] // len(winner_idxs)
                    for idx in winner_idxs:
                        winnings[active_players[idx].player_id] += share
                    remainder = side['amount'] % len(winner_idxs)
                    if remainder > 0:
                        winner_pids = [active_players[idx].player_id for idx in winner_idxs]
                        winner_pids_sorted = sorted(winner_pids, key=lambda pid: position_order.index(pid) if pid in position_order else pid)
                        for i in range(remainder):
                            winnings[winner_pids_sorted[i]] += 1
                continue
            winner_idxs = _winner_indices(eligible, community_cards)
            share = side['amount'] // len(winner_idxs)
            for idx in winner_idxs:
                winnings[eligible[idx].player_id] += share
            # If not divisible, assign remainder chips to lowest player_id(s)
            remainder = side['amount'] % len(winner_idxs)
            if remainder > 0:
                # Award odd chips to winners closest to button (clockwise)
                winner_pids = [eligible[idx].player_id for idx in winner_idxs]
                # Sort by position clockwise from button
                winner_pids_sorted = sorted(winner_pids, key=lambda pid: position_order.index(pid) if pid in position_order else pid)
                for i in range(remainder):
                    winnings[winner_pids_sorted[i]] += 1
    else:
        # No side pots: main pot only
        winner_idxs = _winner_indices(active_players, community_cards)
        share = pot.total // len(winner_idxs)
        for idx in winner_idxs:
            winnings[active_players[idx].player_id] += share
        remainder = pot.total % len(winner_idxs)
        if remainder > 0:
            # Award odd chips to winners closest to button (clockwise)
            winner_pids = [active_players[idx].player_id for idx in winner_idxs]
            winner_pids_sorted = sorted(winner_pids, key=lambda pid: position_order.index(pid) if pid in position_order else pid)
            for i in range(remainder):
                winnings[winner_pids_sorted[i]] += 1
    return winnings
