import random
from .cards import Deck
from .state import GameState, PlayerState
from .actions import Action
from .pot import Pot
from .showdown import resolve_showdown
from typing import List


class PokerGame:
    def _advance_turn(self):
        """Advance to the next player who needs to act, or end the round."""
        num_players = len(self.state.players)
        
        # Check if betting round should end
        if self._should_end_betting_round():
            self._start_next_round()
            return
        
        # Find next player who can act
        for i in range(1, num_players + 1):
            next_idx = (self.state.current_player + i) % num_players
            player = self.state.players[next_idx]
            
            # Skip folded or all-in players
            if player.has_folded or player.is_all_in:
                continue
            
            # This player needs to act if:
            # 1. They haven't acted this round yet, OR
            # 2. Their bet is less than the current max bet
            current_max_bet = max(p.bet for p in self.state.players if not p.has_folded)
            if next_idx not in self.acted_since_last_raise or player.bet < current_max_bet:
                self.state.current_player = next_idx
                return
        
        # No one left to act, end the round
        self._start_next_round()

    def _should_end_betting_round(self):
        """Check if betting round should end."""
        active = [p for p in self.state.players if not p.has_folded]
        
        # If only one player left, round ends
        if len(active) <= 1:
            return True
        
        # If all active players are all-in, round ends
        if all(p.is_all_in for p in active):
            return True
        
        # Get players who can still act (not folded, not all-in)
        can_act = [p for p in active if not p.is_all_in]
        if not can_act:
            return True
        
        # Check if all players who can act have acted and bets are equal
        max_bet = max(p.bet for p in active)
        all_acted = all(p.player_id in self.acted_since_last_raise for p in can_act)
        all_bets_equal = all(p.bet == max_bet for p in can_act)
        
        # BB option: if preflop, no raises, and BB hasn't acted yet, don't end round
        if (self.state.betting_round == "preflop" and 
            self.bb_has_option and 
            self.bb_position is not None and
            self.bb_position not in self.acted_since_last_raise and
            not self.players[self.bb_position].has_folded and
            not self.players[self.bb_position].is_all_in):
            return False
        
        return all_acted and all_bets_equal

    def _start_next_round(self):
        """Reset bets and advance to the next betting round."""
        # Reset per-round bets
        for p in self.state.players:
            p.bet = 0
        self.current_bet = 0
        self.last_raise_size = self.state.big_blind  # Reset min raise to BB
        self.acted_since_last_raise.clear()
        self.bb_has_option = False  # BB option only applies preflop
        
        # Advance game stage
        prev_round = self.state.betting_round
        self.state.advance_stage()
        
        # Deal community cards as appropriate (with burn cards)
        if prev_round == 'preflop':
            self.deck.deal(1)  # Burn card
            self.state.community_cards += self.deck.deal(3)  # Flop
        elif prev_round == 'flop':
            self.deck.deal(1)  # Burn card
            self.state.community_cards += self.deck.deal(1)  # Turn
        elif prev_round == 'turn':
            self.deck.deal(1)  # Burn card
            self.state.community_cards += self.deck.deal(1)  # River
        
        # Set first player to act in new round (or None if showdown)
        if self.state.betting_round != 'showdown':
            self.state.current_player = self._first_to_act(self.state.betting_round)
            # If no one can act (all all-in), keep advancing
            if self.state.current_player is None:
                self._start_next_round()
        else:
            self.state.current_player = None
    def __init__(self, player_stacks: List[int], small_blind: int, big_blind: int, 
                 ante: int = 0, seed: int = None, enable_history: bool = False):
        self.players = [PlayerState(i, stack, i) for i, stack in enumerate(player_stacks)]
        self.state = GameState(self.players, button=0, small_blind=small_blind, big_blind=big_blind, seed=seed)
        self.state.ante = ante  # Store ante amount
        self.deck = Deck(seed)

        # Hands dealt by THIS game object.  Reset here rather than probed with
        # hasattr() so a game recycled through a pool starts from the same state
        # as a fresh one — otherwise a recycled game kept the attribute and
        # opened on button 1 while a fresh game opened on button 0.
        self._hands_dealt = 0
        # Draws a fresh deck seed for each subsequent hand, deterministically
        # from the master seed.  See reset_hand().
        self._deck_rng = random.Random(seed)

        self.current_bet = 0
        self.last_raiser = None
        self.last_raise_size = 0  # Track the size of the last raise for min-raise enforcement
        self.acted_since_last_raise = set()
        self.bb_has_option = False  # Track if BB still has option to raise preflop
        self.bb_position = None  # Track BB position for option
        
        # Action history for hand logging (disabled during training for speed)
        self.enable_history = enable_history
        self.action_history = [] if enable_history else None

        self.reset_hand()

    # ---------- HAND SETUP ----------

    def reset_hand(self):
        # Advance button position for new hand (except first hand)
        # Skip over busted players when moving button
        if self._hands_dealt > 0:
            n = len(self.players)
            new_button = (self.state.button + 1) % n
            # Find next player with chips for button
            for _ in range(n):
                if self.players[new_button].stack > 0:
                    break
                new_button = (new_button + 1) % n
            self.state.button = new_button

        # Deal a DIFFERENT deck each hand.
        #
        # This used to be `Deck(self.state.deck_seed)` — the same seed every
        # time — so every hand of a session got an identical shuffle and only
        # the button rotation varied.  A session therefore contained just two
        # distinct deals, alternating.  Anything that played hands by looping
        # reset_hand() (self_play.play_match, and so every round-robin
        # tournament, plus the GUI) was scoring agents on two repeated hands.
        #
        # The first hand still uses the master seed, so `PokerGame(seed=s)`
        # deals exactly what it always did; later hands draw their seed from an
        # RNG that is itself seeded from the master, keeping whole sessions
        # reproducible.
        if self.state.deck_seed is None:
            deck_seed = None                      # unseeded: fresh entropy per hand
        elif self._hands_dealt == 0:
            deck_seed = self.state.deck_seed
        else:
            deck_seed = self._deck_rng.randrange(2 ** 31)
        self.deck = Deck(deck_seed)
        self._hands_dealt += 1

        for p in self.players:
            p.reset_for_new_hand()
            # Mark busted players as folded so they can't act
            if p.stack == 0:
                p.has_folded = True

        self.state.community_cards = []
        self.state.pot.reset()
        self.state.betting_round = "preflop"
        self.state.betting_history = []

        self.deal_hole_cards()
        self.post_blinds()

        self.current_bet = self.state.big_blind
        self.last_raiser = None
        self.last_raise_size = self.state.big_blind  # Initial raise size is BB
        self.acted_since_last_raise = set()
        if self.enable_history:
            self.action_history = []  # Reset action history for new hand
        
        # Track BB position for option
        _, self.bb_position = self._blind_positions()
        self.bb_has_option = True  # BB gets option if no raises preflop

        self.state.current_player = self._first_to_act("preflop")

    def _blind_positions(self):
        """
        Seats posting the small and big blind, skipping busted players.

        Blinds used to be assigned by seat offset alone, so a seat with no chips
        could be handed one — it then "posted" zero, which both left the pot
        short and marked a busted player all-in.  With enough busted seats a hand
        could begin with nobody having posted anything at all.

        Returns (sb_seat, bb_seat); either may be None if too few players are
        funded to make a hand.
        """
        n = len(self.players)
        funded = [(self.state.button + i) % n for i in range(n)
                  if self.players[(self.state.button + i) % n].stack > 0]

        if len(funded) < 2:
            return (funded[0] if funded else None), None

        if len(funded) == 2:
            # Heads-up: the button posts the small blind.
            return funded[0], funded[1]

        # Three or more: small blind is the first funded seat left of the button,
        # big blind the next one after that.
        left_of_button = [s for s in funded if s != self.state.button] or funded
        return left_of_button[0], left_of_button[1 % len(left_of_button)]

    def deal_hole_cards(self):
        # Deal cards starting left of button, one at a time (standard order)
        n = len(self.players)
        for card_num in range(2):  # Deal 2 cards
            for i in range(n):
                pos = (self.state.button + 1 + i) % n
                p = self.players[pos]
                # Skip busted players
                if p.stack == 0 and p.has_folded:
                    continue
                p.hole_cards.extend(self.deck.deal(1))

    def post_blinds(self):
        n = len(self.players)
        
        # Post antes first (if configured)
        ante = getattr(self.state, 'ante', 0)
        if ante > 0:
            for i in range(n):
                p = self.players[i]
                if p.stack > 0 and not p.has_folded:
                    ante_amt = min(ante, p.stack)
                    p.stack -= ante_amt
                    p.total_contributed += ante_amt
                    self.state.pot.add_contribution(i, ante_amt)
        
        sb, bb = self._blind_positions()

        if sb is not None:
            self.state.pot.add_contribution(sb, self.players[sb].post_blind(self.state.small_blind))
        if bb is not None:
            self.state.pot.add_contribution(bb, self.players[bb].post_blind(self.state.big_blind))

    # ---------- TURN ORDER ----------

    def _first_to_act(self, street):
        n = len(self.players)
        if n == 2:
            # Heads-up: button/SB acts first preflop, BB acts first post-flop
            if street == "preflop":
                pos = self.state.button  # Button (who is SB) acts first
            else:
                pos = (self.state.button + 1) % n  # BB acts first post-flop
        else:
            # Standard: UTG (button+3) preflop, SB (button+1) post-flop
            pos = (self.state.button + (3 if street == "preflop" else 1)) % n
        for i in range(n):
            idx = (pos + i) % n
            p = self.players[idx]
            if not p.has_folded and not p.is_all_in:
                return idx
        return None

    # NOTE: a second, unreachable street-advancement path used to live here
    # (`_reset_bets_for_new_round` / `next_betting_round` / `_end_betting_round`
    # / `betting_closed` / `_next_player`).  It duplicated the card-dealing
    # logic of `_start_next_round()` with subtly different bookkeeping — it
    # never reset `last_raise_size` or `bb_has_option` — so calling it would
    # have corrupted min-raise enforcement.  Nothing referenced it; removed so
    # `_advance_turn` → `_should_end_betting_round` → `_start_next_round`
    # is the only way a street advances.

    # ---------- CORE ACTION LOGIC ----------

    def get_legal_actions(self, player_idx: int):
        """
        Returns a list of legal actions for the given player.
        Each action is a dict with 'type' and optional 'min'/'max' for amounts.
        """
        player = self.state.players[player_idx]
        legal = []
        
        if player.has_folded or player.is_all_in or player.stack == 0:
            return legal
        
        current_max_bet = max((p.bet for p in self.state.players if not p.has_folded), default=0)
        to_call = current_max_bet - player.bet
        min_raise = max(self.last_raise_size, self.state.big_blind)
        
        # Fold is always legal
        legal.append({'type': 'fold'})
        
        # Check is legal if no bet to call
        if to_call == 0:
            legal.append({'type': 'check'})
        
        # Call is legal if there's a bet to call and player has chips
        if to_call > 0 and player.stack > 0:
            legal.append({'type': 'call', 'amount': min(to_call, player.stack)})
        
        # Raise is legal only if the player can actually reach the minimum
        # raise-to amount.  Having *some* chips beyond the call is not enough:
        # if the stack falls short of a full min-raise the only aggressive
        # action available is an all-in, which is listed separately below.
        # (Advertising such a raise produced entries with min > max.)
        min_raise_to = current_max_bet + min_raise
        max_raise_to = player.bet + player.stack
        if player.stack > to_call and max_raise_to >= min_raise_to:
            legal.append({'type': 'raise', 'min': min_raise_to, 'max': max_raise_to})
        
        # All-in is always legal if player has chips
        if player.stack > 0:
            legal.append({'type': 'all-in', 'amount': player.stack})
        
        return legal

    def is_action_legal(self, player_idx: int, action) -> bool:
        """Check if an action is legal for the given player."""
        player = self.state.players[player_idx]
        
        # Can't act if not your turn
        if self.state.current_player != player_idx:
            return False
        
        # Can't act if folded, all-in, or busted
        if player.has_folded or player.is_all_in or player.stack == 0:
            return False
        
        current_max_bet = max((p.bet for p in self.state.players if not p.has_folded), default=0)
        to_call = current_max_bet - player.bet
        min_raise = max(self.last_raise_size, self.state.big_blind)
        
        if action.action_type == 'fold':
            return True
        elif action.action_type == 'check':
            return to_call == 0
        elif action.action_type == 'call':
            return to_call > 0
        elif action.action_type == 'raise':
            if player.stack <= to_call:
                return False
            # Allow any raise amount (will be adjusted to min if too low)
            return True
        elif action.action_type == 'all-in':
            return player.stack > 0
        
        return False

    def apply_action(self, player_idx: int, action):
        """Apply an action for the given player. All state mutation happens here."""
        player = self.state.players[player_idx]
        
        # Validate action is legal
        if not self.is_action_legal(player_idx, action):
            raise ValueError(f"Illegal action {action} for player {player_idx}")
        
        # Determine the required amount to call
        current_max_bet = max((p.bet for p in self.state.players if not p.has_folded), default=0)
        to_call = current_max_bet - player.bet
        
        # Calculate minimum raise (must be at least the size of the last raise)
        min_raise = max(self.last_raise_size, self.state.big_blind)
        
        # Record action for history (before mutation for accurate amounts)
        action_amount = getattr(action, 'amount', 0) or 0

        # Handle folds
        if action.action_type == "fold":
            player.has_folded = True
            self.acted_since_last_raise.add(player_idx)
            if self.enable_history:
                self.action_history.append((player_idx, action.action_type, 0, self.state.betting_round))
            # If BB folds, they used their option
            if player_idx == self.bb_position:
                self.bb_has_option = False
            self._advance_turn()
            return

        # Handle checks
        elif action.action_type == "check":
            # Check is only valid if no bet to call
            self.acted_since_last_raise.add(player_idx)
            if self.enable_history:
                self.action_history.append((player_idx, action.action_type, 0, self.state.betting_round))
            # If BB checks preflop, they used their option
            if player_idx == self.bb_position and self.state.betting_round == "preflop":
                self.bb_has_option = False

        # Handle calls
        elif action.action_type == "call":
            call_amount = min(to_call, player.stack)
            player.stack -= call_amount
            player.bet += call_amount
            player.total_contributed += call_amount
            self.state.pot.total += call_amount
            self.state.pot.contributions[player_idx] = self.state.pot.contributions.get(player_idx, 0) + call_amount
            if player.stack == 0:
                player.is_all_in = True
            self.acted_since_last_raise.add(player_idx)
            if self.enable_history:
                self.action_history.append((player_idx, action.action_type, call_amount, self.state.betting_round))

        # Handle raises
        elif action.action_type == "raise":
            # Enforce minimum raise
            raise_amount = action.amount if action.amount else min_raise
            if raise_amount < min_raise and player.stack > to_call + min_raise:
                raise_amount = min_raise  # Force minimum raise
            
            total_needed = to_call + raise_amount
            actual_bet = min(total_needed, player.stack)
            actual_raise = actual_bet - to_call  # The actual raise portion
            
            player.stack -= actual_bet
            player.bet += actual_bet
            player.total_contributed += actual_bet
            self.state.pot.total += actual_bet
            self.state.pot.contributions[player_idx] = self.state.pot.contributions.get(player_idx, 0) + actual_bet
            self.current_bet = player.bet
            
            if player.stack == 0:
                player.is_all_in = True
            
            # Update last raise size for min-raise tracking
            if actual_raise >= min_raise:
                self.last_raise_size = actual_raise
                # Full raise reopens betting
                self.acted_since_last_raise.clear()
                self.bb_has_option = False  # A raise negates BB option
                self.state.last_aggressor = player_idx  # Track for showdown order
            # else: short all-in raise doesn't reopen betting
            
            self.acted_since_last_raise.add(player_idx)
            if self.enable_history:
                self.action_history.append((player_idx, action.action_type, player.bet, self.state.betting_round))

        # Handle all-in
        elif action.action_type == "all-in":
            all_in_amount = player.stack
            new_bet = player.bet + all_in_amount
            raise_portion = new_bet - current_max_bet
            
            player.bet = new_bet
            player.total_contributed += all_in_amount
            self.state.pot.total += all_in_amount
            self.state.pot.contributions[player_idx] = self.state.pot.contributions.get(player_idx, 0) + all_in_amount
            player.stack = 0
            player.is_all_in = True
            
            # Only reopen betting if this is a full raise (>= min raise)
            if raise_portion >= min_raise:
                self.current_bet = new_bet
                self.last_raise_size = raise_portion
                self.acted_since_last_raise.clear()
                self.bb_has_option = False
                self.state.last_aggressor = player_idx  # Track for showdown order
            elif new_bet > current_max_bet:
                # Short all-in: update current bet but DON'T reopen betting
                self.current_bet = new_bet
            
            self.acted_since_last_raise.add(player_idx)
            if self.enable_history:
                self.action_history.append((player_idx, action.action_type, new_bet, self.state.betting_round))

        # Advance the turn
        self._advance_turn()

        # Assertion to verify pot consistency
        pot_sum = sum(p.total_contributed for p in self.state.players)
        assert self.state.pot.total == pot_sum, f"Pot {self.state.pot.total} != sum bets {pot_sum}"

    # ---------- END CONDITIONS ----------

    def is_hand_over(self):
        not_folded = [p for p in self.players if not p.has_folded]
        return len(not_folded) <= 1 or self.state.betting_round == "showdown"

    def resolve_showdown(self):
        """
        Build side pots and resolve the showdown. Only non-folded players are eligible for pots.
        Updates player stacks with winnings and returns the winnings dict.
        """
        self.state.pot.create_side_pots(self.players)
        winnings = resolve_showdown(self.players, self.state.community_cards, self.state.pot, self.state.button)

        # Update player stacks with winnings
        for pid, amount in winnings.items():
            self.players[pid].stack += amount

        # Empty the pot now that it has been paid out.  It used to keep its
        # pre-payout total until the next reset_hand(), so `sum(stacks) + pot`
        # double-counted the money and any caller reading pot.total after a
        # showdown saw a stale figure.  Clearing here also makes a second call
        # a no-op rather than a double payout.
        self.state.pot.reset()

        return winnings

    def __repr__(self):
        return repr(self.state)
