"""
Hand history logging in PokerStars-compatible format.
Generates human-readable and parseable hand histories for analysis and training.
"""
from typing import List, Optional, TextIO
from datetime import datetime
import os

from .state import GameState, PlayerState
from .cards import Card
from .actions import Action


class HandHistoryLogger:
    """
    Logs poker hands in PokerStars-style format.
    Can write to file or return as string.
    """
    
    def __init__(self, log_dir: str = "logs/hands", table_name: str = "Table1"):
        self.log_dir = log_dir
        self.table_name = table_name
        self.hand_number = 0
        self.current_hand: List[str] = []
        self._file: Optional[TextIO] = None
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
    
    def _format_cards(self, cards: List[Card]) -> str:
        """Format cards as PokerStars style: [Ah Kd]"""
        if not cards:
            return ""
        return "[" + " ".join(f"{c.rank}{c.suit[0]}" for c in cards) + "]"
    
    def _format_hole_cards(self, cards: List[Card]) -> str:
        """Format hole cards without brackets"""
        if not cards:
            return ""
        return " ".join(f"{c.rank}{c.suit[0]}" for c in cards)
    
    def start_hand(self, game, hand_id: Optional[int] = None):
        """Begin logging a new hand."""
        self.hand_number = hand_id if hand_id else self.hand_number + 1
        self.current_hand = []
        
        state = game.state
        players = game.players
        
        timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        
        # Header
        self.current_hand.append(
            f"PokerStars Hand #{self.hand_number}: Hold'em No Limit "
            f"(${state.small_blind}/${state.big_blind}) - {timestamp}"
        )
        self.current_hand.append(f"Table '{self.table_name}' {len(players)}-max Seat #{state.button + 1} is the button")
        
        # Seats
        for i, p in enumerate(players):
            in_chips = p.stack + p.total_contributed
            self.current_hand.append(f"Seat {i + 1}: Player{i + 1} (${in_chips} in chips)")
        
        # Blinds
        num_players = len(players)
        if num_players == 2:
            sb_idx = state.button
            bb_idx = (state.button + 1) % 2
        else:
            sb_idx = (state.button + 1) % num_players
            bb_idx = (state.button + 2) % num_players
        
        self.current_hand.append(f"Player{sb_idx + 1}: posts small blind ${state.small_blind}")
        self.current_hand.append(f"Player{bb_idx + 1}: posts big blind ${state.big_blind}")
        
        # Hole cards (mark as dealt)
        self.current_hand.append("*** HOLE CARDS ***")
    
    def log_hole_cards(self, player_id: int, cards: List[Card], hero: bool = False):
        """Log hole cards dealt to a player. If hero=True, show cards."""
        if hero and cards:
            self.current_hand.append(f"Dealt to Player{player_id + 1} {self._format_cards(cards)}")
    
    def log_action(self, player_id: int, action: Action, amount: int = 0):
        """Log a player action."""
        action_str = f"Player{player_id + 1}: "
        
        action_type = action.action_type if hasattr(action, 'action_type') else str(action)
        
        if action_type == 'fold':
            action_str += "folds"
        elif action_type == 'check':
            action_str += "checks"
        elif action_type == 'call':
            action_str += f"calls ${amount}"
        elif action_type == 'bet':
            action_str += f"bets ${amount}"
        elif action_type == 'raise':
            action_str += f"raises to ${amount}"
        elif action_type == 'all-in':
            action_str += f"is all-in ${amount}"
        else:
            action_str += str(action_type)
        
        self.current_hand.append(action_str)
    
    def log_street(self, street: str, cards: List[Card]):
        """Log a new street (flop, turn, river)."""
        street_names = {
            'flop': 'FLOP',
            'turn': 'TURN', 
            'river': 'RIVER'
        }
        name = street_names.get(street, street.upper())
        self.current_hand.append(f"*** {name} *** {self._format_cards(cards)}")
    
    def log_showdown(self, player_id: int, cards: List[Card], hand_name: str):
        """Log a player showing their hand at showdown."""
        self.current_hand.append(f"Player{player_id + 1}: shows {self._format_cards(cards)} ({hand_name})")
    
    def log_pot_won(self, player_id: int, amount: int, pot_type: str = "pot"):
        """Log a player winning a pot."""
        self.current_hand.append(f"Player{player_id + 1} collected ${amount} from {pot_type}")
    
    def log_uncalled_bet(self, player_id: int, amount: int):
        """Log uncalled bet returned to player."""
        self.current_hand.append(f"Uncalled bet (${amount}) returned to Player{player_id + 1}")
    
    def log_summary(self, game, winners: List[int], amounts: List[int]):
        """Log the hand summary."""
        self.current_hand.append("*** SUMMARY ***")
        
        total_pot = sum(amounts)
        self.current_hand.append(f"Total pot ${total_pot}")
        
        # Board
        if game.state.community_cards:
            self.current_hand.append(f"Board {self._format_cards(game.state.community_cards)}")
        
        # Each player's result
        for i, p in enumerate(game.players):
            seat_info = f"Seat {i + 1}: Player{i + 1}"
            
            if p.has_folded:
                seat_info += " folded"
                if game.state.betting_round == 'preflop':
                    seat_info += " before Flop"
                else:
                    seat_info += f" on the {game.state.betting_round.capitalize()}"
            elif i in winners:
                idx = winners.index(i)
                seat_info += f" showed {self._format_cards(p.hole_cards)} and won (${amounts[idx]})"
            else:
                seat_info += f" showed {self._format_cards(p.hole_cards)} and lost"
            
            self.current_hand.append(seat_info)
    
    def end_hand(self) -> str:
        """Finalize and return the hand history."""
        self.current_hand.append("")  # Blank line between hands
        return "\n".join(self.current_hand)
    
    def save_hand(self, filename: Optional[str] = None):
        """Save the current hand to a file."""
        if filename is None:
            filename = f"hand_{self.hand_number}.txt"
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'a') as f:
            f.write(self.end_hand())
            f.write("\n")
    
    def get_hand_history(self) -> str:
        """Return the current hand history as a string."""
        return "\n".join(self.current_hand)


class SessionLogger:
    """
    Manages logging for an entire session of hands.
    Writes all hands to a single session file.
    """
    
    def __init__(self, log_dir: str = "logs/sessions", session_name: Optional[str] = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        
        self.session_file = os.path.join(log_dir, f"{session_name}.txt")
        self.hand_count = 0
        self.logger = HandHistoryLogger(log_dir)
    
    def log_hand(self, game, actions_log: List[tuple], winners: List[int], 
                 amounts: List[int], hero_id: Optional[int] = None) -> str:
        """
        Log a complete hand.
        
        Args:
            game: The PokerGame instance
            actions_log: List of (player_id, Action, amount) tuples
            winners: List of winning player IDs
            amounts: List of amounts won
            hero_id: Player ID whose hole cards to show
        
        Returns:
            The hand history string
        """
        self.hand_count += 1
        self.logger.start_hand(game, self.hand_count)
        
        # Log hero's hole cards
        if hero_id is not None and hero_id < len(game.players):
            self.logger.log_hole_cards(hero_id, game.players[hero_id].hole_cards, hero=True)
        
        # Log actions by street
        current_street = 'preflop'
        for player_id, action, amount in actions_log:
            # Check for street changes
            if hasattr(action, 'street') and action.street != current_street:
                current_street = action.street
                if current_street == 'flop':
                    self.logger.log_street('flop', game.state.community_cards[:3])
                elif current_street == 'turn':
                    self.logger.log_street('turn', game.state.community_cards[:4])
                elif current_street == 'river':
                    self.logger.log_street('river', game.state.community_cards)
            
            self.logger.log_action(player_id, action, amount)
        
        # Log showdown
        if game.state.betting_round == 'showdown':
            self.logger.current_hand.append("*** SHOW DOWN ***")
            for i in winners:
                if i < len(game.players):
                    p = game.players[i]
                    hand_name = "a hand"  # Could evaluate actual hand name
                    self.logger.log_showdown(i, p.hole_cards, hand_name)
        
        # Log summary
        self.logger.log_summary(game, winners, amounts)
        
        # Save to session file
        history = self.logger.end_hand()
        with open(self.session_file, 'a') as f:
            f.write(history)
            f.write("\n")
        
        return history


def format_action_for_log(action: Action, amount: int = 0) -> str:
    """Format an action for display."""
    action_type = action.action_type if hasattr(action, 'action_type') else str(action)
    
    if action_type == 'fold':
        return "folds"
    elif action_type == 'check':
        return "checks"
    elif action_type == 'call':
        return f"calls ${amount}"
    elif action_type == 'bet':
        return f"bets ${amount}"
    elif action_type == 'raise':
        return f"raises to ${amount}"
    elif action_type == 'all-in':
        return f"all-in ${amount}"
    return str(action_type)
