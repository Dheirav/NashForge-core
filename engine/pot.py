from typing import List, Dict

class Pot:
    def __init__(self):
        self.total = 0
        self.side_pots: List[Dict] = []  # Each side pot: {'amount': int, 'eligible': set(player_ids)}
        self.contributions: Dict[int, int] = {}  # player_id -> amount contributed

    def add_contribution(self, player_id: int, amount: int):
        """
        Add chips to the pot for a player. Tracks total and per-player contributions.
        """
        if amount <= 0:
            return
        self.total += amount
        self.contributions[player_id] = self.contributions.get(player_id, 0) + amount

    def create_side_pots(self, players: List):
        """
        Build side pots for all-in situations. Each side pot is a dict with 'amount' and 'eligible' (set of player_ids).
        Uses player.total_contributed for accuracy. Folded players' contributions go into the pot but they cannot win.
        This must be called before showdown.
        """
        # Get all contributions (including folded players) for pot calculation
        all_contribs = [(p.player_id, p.total_contributed, p.has_folded) for p in players if p.total_contributed > 0]
        
        # Get unique contribution levels from ALL players (including folded)
        levels = sorted(set(c[1] for c in all_contribs))
        
        pots = []
        prev_level = 0
        
        for level in levels:
            # Count how many players contributed at least this level
            contributors_at_level = [c for c in all_contribs if c[1] >= level]
            num_contributors = len(contributors_at_level)
            
            # Pot amount is (level - prev_level) * number of contributors
            pot_amt = (level - prev_level) * num_contributors
            
            if pot_amt > 0:
                # Only non-folded players who contributed at least this level are eligible
                eligible = set(c[0] for c in contributors_at_level if not c[2])
                pots.append({'amount': pot_amt, 'eligible': eligible.copy()})
            
            prev_level = level
        
        self.side_pots = pots

    def reset(self):
        """Reset the pot for a new hand."""
        self.total = 0
        self.side_pots = []
        self.contributions = {}

    def __repr__(self):
        return f"Pot(total={self.total}, side_pots={self.side_pots}, contributions={self.contributions})"
