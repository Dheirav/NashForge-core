import numpy as np
from engine.actions import Action
from engine.features import get_preflop_strength_fast

class HeuristicAgent:
	"""
	Research-grade heuristic agent: uses hand strength, pot odds, position, and basic poker theory.
	- Aggressive with strong hands
	- Folds weak hands to large bets
	- Calls with drawing hands if pot odds are favorable
	- Plays tighter in early position
	"""
	def select_action(self, game, player_idx):
		legal = game.get_legal_actions(player_idx)
		if not legal:
			return Action(Action.FOLD)

		player = game.state.players[player_idx]
		street = game.state.betting_round
		position = player.seat
		stack = player.stack
		pot = game.state.pot.total
		to_call = max(game.current_bet - player.bet, 0)

		# Preflop hand strength
		hand_strength = get_preflop_strength_fast(player.hole_cards)

		# Position: early (0,1), middle (2,3), late (4,5)
		if position in [0, 1]:
			position_factor = 0.8  # tighter
		elif position in [2, 3]:
			position_factor = 1.0
		else:
			position_factor = 1.2  # looser

		# Aggressive with strong hands
		if hand_strength * position_factor > 0.7:
			for a in legal:
				if a['type'] in ['raise', 'all-in']:
					return Action(a['type'], a.get('amount'))
		# Fold weak hands to big bets
		if hand_strength * position_factor < 0.4 and to_call > 0:
			for a in legal:
				if a['type'] == 'fold':
					return Action('fold')
		# Call with drawing hands if pot odds are good
		pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
		if hand_strength * position_factor > 0.5 and pot_odds < 0.2:
			for a in legal:
				if a['type'] == 'call':
					return Action('call', a.get('amount'))
		# Default: check if possible, else call
		for a in legal:
			if a['type'] == 'check':
				return Action('check')
		for a in legal:
			if a['type'] == 'call':
				return Action('call', a.get('amount'))
		return Action('fold')
