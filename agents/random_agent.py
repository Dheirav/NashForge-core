import numpy as np
from engine.actions import Action

class RandomAgent:
	"""
	Research-grade random agent: selects uniformly random legal actions.
	No memory, no adaptation, pure baseline.
	"""
	def __init__(self, seed=None):
		self.rng = np.random.default_rng(seed)

	def select_action(self, game, player_idx):
		legal = game.get_legal_actions(player_idx)
		if not legal:
			return Action(Action.FOLD)
		action_dict = self.rng.choice(legal)
		return Action(action_dict['type'], action_dict.get('amount'))
