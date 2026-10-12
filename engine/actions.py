from typing import Optional

class Action:
	FOLD = 'fold'
	CHECK = 'check'
	CALL = 'call'
	RAISE = 'raise'
	ALL_IN = 'all-in'

	def __init__(self, action_type: str, amount: Optional[int] = None):
		assert action_type in {self.FOLD, self.CHECK, self.CALL, self.RAISE, self.ALL_IN}, f"Invalid action: {action_type}"
		self.action_type = action_type
		self.amount = amount

	def __repr__(self):
		if self.amount is not None:
			return f"Action({self.action_type}, {self.amount})"
		return f"Action({self.action_type})"

def validate_action(action: Action, min_call: int, min_raise: int, player_stack: int, current_bet: int, player_bet: int) -> bool:
	"""
	Validates if the action is legal given the current state.
	"""
	if action.action_type == Action.FOLD:
		return True
	if action.action_type == Action.CHECK:
		return current_bet == player_bet
	if action.action_type == Action.CALL:
		call_amount = min(current_bet - player_bet, player_stack)
		return call_amount > 0 and player_stack >= call_amount
	if action.action_type == Action.RAISE:
		if action.amount is None or action.amount < min_raise:
			return False
		total_bet = current_bet + action.amount
		return player_stack >= (total_bet - player_bet)
	if action.action_type == Action.ALL_IN:
		return player_stack > 0
	return False
