from typing import List, Optional
from .cards import Card

class PlayerState:
	def __init__(self, player_id: int, stack: int, seat: int):
		self.player_id = player_id
		self.stack = stack
		self.seat = seat
		self.hole_cards: List[Card] = []
		self.bet = 0
		self.has_folded = False
		self.is_all_in = False
		self.total_contributed = 0  # Total chips put into pot this hand (not reset between rounds)

	def post_blind(self, amount: int):
		posted = min(self.stack, amount)
		self.stack -= posted
		self.bet += posted
		self.total_contributed += posted
		if self.stack == 0:
			self.is_all_in = True
		return posted

	def reset_for_new_hand(self):
		self.hole_cards = []
		self.bet = 0
		self.has_folded = False
		self.is_all_in = False
		self.total_contributed = 0

	def __repr__(self):
		return f"PlayerState(id={self.player_id}, stack={self.stack}, bet={self.bet}, folded={self.has_folded}, all_in={self.is_all_in}, total_contributed={self.total_contributed})"


# --- GameState ---
from .pot import Pot

class GameState:
	def advance_stage(self):
		# Only advance the betting round label
		if self.betting_round == 'preflop':
			self.betting_round = 'flop'
		elif self.betting_round == 'flop':
			self.betting_round = 'turn'
		elif self.betting_round == 'turn':
			self.betting_round = 'river'
		elif self.betting_round == 'river':
			self.betting_round = 'showdown'
	def __init__(self, players: List[PlayerState], button: int, small_blind: int, big_blind: int, seed: int = None):
		self.players = players  # List[PlayerState]
		self.button = button  # Dealer position
		self.small_blind = small_blind
		self.big_blind = big_blind
		self.community_cards: List[Card] = []
		self.pot = Pot()
		self.betting_round = 'preflop'  # preflop, flop, turn, river, showdown
		self.current_player = None  # Index of player to act
		self.betting_history: List[dict] = []
		self.deck_seed = seed
		self.last_aggressor = None

	def serialize(self):
		# Returns a dict representation for debugging/testing
		return {
			# Full player info (includes hole cards). Use public_view() for agent-visible state.
			'players': [
				{
					'player_id': p.player_id,
					'stack': p.stack,
					'bet': p.bet,
					'folded': p.has_folded,
					'all_in': p.is_all_in,
					'hole_cards': [str(c) for c in p.hole_cards],
					'seat': p.seat,
				}
				for p in self.players
			],
			'button': self.button,
			'small_blind': self.small_blind,
			'big_blind': self.big_blind,
			'community_cards': [str(c) for c in self.community_cards],
			'pot': repr(self.pot),
			'betting_round': self.betting_round,
			'current_player': self.current_player,
			'betting_history': self.betting_history,
			'deck_seed': self.deck_seed,
			'last_aggressor': self.last_aggressor,
		}

	def __repr__(self):
		return f"GameState(round={self.betting_round}, pot={self.pot}, players={self.players}, community={self.community_cards})"

	def public_view(self, for_player_id: int):
		"""
		Return a JSON-serializable view of the game state appropriate for agents.
		Only the requesting player's hole cards are included; other players' hole cards are masked.
		"""
		players_view = []
		for p in self.players:
			if p.player_id == for_player_id:
				hole = [str(c) for c in p.hole_cards]
			else:
				# Mask other players' private cards
				hole = [None for _ in p.hole_cards]
			players_view.append({
				'player_id': p.player_id,
				'stack': p.stack,
				'bet': p.bet,
				'folded': p.has_folded,
				'all_in': p.is_all_in,
				'hole_cards': hole,
				'seat': p.seat,
			})
		return {
			'players': players_view,
			'button': self.button,
			'small_blind': self.small_blind,
			'big_blind': self.big_blind,
			'community_cards': [str(c) for c in self.community_cards],
			'pot': repr(self.pot),
			'betting_round': self.betting_round,
			'current_player': self.current_player,
			'betting_history': self.betting_history,
			'deck_seed': self.deck_seed,
		}

	def reveal_all(self):
		"""
		Return full state (including all hole cards). Use only for trusted debugging, UI, or self-play.
		"""
		return self.serialize()
