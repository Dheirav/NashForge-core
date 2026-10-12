import argparse
from engine.game import PokerGame
from engine.actions import Action

def get_legal_actions(player, game):
    actions = []
    to_call = game.current_bet - player.bet
    min_raise = game.state.big_blind
    if player.has_folded or player.is_all_in or player.stack == 0:
        return []
    if to_call == 0:
        actions.append('check')
    else:
        if player.stack > to_call:
            actions.append('call')
        elif player.stack == to_call:
            actions.append('all-in')
        else:
            actions.append('fold')
    if player.stack > to_call + min_raise:
        actions.append('raise')
    actions.append('fold')
    return actions

def main():
    parser = argparse.ArgumentParser(description="Play a CLI Texas Hold'em hand.")
    parser.add_argument('--players', type=int, default=2, help='Number of players (2-9)')
    parser.add_argument('--stack', type=int, default=100, help='Starting stack for each player')
    parser.add_argument('--sb', type=int, default=5, help='Small blind')
    parser.add_argument('--bb', type=int, default=10, help='Big blind')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: random)')
    parser.add_argument('--all-ai', action='store_true', help='Run all players as AI (no user input)')
    parser.add_argument('--logfile', type=str, default=None, help='Write detailed hand log to this file')
    args = parser.parse_args()

    import random
    seed = args.seed if args.seed is not None else random.randint(1, 1_000_000_000)
    print(f"[INFO] Using seed: {seed}")
    game = PokerGame([args.stack] * args.players, small_blind=args.sb, big_blind=args.bb, seed=seed)
    print("Starting a new hand!")
    print(game)

    log_lines = []
    while not game.is_hand_over():
        # Display current state
        round_str = f"\n--- {game.state.betting_round.upper()} ---"
        comm_str = f"Community: {game.state.community_cards}"
        players_str = []
        for p in game.players:
            pv = game.state.public_view(p.player_id)['players'][p.player_id]
            players_str.append(f"Player {p.player_id} ({'You' if p.player_id == 0 else 'AI'}), Stack: {pv['stack']}, Bet: {pv['bet']}, Folded: {p.has_folded}, All-in: {p.is_all_in}")
        pot_str = f"Pot: {game.state.pot.total}"
        log_lines.append(round_str)
        log_lines.append(comm_str)
        log_lines.extend(players_str)
        log_lines.append(pot_str)

        # Get current player from engine
        player = game.players[game.state.current_player]
        state_view = game.state.public_view(player.player_id)
        player_view = state_view['players'][player.player_id]
        turn_str = f"Player {player.player_id} ({'You' if player.player_id == 0 else 'AI'})'s turn."
        cards_str = f"Cards: {player_view['hole_cards']}"
        legal_str = f"Legal actions: {get_legal_actions(player, game)}"
        log_lines.append(turn_str)
        log_lines.append(cards_str)
        log_lines.append(legal_str)
        if args.all_ai or player.player_id != 0:
            import random
            legal = get_legal_actions(player, game)
            if not legal:
                # No legal actions, skip turn (should not happen, but safe guard)
                log_lines.append("AI: No legal actions available, skipping turn.")
                continue
            # Weighted action selection for more variety
            weights = []
            for act in legal:
                if act == 'fold':
                    weights.append(0.15)
                elif act == 'check':
                    weights.append(0.25)
                elif act == 'call':
                    weights.append(0.25)
                elif act == 'raise':
                    weights.append(0.25)
                elif act == 'all-in':
                    weights.append(0.10)
                else:
                    weights.append(0.01)
            # Normalize
            total = sum(weights)
            if total == 0:
                log_lines.append("AI: No valid weights for actions, skipping turn.")
                continue
            weights = [w/total for w in weights]
            ai_action = random.choices(legal, weights=weights, k=1)[0]
            if ai_action == 'fold':
                action = Action(Action.FOLD)
            elif ai_action == 'check':
                action = Action(Action.CHECK)
            elif ai_action == 'call':
                action = Action(Action.CALL)
            elif ai_action == 'raise':
                min_raise = game.state.big_blind
                to_call = game.current_bet - player.bet
                min_valid = game.current_bet + min_raise
                max_valid = player.bet + player.stack
                # Randomize raise target between min and max
                if max_valid > min_valid:
                    raise_to = random.randint(min_valid, max_valid)
                else:
                    raise_to = min_valid
                if raise_to > game.current_bet:
                    action = Action(Action.RAISE, raise_to)
                else:
                    action = Action(Action.ALL_IN)
            elif ai_action == 'all-in':
                action = Action(Action.ALL_IN)
            log_lines.append(f"AI chooses: {action}")
        else:
            action_str = input("Enter action: ").strip().lower()
            if action_str not in get_legal_actions(player, game):
                log_lines.append("Invalid or illegal action.")
                print("Invalid or illegal action.")
                continue
            if action_str == 'fold':
                action = Action(Action.FOLD)
            elif action_str == 'check':
                action = Action(Action.CHECK)
            elif action_str == 'call':
                action = Action(Action.CALL)
            elif action_str == 'raise':
                try:
                    amt = int(input("Enter raise amount: "))
                    action = Action(Action.RAISE, amt)
                except Exception:
                    log_lines.append("Invalid raise amount.")
                    print("Invalid raise amount.")
                    continue
            elif action_str == 'all-in':
                action = Action(Action.ALL_IN)
            log_lines.append(f"User chooses: {action}")
        game.apply_action(player.player_id, action)

    log_lines.append("\nHand over!")
    log_lines.append(str(game))
    from engine.hand_eval import evaluate_hand, HAND_RANKS
    log_lines.append("Showdown results:")
    # Track active (not folded) players
    active_players = [p for p in game.players if not p.has_folded]
    if len(active_players) == 1:
        winner = active_players[0]
        log_lines.append(f"Player {winner.player_id} wins the pot of {game.state.pot.total} by default.")
    elif len(active_players) > 1:
        player_best = {}
        enough_community = len(game.state.community_cards) >= 3
        for p in active_players:
            total_cards = len(p.hole_cards) + len(game.state.community_cards)
            if total_cards >= 5 and enough_community:
                full_hand = p.hole_cards + game.state.community_cards
                best = evaluate_hand(full_hand)
                if best is not None:
                    player_best[p.player_id] = best
                    log_lines.append(f"Player {p.player_id} cards: {p.hole_cards} | Best hand: {best} ({HAND_RANKS[best.hand_rank]})")
                else:
                    log_lines.append(f"Player {p.player_id} cards: {p.hole_cards} | No valid hand.")
            else:
                log_lines.append(f"Player {p.player_id} cards: {p.hole_cards} | Not enough cards for a hand.")
        results = game.resolve_showdown()
        max_win = max(results.values()) if results else 0
        for pid, win in results.items():
            if win == max_win and pid in player_best:
                best = player_best[pid]
                log_lines.append(f"Player {pid} WINS {win} with {HAND_RANKS[best.hand_rank]}: {best.hand}")
            else:
                log_lines.append(f"Player {pid} wins {win}")

    # Write log to file if requested
    if args.logfile:
        with open(args.logfile, 'w') as f:
            for line in log_lines:
                f.write(line + '\n')
    # Also print to stdout for normal runs
    else:
        for line in log_lines:
            print(line)

if __name__ == "__main__":
    main()
