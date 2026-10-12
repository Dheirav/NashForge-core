"""
Play the project's agents, in the game the project measures them in.

    venv/bin/python -m gui.main                     # against the CFR agent
    venv/bin/python -m gui.main --opponent random   # the panel's floor
    venv/bin/python -m gui.main --seat 1            # sit in the other seat

Keys: 1-6 pick an action, space deals the next hand, d shows the debug
overlay, q quits.

The opponent choices are the benchmark panel's, deliberately. Playing the same
three opponents the agents are scored against is what makes a session here
comparable to a number in the reports; an opponent invented for the viewer
would not be.
"""
import argparse
import sys

import pygame

from gui.game_controller import AgentUnavailable, GameController
from gui.renderer import Renderer
from gui.ui_components import get_action_buttons
from gui.layout import get_layout

WIDTH, HEIGHT = 1280, 800
FPS = 60


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", choices=["cfr", "random", "always-call"],
                        default="cfr",
                        help="which panel agent to play against (default: cfr)")
    parser.add_argument("--seat", type=int, choices=[0, 1], default=0,
                        help="which seat you take (default: 0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the deals, for a reproducible session")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Built before the window, so a missing strategy prints a line a person can
    # act on instead of opening a black window and dying behind it.
    try:
        controller = GameController(opponent=args.opponent, human_seat=args.seat,
                                    seed=args.seed)
    except AgentUnavailable as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 1

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption(f"NashForge — heads-up vs {args.opponent}")
    clock = pygame.time.Clock()
    renderer = Renderer(screen, controller)
    debug_overlay = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                renderer.screen = screen
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_d:
                    debug_overlay = not debug_overlay
                elif event.key == pygame.K_SPACE:
                    _deal_next(controller, renderer)
                elif pygame.K_1 <= event.key <= pygame.K_6:
                    controller.choose(event.key - pygame.K_1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if controller.hand_over:
                    _deal_next(controller, renderer)
                elif controller.awaiting_human:
                    layout = get_layout(*screen.get_size())
                    for button in get_action_buttons(controller, layout):
                        if button["enabled"] and button["rect"].collidepoint(event.pos):
                            controller.choose(button["index"])
                            break

        controller.update()
        if controller.hand_over and renderer.last_feedback is None:
            renderer.show_feedback(_result_text(controller))

        renderer.render(debug_overlay)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    _print_session(controller)
    return 0


def _deal_next(controller, renderer):
    if controller.hand_over:
        controller.next_hand()
        renderer.last_feedback = None


def _result_text(controller):
    chips = controller.last_result
    if chips is None:
        return ""
    if chips > 0:
        return f"you win {chips}"
    if chips < 0:
        return f"you lose {-chips}"
    return "split"


def _print_session(controller):
    """
    The session in the reports' units.

    Stated with the hand count because, in this project, a rate without one has
    repeatedly turned out to be noise -- and a session played by hand is far
    shorter than anything that could settle a question.
    """
    if controller.hands_played == 0:
        return
    rate = controller.bb_per_100
    print(f"{controller.hands_played} hands vs {controller.opponent_name}: "
          f"{controller.total_chips:+d} chips, {rate:+.1f} BB/100")
    print("Far too few hands to mean anything; the endpoint tests use 40,000.")


if __name__ == "__main__":
    raise SystemExit(main())
