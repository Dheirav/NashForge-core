"""
The six abstract actions, as buttons.

The labels are the abstraction's, not poker's general vocabulary: this agent
chooses among six actions and nothing else, and a button offering anything
further would be describing a game it cannot play.

Positions come from `layout`, which the renderer also draws from, so the
rectangle a click is tested against is the rectangle that was drawn. They were
two different sets of coordinates before, one hardcoded here and one computed
there, which is the sort of thing that works until the window is resized.
"""
import pygame

from gui.game_controller import ACTION_LABELS

BUTTON_COLORS = [
    (200, 60, 60),      # fold
    (60, 120, 220),     # check/call
    (220, 180, 60),     # raise 1/2
    (220, 180, 60),     # raise pot
    (220, 180, 60),     # raise 2x
    (120, 60, 180),     # all-in
]
DISABLED = (70, 74, 72)


def get_action_buttons(controller, layout):
    """One dict per abstract action: where it is, what it says, whether it is legal."""
    legal = controller.legal_actions()
    buttons = []
    for index in range(len(ACTION_LABELS)):
        x, y, w, h = layout["buttons"][index]
        buttons.append({
            "rect": pygame.Rect(x, y, w, h),
            "label": controller.action_label(index),
            "color": BUTTON_COLORS[index] if legal[index] else DISABLED,
            "enabled": legal[index],
            "index": index,
        })
    return buttons
