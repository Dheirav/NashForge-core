"""
Where everything sits, derived from the window rather than assumed.

The board used to be sized from the full window width while the two side
panels were placed independently, so at 1280x800 the panels were drawn on top
of the first and last community cards. Sizes here are derived from the space
that is actually left over instead: the panels claim their columns first, and
the board is fitted to the gap between them, so the overlap cannot come back
at some other window size.
"""

#: Table proportions, as fractions of the window.
PANEL_FRACTION = 0.20      # width of each side panel
MARGIN_FRACTION = 0.03
GUTTER_FRACTION = 0.02     # clearance between a panel and the board
CARD_ASPECT = 1.375        # height / width, standard playing card


def get_layout(WIDTH, HEIGHT):
    """Every rectangle the renderer and the click handler both need."""
    layout = {}
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    margin = int(WIDTH * MARGIN_FRACTION)
    gutter = int(WIDTH * GUTTER_FRACTION)

    # The panels take their columns first.
    panel_w = int(WIDTH * PANEL_FRACTION)
    panel_h = int(HEIGHT * 0.40)
    panel_y = center_y - panel_h // 2
    layout['policy_panel'] = (margin, panel_y, panel_w, panel_h)
    layout['stats_panel'] = (WIDTH - panel_w - margin, panel_y, panel_w, panel_h)

    # Whatever is left in the middle is the board, five cards wide.
    board_left = margin + panel_w + gutter
    board_right = WIDTH - margin - panel_w - gutter
    board_w = max(board_right - board_left, 5)
    card_spacing = max(6, int(board_w * 0.022))
    card_w = (board_w - 4 * card_spacing) // 5
    card_h = int(card_w * CARD_ASPECT)
    layout['card_size'] = (card_w, card_h)

    comm_y = center_y - card_h // 2
    layout['community'] = [(board_left + i * (card_w + card_spacing), comm_y)
                           for i in range(5)]

    # Hole cards sit in the same column as the board, above and below it.
    hole_total_w = 2 * card_w + card_spacing
    hole_x0 = center_x - hole_total_w // 2
    agent_y = margin
    layout['agent'] = [(hole_x0 + i * (card_w + card_spacing), agent_y)
                       for i in range(2)]

    # The action bar is placed first, and the player's cards are put above it,
    # because a card drawn underneath a button is the one thing here that must
    # not happen.
    btn_h = int(HEIGHT * 0.085)
    btn_y = HEIGHT - margin - btn_h
    btn_spacing = max(8, int(WIDTH * 0.012))
    btn_w = (WIDTH - 2 * margin - 5 * btn_spacing) // 6
    btn_x0 = center_x - (6 * btn_w + 5 * btn_spacing) // 2
    layout['buttons'] = [(btn_x0 + i * (btn_w + btn_spacing), btn_y, btn_w, btn_h)
                         for i in range(6)]

    player_y = btn_y - gutter - card_h
    layout['player'] = [(hole_x0 + i * (card_w + card_spacing), player_y)
                        for i in range(2)]

    layout['player_glow'] = (center_x, player_y + card_h // 2, int(card_w * 1.1))
    layout['agent_glow'] = (center_x, agent_y + card_h // 2, int(card_w * 1.1))
    layout['feedback'] = (center_x, comm_y - int(card_h * 0.42))
    return layout
