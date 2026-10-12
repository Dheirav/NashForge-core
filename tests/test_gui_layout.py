"""
The viewer's rectangles, checked for the failure they actually had.

At 1280x800 the side panels were drawn on top of the first and last community
cards, because the board was sized from the whole window while the panels were
placed independently of it. That is invisible to every test that asks whether
a function returns a dict, so these ask the only question that mattered: does
anything overlap anything it must not, at any window size a person might use.
"""
import pytest

from gui.layout import get_layout

#: Sizes worth holding: the default, the old default, a laptop, something wide
#: and something nearly square.
SIZES = [(1280, 800), (1000, 700), (1366, 768), (1920, 1080), (900, 850)]


def _rect(x, y, w, h):
    return (x, y, x + w, y + h)


def _overlaps(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _card_rects(layout, key):
    card_w, card_h = layout["card_size"]
    return [_rect(x, y, card_w, card_h) for x, y in layout[key]]


@pytest.mark.parametrize("size", SIZES)
def test_panels_do_not_cover_the_board(size):
    layout = get_layout(*size)
    panels = {name: _rect(*layout[name])
              for name in ("policy_panel", "stats_panel")}

    for group in ("community", "player", "agent"):
        for index, card in enumerate(_card_rects(layout, group)):
            for name, panel in panels.items():
                assert not _overlaps(card, panel), (
                    f"{name} covers {group} card {index} at {size}")


@pytest.mark.parametrize("size", SIZES)
def test_buttons_do_not_cover_the_hole_cards(size):
    """A card underneath a button cannot be read and can still be clicked through."""
    layout = get_layout(*size)
    buttons = [_rect(*spec) for spec in layout["buttons"]]
    for index, card in enumerate(_card_rects(layout, "player")):
        for button_index, button in enumerate(buttons):
            assert not _overlaps(card, button), (
                f"button {button_index} covers player card {index} at {size}")


@pytest.mark.parametrize("size", SIZES)
def test_buttons_are_disjoint_and_on_screen(size):
    """
    Six buttons, none overlapping, all inside the window.

    Overlapping buttons are worse than ugly here: the click handler returns the
    first rectangle that contains the point, so an overlap silently makes one
    action unreachable.
    """
    width, height = size
    buttons = [_rect(*spec) for spec in get_layout(width, height)["buttons"]]
    assert len(buttons) == 6

    for i, first in enumerate(buttons):
        x0, y0, x1, y1 = first
        assert x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height, (
            f"button {i} falls outside {size}")
        for second in buttons[i + 1:]:
            assert not _overlaps(first, second), f"buttons overlap at {size}"


@pytest.mark.parametrize("size", SIZES)
def test_board_has_five_slots_in_a_row(size):
    layout = get_layout(*size)
    community = layout["community"]
    assert len(community) == 5
    assert len({y for _, y in community}) == 1, "board is not level"

    card_w, _ = layout["card_size"]
    assert card_w > 0
    gaps = [b - a - card_w for (a, _), (b, _) in zip(community, community[1:])]
    assert all(gap >= 0 for gap in gaps), f"community cards overlap at {size}"
