"""
The Slumbot HTTP protocol, and nothing else.

This is the first step of `docs/EXTERNAL_BENCHMARK.md`: every strength figure in
this project was computed by this project about itself, and item 1 closed with no
usable bound, so no-limit has no exploitability figure at all. Slumbot is a fixed
CFR strategy at heads-up no-limit behind a free public API, and results against it
are reported in mbb/hand in published work.

Kept deliberately free of any agent. The brief is specific that action translation
is most of the engineering and that a translation bug looks exactly like a weak
strategy, so the layer that talks to the server and the layer that decides what to
say are separate files with separate tests.

The protocol, as measured rather than assumed
----------------------------------------------
``POST /api/new_hand {}`` opens a hand and returns a token. ``POST /api/act
{token, incr}`` sends one action. Both return the same shape; a terminal response
adds ``winnings`` and the bot's cards.

Actions are a compact string: ``b200`` bets *to* 200, ``c`` calls, ``k`` checks,
``f`` folds, and ``/`` separates streets. So ``b200c/kb200`` reads: raise to 200,
call, new street, check, bet to 200.

The stakes, which the translation layer has to bridge
-----------------------------------------------------
Slumbot plays **20,000 chips at 50/100 — 200 big blinds deep**. Measured: folding
as the big blind to an opening raise loses exactly 100. This project's engine and
solver use 200 chips at 1/2, which is **100 big blinds**.

That is not a units conversion, it is a different game. A strategy fitted for 100bb
is off-tree at 200bb from the first decision, and no amount of careful chip
arithmetic fixes it. Whatever plays here has to be built for the depth Slumbot
plays, or the depth has to be reported as a caveat on every number that follows.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

BASE = "https://slumbot.com/api"

#: Slumbot's stakes, confirmed against the live server rather than taken from a
#: reference implementation.
BIG_BLIND = 100
SMALL_BLIND = 50
STARTING_STACK = 20_000

#: Seconds between requests. The API is free and unauthenticated; a measurement
#: run is tens of thousands of hands, and there is no reason for it to look like
#: an attack.
COURTESY_DELAY = 0.05

ACTION = re.compile(r"b(\d+)|([ckf])")


class SlumbotError(RuntimeError):
    """A protocol failure, carrying what was sent so it can be reproduced."""


@dataclass
class HandState:
    """One response, parsed. Terminal states carry ``winnings``."""
    token: str
    action: str
    old_action: str
    client_pos: int
    hole_cards: List[str]
    board: List[str]
    winnings: Optional[int] = None
    bot_hole_cards: List[str] = field(default_factory=list)
    baseline_winnings: Optional[float] = None
    raw: Dict = field(default_factory=dict)

    @property
    def over(self) -> bool:
        return self.winnings is not None

    @property
    def street(self) -> int:
        """0 preflop, 1 flop, 2 turn, 3 river, counted from the separators."""
        return self.action.count("/")

    @property
    def facing_bet(self) -> bool:
        """
        Whether the last action on this street was a bet or raise.

        Determines whether ``c`` or ``k`` is the legal passive reply, and getting
        it wrong is rejected by the server rather than silently reinterpreted --
        which is the one mercy in this protocol.
        """
        if not self.action and self.client_pos == 1:
            # The blinds are not in the string: the small blind opens the hand
            # owing half a blind, so "k" there is an illegal check. The 16
            # September calibration lost all 500 of its button hands to that.
            return True
        current = self.action.split("/")[-1]
        return bool(current) and current[-1].isdigit()

    def bet_levels(self) -> List[int]:
        """
        The cumulative bet levels on the current street, in order.

        Deliberately not a `to_call`. Bets are cumulative *to* an amount, so the
        increment owed is a difference of levels -- but preflop the blinds are
        already committed and are not in this string, so the big blind facing
        `b200` owes 100 rather than 200. Getting that wrong overpays on every
        preflop call, and chip accounting would still balance because the engine
        takes whatever it is told; only the result would move.

        Position and posted blinds are the translation layer's job, and it can
        compute what is owed from these levels once it has them. A half-correct
        helper here would be used by the layer that does not yet exist, which is
        how the wrong number becomes load-bearing.
        """
        street = self.action.split("/")[-1]
        return [int(m.group(1)) for m in ACTION.finditer(street) if m.group(1)]


def _post(path: str, payload: Dict, retries: int = 3) -> Dict:
    # Imported here rather than at the top: `slumbot.bridge` reads this module's constants, the arena
    # bot shares that bridge, and the upload sandbox refuses urllib.
    import urllib.error
    import urllib.request
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE}/{path}", data=body,
        headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last = error
            # A transient network failure mid-run should cost a retry, not the
            # session: the alternative is a partial hand count, which is worse
            # than a slow one.
            time.sleep(0.5 * (attempt + 1))
    raise SlumbotError(f"{path} failed after {retries} attempts: {last}")


def _state(token: str, payload: Dict) -> HandState:
    if "error_msg" in payload:
        raise SlumbotError(payload["error_msg"])
    return HandState(
        token=token,
        action=payload.get("action", ""),
        old_action=payload.get("old_action", ""),
        client_pos=payload.get("client_pos", 0),
        hole_cards=payload.get("hole_cards", []),
        board=payload.get("board", []),
        winnings=payload.get("winnings"),
        bot_hole_cards=payload.get("bot_hole_cards", []),
        baseline_winnings=payload.get("baseline_winnings"),
        raw=payload,
    )


def new_hand(token: Optional[str] = None) -> HandState:
    """Open a hand. Passing the previous token continues the same session."""
    payload = _post("new_hand", {"token": token} if token else {})
    return _state(payload.get("token", token or ""), payload)


def act(state: HandState, incr: str) -> HandState:
    """Send one action. ``incr`` is Slumbot's own notation: b<amount>, c, k, f."""
    time.sleep(COURTESY_DELAY)
    payload = _post("act", {"token": state.token, "incr": incr})
    return _state(state.token, payload)


Policy = Callable[[HandState], str]


def play_hand(policy: Policy, token: Optional[str] = None,
              guard: int = 60) -> HandState:
    """
    One hand from deal to settlement.

    ``policy`` receives the state and returns Slumbot's notation for its choice.
    The guard is here because a policy that returns an action the server rejects
    can otherwise loop against it forever, and an unbounded loop against someone
    else's free API is the rudest possible bug.
    """
    state = new_hand(token)
    for _ in range(guard):
        if state.over:
            return state
        state = act(state, policy(state))
    raise SlumbotError(f"hand did not terminate: {state.action!r}")


def always_fold(state: HandState) -> str:
    """
    Folds every decision. Heads-up that loses 750 mbb/hand against an opponent
    who never folds a blind, and about 700 against Slumbot, which folds its
    small blind roughly a fifth of the time: measured 16 September 2026 over
    1,000 hands at −699 ± 21, button hands −500 exactly and big-blind hands
    −900 on average. So the raw column is trustworthy to its interval, and the
    baseline-differenced column, −646 ± 1,193 on the same hands, is not.
    """
    return "f" if state.facing_bet else "k"


def call_station(state: HandState) -> str:
    """
    Check or call, whichever is legal. The floor, and useful for a protocol test.

    It is not an agent and its results are not a benchmark of anything; it exists
    so the transport can be exercised without the translation layer existing yet.
    """
    return "c" if state.facing_bet else "k"
