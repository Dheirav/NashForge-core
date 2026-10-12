"""
The arena's WebSocket protocol, driven from this side.

Adapted from the reference client in the chipzen-sdk repository
(`examples/external-api-bot/client.py`, Apache-2.0), which is the platform's
supported path for remote bots. The protocol frames are the reference's; what
is added is what a bot that has to survive a week of fixtures needs and the
reference deliberately leaves out:

- the lobby connection is kept open while a match is played, and reconnected
  with backoff when it drops, because a bot that is not in the lobby when a
  fixture is dispatched loses it on the clock
- a match socket that drops before `match_end` is re-dialled, which the
  executor supports for a 30-second grace window
- every decision is logged with the state it was taken from, so a hand can be
  replayed offline and compared with the arena's own replay
- a status file is rewritten on every event, for a progress reader

Protocol references: `docs/EXTERNAL-API-BOT-PROTOCOL.md` and
`docs/protocol/TRANSPORT-PROTOCOL.md` in that repository.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import websockets

logger = logging.getLogger("chipzen")

#: The platform's bot profile lags a finished match. Poll briefly for the
#: counter to move rather than recording a row that describes the match before
#: this one (23 September: a loss recorded as a win).
RATING_RETRIES = 6
RATING_RETRY_S = 5.0

PROTOCOL_VERSIONS = ["1.0"]
CLIENT_NAME = "nashforge"
CLIENT_VERSION = "0.1.0"
#: Sentinel subprotocol that marks the token in `Sec-WebSocket-Protocol`; the
#: gateway reads the token from the header so it never lands in a URL or a log.
BOT_TOKEN_SUBPROTOCOL = "chipzen-bot-token"
#: The executor's mid-match reconnect budget, per the transport spec.
MATCH_RECONNECTS = 3
#: Wall-clock budget for re-dialling a dropped match, inside the platform's grace window.
MATCH_REDIAL_BUDGET_S = 25.0
#: The server pings the lobby every 15 s. A socket that has gone this long
#: without any frame is dead whatever the OS says: on 14 September the machine
#: slept, the server closed us, and the client sat "connected" on a socket
#: that would never deliver another byte, missing every match for a day.
LOBBY_SILENCE = 60.0
MATCH_SILENCE = 120.0


@dataclass
class Status:
    """Everything the progress reader shows. Rewritten on every event."""
    started_at: float = field(default_factory=time.time)
    lobby: str = "starting"
    lobby_connects: int = 0
    last_event: str = ""
    last_event_at: float = 0.0
    matches_started: int = 0
    matches_finished: int = 0
    matches_active: int = 0
    wins: int = 0
    losses: int = 0
    hands: int = 0
    decisions: int = 0
    misses: int = 0
    fallbacks: int = 0
    rejected: int = 0
    slowest_ms: float = 0.0
    current: Dict[str, object] = field(default_factory=dict)
    recent: List[Dict[str, object]] = field(default_factory=list)


class Arena:
    """One bot, one lobby connection, any number of dispatched matches."""

    def __init__(self, base_url: str, bot_id: str, token: str, player,
                 log_dir: str, status_path: Optional[str] = None,
                 once: bool = False, matches_dir: Optional[str] = None,
                 version: Optional[dict] = None):
        self.base = _normalise_base(base_url)
        #: What the bot was when it played: commit, solver set, flags, label.
        #: Written into every match log so scripts/chipzen_ledger.py can put a
        #: win rate against each change.
        self.version = version or {}
        self.bot_id = bot_id
        self.token = token
        self.player = player
        self.log_dir = log_dir
        self.status_path = status_path or os.path.join(log_dir, "status.json")
        self.once = once
        self.status = Status()
        self._tasks: set = set()
        self._done = asyncio.Event()
        #: Per-match decision logs. Kept apart from the process log because
        #: they are the record of how the bot played, which outlives any run.
        self.matches_dir = matches_dir or os.path.join(log_dir, "matches")
        os.makedirs(self.matches_dir, exist_ok=True)

    # ---- status ------------------------------------------------------------

    def _event(self, text: str) -> None:
        self.status.last_event = text
        self.status.last_event_at = time.time()
        stats = self.player.stats
        self.status.decisions = stats.decisions
        self.status.misses = stats.misses
        self.status.fallbacks = stats.fallbacks
        self.status.slowest_ms = stats.slowest_ms
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(asdict(self.status), handle, indent=1)
        os.replace(tmp, self.status_path)

    # ---- lobby -------------------------------------------------------------

    async def run(self) -> None:
        """Hold the lobby, dispatching every `matched` into its own task."""
        backoff = 1.0
        while not self._done.is_set():
            try:
                await self._lobby_session()
                backoff = 1.0
            except (OSError, websockets.exceptions.WebSocketException,
                    ConnectionError, asyncio.TimeoutError) as error:
                self.status.lobby = f"reconnecting ({type(error).__name__}: {error})"
                self._event(f"lobby dropped: {error}")
                logger.warning("lobby: %s; retrying in %.0fs", error, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            if self.once and self.status.matches_finished:
                break
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _lobby_session(self) -> None:
        url = f"{self.base}/ws/external/bot/{self.bot_id}"
        logger.info("lobby: connecting to %s", url)
        self.status.lobby = "connecting"
        self._event("lobby connecting")
        async with websockets.connect(url, max_size=2 ** 24) as ws:
            self._lobby_ws = ws
            await ws.send(json.dumps({"type": "authenticate", "token": self.token}))
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=LOBBY_SILENCE)
                except asyncio.TimeoutError:
                    raise ConnectionError(f"no lobby frame for {LOBBY_SILENCE:.0f}s; assuming the socket is dead")
                except websockets.exceptions.ConnectionClosed as closed:
                    raise ConnectionError(f"lobby closed: {closed}")
                msg = _loads(raw)
                kind = msg.get("type")
                if kind == "hello":
                    self.status.lobby = "connected"
                    self.status.lobby_connects += 1
                    self._event("lobby connected")
                    logger.info("lobby: connected (endpoint=%s)", msg.get("endpoint"))
                elif kind == "ping":
                    await ws.send(json.dumps({"type": "pong"}))
                elif kind == "matched":
                    logger.info("lobby: matched match=%s rated=%s resume=%s",
                                msg.get("match_id"), msg.get("rated"), msg.get("resume"))
                    task = asyncio.create_task(self._play(msg))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                elif kind == "evict":
                    raise ConnectionError("evicted: replaced by a newer connection")
                else:
                    logger.debug("lobby: ignoring %s", kind)
                if self.once and self.status.matches_finished:
                    return

    async def drop_lobby(self, why: str) -> None:
        """Close the lobby socket so `run` reconnects; used when the platform says we are offline."""
        ws = getattr(self, "_lobby_ws", None)
        if ws is not None:
            logger.warning("lobby: dropping the connection (%s)", why)
            self._event(f"lobby dropped on purpose: {why}")
            try:
                await ws.close()
            except Exception:
                pass

    # ---- match -------------------------------------------------------------

    async def _play(self, matched: dict) -> None:
        match_id = matched["match_id"]
        gateway = _resolve(self.base, matched["gateway_ws_url"])
        log_path = os.path.join(self.matches_dir, f"{match_id}.jsonl")
        self.status.matches_started += 1
        self.status.matches_active += 1
        self.status.current = {"match_id": match_id, "rated": matched.get("rated"),
                               "hands": 0, "opponent": None, "seat": None}
        self._event("match starting")
        result = None
        attempts = 0
        # Re-dial until the match has an end or the platform's grace window
        # (about 30 s before a walkover) is spent, whichever first. A clean
        # server-side close before match_end used to end the loop without an
        # exception and abandon the match; three attempts a second apart used
        # up the budget in four seconds.
        deadline = time.monotonic() + MATCH_REDIAL_BUDGET_S
        with open(log_path, "a") as log:
            while True:
                try:
                    result = await self._match_session(gateway, match_id, log)
                    if result is not None:
                        break
                    reason = "closed before match_end"
                except (OSError, websockets.exceptions.WebSocketException,
                        asyncio.TimeoutError) as error:
                    reason = str(error)
                attempts += 1
                if time.monotonic() > deadline or attempts > MATCH_RECONNECTS * 4:
                    logger.warning("match %s: giving up after %d re-dials (%s)", match_id, attempts, reason)
                    break
                logger.warning("match %s: %s; re-dialling (%d)", match_id, reason, attempts)
                self._event(f"match dropped: {reason}")
                await asyncio.sleep(1.0)
        self.status.matches_active -= 1
        self.status.matches_finished += 1
        summary = {"match_id": match_id, "rated": matched.get("rated"),
                   "opponent": self.status.current.get("opponent"),
                   "hands": self.status.current.get("hands"),
                   "reason": (result or {}).get("reason"),
                   "outcome": _outcome(result, self.status.current.get("seat")),
                   "finished_at": time.time()}
        if summary["outcome"] == "won":
            self.status.wins += 1
        elif summary["outcome"] == "lost":
            self.status.losses += 1
        self.status.recent = ([summary] + self.status.recent)[:20]
        self.status.current = {}
        self._event(f"match {summary['outcome'] or 'ended'}")
        logger.info("match %s ended: %s", match_id, summary)
        if self.once:
            self._done.set()

    async def _match_session(self, gateway: str, match_id: str, log) -> Optional[dict]:
        async with websockets.connect(gateway, max_size=2 ** 24,
                                      subprotocols=[BOT_TOKEN_SUBPROTOCOL, self.token]) as ws:
            # The executor ignores the token on this leg (the gateway's own
            # credential is authoritative) but the frame must come first.
            await ws.send(json.dumps({"type": "authenticate", "match_id": match_id, "token": ""}))
            hello = _loads(await ws.recv())
            if hello.get("type") != "hello":
                raise ConnectionError(f"expected hello, got {hello.get('type')!r}")
            await ws.send(json.dumps({
                "type": "hello", "match_id": match_id,
                "supported_versions": PROTOCOL_VERSIONS,
                "client_name": CLIENT_NAME, "client_version": CLIENT_VERSION}))
            self._event("match handshake done")

            seat: Optional[int] = None
            stacks_before = None
            rejections_this_turn = 0
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=MATCH_SILENCE)
                except asyncio.TimeoutError:
                    raise ConnectionError(f"no match frame for {MATCH_SILENCE:.0f}s; re-dialling")
                except websockets.exceptions.ConnectionClosedOK:
                    break
                msg = _loads(raw)
                kind = msg.get("type")
                if kind == "ping":
                    await ws.send(json.dumps({"type": "pong", "match_id": match_id}))
                    continue
                if kind == "match_start":
                    for entry in msg.get("seats") or []:
                        if entry.get("is_self"):
                            seat = int(entry["seat"])
                        else:
                            self.status.current["opponent"] = entry.get("display_name")
                    self.status.current["seat"] = seat
                    self.player.opponent = self.status.current.get("opponent")
                    self.status.current["config"] = msg.get("game_config")
                    self.status.current["timeout_ms"] = msg.get("turn_timeout_ms")
                    log.write(json.dumps({"frame": "match_start", "seat": seat,
                                          "match_id": match_id, "at": time.time(),
                                          "version": self.version,
                                          "rated": self.status.current.get("rated"),
                                          "seats": msg.get("seats"),
                                          "game_config": msg.get("game_config"),
                                          "turn_timeout_ms": msg.get("turn_timeout_ms")}) + "\n")
                    self._event("match started")
                elif kind == "round_start":
                    stacks_before = (msg.get("state") or {}).get("stacks")
                    self.status.current["hands"] = (msg.get("state") or {}).get("hand_number")
                    log.write(json.dumps({"frame": "round_start", "state": msg.get("state")}) + "\n")
                elif kind == "turn_request":
                    rejections_this_turn = 0
                    acting = int(msg.get("seat", seat if seat is not None else 0))
                    if seat is None:
                        # A resumed match re-sends no match_start; the turn
                        # request names the seat that must act, which is ours.
                        seat = acting
                        self.status.current["seat"] = seat
                    state = msg.get("state") or {}
                    valid = msg.get("valid_actions") or []
                    started = time.perf_counter()
                    try:
                        decision = self.player.decide(state, valid, acting)
                    except Exception as error:      # never let a bug become a timeout
                        logger.exception("decide failed; safe action instead")
                        decision = {"action": "check" if "check" in valid else "fold",
                                    "params": {}, "record": {"error": repr(error)}}
                    await ws.send(json.dumps({
                        "type": "turn_action", "match_id": match_id,
                        "request_id": msg.get("request_id"),
                        "action": decision["action"], "params": decision["params"]}))
                    record = dict(decision.get("record") or {})
                    record["frame"] = "decision"
                    record["timeout_ms"] = msg.get("timeout_ms")
                    record["round_trip_ms"] = round((time.perf_counter() - started) * 1000, 2)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    self._event(f"hand {state.get('hand_number')} {state.get('phase')}: "
                                f"{decision['action']} {decision['params'] or ''}")
                elif kind == "action_rejected":
                    self.status.rejected += 1
                    rejections_this_turn += 1
                    fallback = _rejection_fallback(msg.get("valid_actions"), rejections_this_turn)
                    logger.warning("match %s: action rejected (%s); sending %s",
                                   match_id, msg.get("reason"), fallback)
                    log.write(json.dumps({"frame": "rejected", "reason": msg.get("reason"),
                                          "message": msg.get("message"),
                                          "fallback": fallback}) + "\n")
                    await ws.send(json.dumps({
                        "type": "turn_action", "match_id": match_id,
                        "request_id": msg.get("request_id"),
                        "action": fallback, "params": {}}))
                elif kind == "round_result":
                    result = msg.get("result") or {}
                    self.status.hands += 1
                    profiles = getattr(self.player, "profiles", None)
                    if profiles is not None and seat is not None and self.player.opponent:
                        after = result.get("stacks")
                        net = after[seat] - stacks_before[seat] if stacks_before and after else 0
                        profiles.observe(result, seat, self.player.opponent, net)
                        profiles.save()
                    log.write(json.dumps({"frame": "round_result", "result": result}) + "\n")
                    log.flush()
                elif kind == "match_end":
                    log.write(json.dumps({"frame": "match_end", "reason": msg.get("reason"),
                                          "results": msg.get("results")}) + "\n")
                    log.flush()
                    self._record_rating(match_id)
                    return msg
                elif kind == "error":
                    logger.warning("match %s: error [%s] %s", match_id,
                                   msg.get("code"), msg.get("message"))
                    log.write(json.dumps({"frame": "error", "code": msg.get("code"),
                                          "message": msg.get("message")}) + "\n")
                else:
                    logger.debug("match: observed %s", kind)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

    def _record_rating(self, match_id: str) -> None:
        """
        Our platform rating after a match, appended beside the match logs.

        The platform keeps no rating history the API will show, so the trace
        of what each version did to the rating is built here, one line per
        match, stamped with the version label. Best effort: a failed request
        must not touch the match.
        """
        try:
            # The profile lags: the platform has not counted the match that just
            # ended when it answers. On 23 September the row stamped with the
            # wsp match we LOST reported a win, because the win it had caught up
            # on was the previous one, and two consecutive rows in the file
            # carry identical counters wherever that has happened. Every
            # per-version rating attribution in the history is therefore shifted
            # by one match. Wait for the counter to move, briefly, and record
            # whether it did, so a row can never silently describe the match
            # before it.
            stats = _http(self.base, self.token, "GET", f"/api/bots/{self.bot_id}")
            before = stats.get("matches_played")
            for _ in range(RATING_RETRIES):
                if stats.get("matches_played") != before or before is None:
                    break
                time.sleep(RATING_RETRY_S)
                stats = _http(self.base, self.token, "GET", f"/api/bots/{self.bot_id}")
            counted = stats.get("matches_played") != before
            row = {"at": time.time(), "match_id": match_id, "counted": counted,
                   "label": self.version.get("label"),
                   "rating": stats.get("rating"), "rating_deviation": stats.get("rating_deviation"),
                   "matches_played": stats.get("matches_played"), "wins": stats.get("wins"),
                   "losses": stats.get("losses"), "bb_per_100": stats.get("bb_per_100")}
            with open(os.path.join(self.matches_dir, "..", "rating_history.jsonl"), "a") as handle:
                handle.write(json.dumps(row) + "\n")
        except Exception as error:               # noqa: BLE001
            logger.debug("rating record skipped: %s", error)


def _rejection_fallback(valid, rejections: int) -> str:
    """
    What to send when the platform rejects an action.

    Check if it is free; otherwise fold before call, because a rejected raise
    at three big blinds turning into a call of the whole outstanding bet is the
    stack, and a fold is at most the blind. A second rejection on the same
    turn goes straight to the cheapest legal action rather than looping until
    the clock folds us anyway.
    """
    valid = list(valid or ["check", "fold"])
    if "check" in valid:
        return "check"
    if rejections >= 2:
        return "fold" if "fold" in valid else (valid[0] if valid else "fold")
    if "fold" in valid:
        return "fold"
    return "call" if "call" in valid else (valid[0] if valid else "fold")


def _normalise_base(base_url: str) -> str:
    parts = urlsplit(base_url)
    scheme = parts.scheme or "wss"
    netloc = parts.netloc or parts.path
    return urlunsplit((scheme, netloc, "", "", "")).rstrip("/")


def _resolve(base: str, gateway_path: str) -> str:
    if gateway_path.startswith(("ws://", "wss://")):
        return gateway_path
    return f"{base}{gateway_path}"


def _loads(raw) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return msg if isinstance(msg, dict) else {}


def _outcome(match_end: Optional[dict], seat: Optional[int]) -> Optional[str]:
    """
    won / lost / None for our seat.

    The arena's `results` is a list of per-seat rows carrying `final_stack` and
    `net_chips` (measured 13 September: no rank field). Elimination means the
    winner holds every chip, so a positive net is a win and a negative one a
    loss; a rank field is honoured if one ever appears.
    """
    if not match_end or seat is None:
        return None
    results = match_end.get("results")
    rows = results.get("rankings") if isinstance(results, dict) else results
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not isinstance(row, dict) or row.get("seat") != seat:
            continue
        if "rank" in row:
            return "won" if int(row["rank"]) == 1 else "lost"
        net = row.get("net_chips")
        if net is not None:
            return "won" if float(net) > 0 else ("lost" if float(net) < 0 else None)
        stack = row.get("final_stack")
        if stack is not None:
            return "won" if float(stack) > 0 else "lost"
    return None


# ---------------------------------------------------------------------------
# Token-authenticated HTTP: the three things a token can start by itself
# ---------------------------------------------------------------------------
#
# Protocol sections 7.4 to 7.6. None of these returns a match: every match is
# seated through the lobby `matched` push, so the bot must be in the lobby
# first and the responses here only say whether the platform agreed.


def _http(base_url: str, token: str, method: str, path: str, body=None) -> dict:
    import requests   # only these helpers need HTTP

    http = _normalise_base(base_url).replace("wss://", "https://").replace("ws://", "http://")
    response = requests.request(method, f"{http}{path}",
                                headers={"Authorization": f"Bearer {token}",
                                         "Content-Type": "application/json"},
                                json=body if method == "POST" else None, timeout=30)
    try:
        payload = response.json()
    except ValueError:
        payload = {"text": response.text}
    if not isinstance(payload, dict):
        payload = {"body": payload}
    payload["http_status"] = response.status_code
    return payload


def house_bot_challenge(base_url: str, token: str, opponent: Optional[str] = None) -> dict:
    """An unrated match against a house bot, on the relaxed 30-second clock."""
    body = {} if opponent is None else {"opponent": opponent}
    return _http(base_url, token, "POST", "/api/external-api/challenges/house-bot", body)


def lobby_opponents(base_url: str, token: str) -> dict:
    """Remote bots of other owners that are in the lobby now, with ratings."""
    return _http(base_url, token, "GET", "/api/external-api/challenges/remote/opponents")


def challenge_remote(base_url: str, token: str, opponent: str) -> dict:
    """A rated challenge to one of them. They must accept within about 60s."""
    return _http(base_url, token, "POST", "/api/external-api/challenges/remote",
                 {"opponent": opponent})


def remote_challenges(base_url: str, token: str) -> dict:
    """Our inbound and outbound challenges and their states."""
    return _http(base_url, token, "GET", "/api/external-api/challenges/remote")


def accept_remote_challenge(base_url: str, token: str, challenge_id: str) -> dict:
    return _http(base_url, token, "POST",
                 f"/api/external-api/challenges/remote/{challenge_id}/accept", {})


def decline_remote_challenge(base_url: str, token: str, challenge_id: str) -> dict:
    return _http(base_url, token, "POST",
                 f"/api/external-api/challenges/remote/{challenge_id}/decline", {})


def upcoming_fixtures(base_url: str, token: str) -> dict:
    """
    Season fixtures scheduled for this bot, with their decision clocks.

    Per the platform's author (13 September): fixtures run one round per
    evening from 18:00 UTC, ten minutes apart, the clock is written on each
    fixture as `decision_clock_seconds` (30 for remote bots), and the process
    must be in the lobby when a fixture opens: it waits about 90 seconds and
    re-kicks twice, then the match is a walkover.
    """
    return _http(base_url, token, "GET", "/api/external-api/fixtures/upcoming")


def join_queue(base_url: str, token: str) -> dict:
    """The rated first-in-first-out queue; pairs with any waiting remote bot."""
    return _http(base_url, token, "POST", "/api/external-api/matchmaking/join", {})


def queue_status(base_url: str, token: str) -> dict:
    return _http(base_url, token, "GET", "/api/external-api/matchmaking/status")


def leave_queue(base_url: str, token: str) -> dict:
    return _http(base_url, token, "POST", "/api/external-api/matchmaking/leave", {})
