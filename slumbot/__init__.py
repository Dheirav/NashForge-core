"""Talking to Slumbot: the transport, kept separate from anything that decides."""
# The names are resolved on first use: the transport imports urllib, and `slumbot.bridge`, which the
# arena bot shares, must import without it (the uploaded bot's sandbox refuses urllib).
_API = ("BIG_BLIND", "SMALL_BLIND", "STARTING_STACK", "HandState", "SlumbotError", "act",
        "call_station", "new_hand", "play_hand")

__all__ = ["HandState", "SlumbotError", "new_hand", "act", "play_hand",
           "call_station", "BIG_BLIND", "SMALL_BLIND", "STARTING_STACK"]


def __getattr__(name):
    if name in _API:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module 'slumbot' has no attribute {name!r}")
