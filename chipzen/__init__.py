"""
NashForge playing on Chipzen, the bot arena at chipzen.ai.

`bridge` turns the arena's structured hand state into the solver's history key
and back; `player` puts the solver behind that; `client` speaks the arena's
WebSocket protocol. Nothing in here changes how the solver is measured on the
panel or against Slumbot: the lookup is `evaluation.benchmark.cfr_agent`, the
same one both of those use, so a rating earned here is earned by the agent that
was measured and not by a second copy of it.
"""
