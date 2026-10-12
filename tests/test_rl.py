"""
The reinforcement-learning stack, which had no tests at all.

`rl/` is 1,917 lines and was carried through the August audit untested, on the
grounds that its *implementation* looked sound. That is not the same as knowing
it. The audit's own finding was that a correct-looking training loop had been
optimising a metric that scored the wrong player for 89 runs, and nothing in
the code said so — only a measurement did.

So these are the invariants a training run silently violates rather than
crashes on:

* the observation the environment hands the network is the one the engine
  defines, and stays that size,
* chips are conserved through an episode,
* an illegal action requested by the network never reaches the engine,
* an episode ends,
* and one PPO update runs without producing NaN.

The last matters most. A NaN in the policy does not raise; it makes every
subsequent action uniform, and the run continues to completion looking merely
disappointing.

    python -m pytest tests/test_rl.py -q
"""
import random

import numpy as np
import pytest
import torch

from engine import get_abstract_action_mask, get_feature_names
from rl import PokerEnv, PPOAgent, PPOConfig, PPOTrainer

TOTAL_CHIPS = 2 * 200


@pytest.fixture
def env():
    return PokerEnv(num_players=2, starting_stack=200, small_blind=1,
                    big_blind=2, seed=7)


@pytest.fixture
def config(tmp_path):
    """A trainer small enough to run one real update inside a test."""
    return PPOConfig(num_players=2, starting_stack=200, small_blind=1,
                     big_blind=2, n_steps=24, batch_size=8, n_epochs=1,
                     hidden_size=16, num_layers=1, total_hands=24,
                     eval_every=10 ** 9, save_every=10 ** 9,
                     device="cpu", seed=0, checkpoint_dir=str(tmp_path),
                     log_dir=str(tmp_path), verbose=False)


# ---------------------------------------------------------------------------
# The observation contract
# ---------------------------------------------------------------------------

def test_the_environment_and_the_engine_agree_on_the_observation_size():
    """
    `PokerEnv.OBS_SIZE` and `PPOConfig.obs_size` are both hardcoded, and the
    engine defines the layout separately. If they drift, the network is fed a
    truncated or padded vector and trains happily on it — which is the exact
    shape of the bug the audit found between `FeatureCache` and
    `get_state_vector`.

    This will fail the moment the feature layer is rebuilt, which is the point:
    it names every place that has to move together.
    """
    engine_features = len(get_feature_names())
    assert PokerEnv.OBS_SIZE == engine_features, (
        f"env expects {PokerEnv.OBS_SIZE} features, engine defines "
        f"{engine_features}")
    assert PPOConfig().obs_size == engine_features, (
        f"PPO config expects {PPOConfig().obs_size}, engine defines "
        f"{engine_features}")


def test_reset_and_step_return_the_declared_shape(env):
    obs, info = env.reset()
    assert obs.shape == (PokerEnv.OBS_SIZE,), obs.shape
    assert obs.dtype == np.float32, obs.dtype
    assert "game_seed" in info

    obs, _reward, _terminated, truncated, _info = env.step(1)
    assert obs.shape == (PokerEnv.OBS_SIZE,), obs.shape
    assert obs.dtype == np.float32, obs.dtype
    assert truncated is False, "the env declares no time-limit truncation"


def test_observations_stay_inside_the_declared_box(env):
    """
    The env advertises a Box(0, 1) observation space. A feature outside it is
    not fatal, but it silently breaks any normalisation assumption downstream —
    and the space is what a reader trusts.
    """
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    seen = [obs]

    for _ in range(200):
        obs, _r, terminated, _t, _i = env.step(int(rng.integers(6)))
        seen.append(obs)
        if terminated:
            obs, _ = env.reset()
            seen.append(obs)

    stacked = np.array(seen)
    assert np.isfinite(stacked).all(), "non-finite feature reached the network"
    # BoxSpace stores the bounds as arrays broadcast to the observation shape.
    floor = float(np.min(env.observation_space.low))
    ceiling = float(np.max(env.observation_space.high))
    offenders = [
        (name, float(stacked[:, i].min()), float(stacked[:, i].max()))
        for i, name in enumerate(get_feature_names())
        if stacked[:, i].min() < floor - 1e-6 or stacked[:, i].max() > ceiling + 1e-6
    ]
    assert not offenders, (
        f"features outside the declared Box({floor}, {ceiling}): {offenders}")


# ---------------------------------------------------------------------------
# What the environment must never do
# ---------------------------------------------------------------------------

def test_an_illegal_action_never_reaches_the_engine(env):
    """
    The network emits an index over all six abstract actions regardless of
    which are legal. The env masks and substitutes. If that substitution were
    wrong the engine would raise — but the audit found the previous training
    loop catching such failures and converting them to folds, so an agent
    emitting illegal actions every hand scored as merely weak.
    """
    rng = np.random.default_rng(1)
    env.reset()

    for _ in range(400):
        # Deliberately unfiltered: ask for whatever, legal or not.
        _obs, _r, terminated, _t, _i = env.step(int(rng.integers(6)))
        if terminated:
            env.reset()


def test_chips_are_conserved_across_an_episode(env):
    """
    Two hundred chips each in, four hundred out, every hand.

    Counting `sum(stacks) + pot.total` alone is what let the unpaid pot hide:
    it balances perfectly while the chips sit in the middle unawarded. So the
    pot must also be *empty* by the end, which is the same statement about a
    finished hand and the one that fails when the pot is destroyed instead of
    paid — as it was on every hand that reached a double fold.
    """
    rng = np.random.default_rng(2)

    for _ in range(20):
        env.reset()
        terminated = False
        guard = 0
        while not terminated and guard < 200:
            _obs, _r, terminated, _t, _i = env.step(int(rng.integers(6)))
            guard += 1
        assert guard < 200, "episode did not terminate"

        game = env._game
        total = sum(p.stack for p in game.players) + game.state.pot.total
        assert total == TOTAL_CHIPS, f"chips became {total}"
        assert game.state.pot.total == 0, "hand ended with the pot unpaid"
        assert sum(p.stack for p in game.players) == TOTAL_CHIPS, (
            "chips were destroyed rather than awarded")


def _play_random_hands(env, hands, seed):
    """
    Play `hands` complete hands of uniform-random legal play. Rewards out.

    The global `random` module is seeded because `RandomOpponent` draws from
    it, so without this the opponent's play depends on whatever earlier tests
    left in the module state and these assertions pass or fail by test order.
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)
    rewards, seats = [], []
    for _ in range(hands):
        env.reset()
        seats.append(env._agent_id)
        reward, terminated, guard = 0.0, False, 0
        while not terminated and guard < 200:
            mask = np.asarray(get_abstract_action_mask(env._game, env._agent_id))
            action = int(rng.choice(np.flatnonzero(mask)))
            _obs, reward, terminated, _t, _i = env.step(action)
            guard += 1
        assert terminated, "episode did not terminate"
        rewards.append(reward)
    return np.array(rewards), np.array(seats)


def test_the_reward_is_consistent_with_zero_in_a_symmetric_matchup(env):
    """
    The check that was missing, and the one that mattered.

    Random against random is symmetric, so its expectation is zero and the
    only question is whether the measurement says so. It did not: the pot was
    never awarded, because `_is_done()` means the betting is finished rather
    than the chips paid, so every hand scored as a loss of whatever had been
    contributed — 300 negative rewards out of 300, mean -53.7 BB.

    Conservation could not catch this. `sum(stacks) + pot.total` balances
    perfectly while the pot sits unpaid; the test above passed throughout. It
    takes an *expectation* against its own error bar to see it, which is the
    habit the rest of this project already runs on.

    Three sigma rather than two, for the reason `preflight_training.py` gives:
    a 95% band on a single sample fails 5% of the time by construction, and a
    test that cries wolf one run in twenty gets ignored on the run that
    matters. The defect it guards against was 17 sigma out.
    """
    rewards, _ = _play_random_hands(env, hands=1500, seed=11)

    mean = float(rewards.mean())
    sigma = float(rewards.std(ddof=1)) / np.sqrt(len(rewards))
    assert abs(mean) < 3.0 * sigma, (
        f"random vs random paid {mean:+.2f} +/- {sigma:.2f} BB per hand "
        f"({mean / sigma:+.1f} sigma); a symmetric matchup must be consistent "
        f"with zero")

    won = (rewards > 0).mean()
    assert won > 0.25, (
        f"only {won:.1%} of hands paid anything positive — the pot is not "
        f"being awarded")


def test_the_agent_is_dealt_both_seats(env):
    """
    `PokerGame` always opens on button 0 and this env deals a fresh game per
    hand, so an agent pinned to seat 0 is dealt the button every hand, learns
    one seat, and is then measured in both — `benchmark()` alternates seats by
    construction. Training and evaluation must see the same positions.
    """
    _rewards, seats = _play_random_hands(env, hands=400, seed=12)

    share = float((seats == 0).mean())
    assert 0.4 < share < 0.6, (
        f"agent sat in seat 0 for {share:.0%} of hands; the seat is not "
        f"alternating")


def test_reset_returns_a_hand_the_agent_can_actually_act_in(env):
    """
    Heads-up the agent is sometimes the big blind and the opponent folds
    preflop, so the hand is over before the agent ever acts — 13% of hands
    against a random opponent. `reset()` used to return that terminal state
    anyway; the caller would step, the engine would accept a fold from a
    player whose hand was already finished, both players would end up folded
    and the pot would be destroyed rather than awarded.
    """
    random.seed(13)

    for _ in range(300):
        env.reset()
        assert not env._is_done(), "reset() returned a finished hand"
        assert env._game.state.current_player == env._agent_id, (
            f"reset() returned a hand where seat "
            f"{env._game.state.current_player} is to act, not the agent "
            f"({env._agent_id})")

    assert env.hands_without_decision > 0, (
        "no hand ended before the agent acted in 300 attempts, so this test "
        "is no longer exercising the case it was written for")


def test_the_trainer_deals_a_different_hand_every_time(config):
    """
    The audit's original finding was that the deck re-dealt the same two hands
    every hand. The trainer had reproduced it exactly: it built a fresh
    `PokerEnv` per hand from a fixed `config.seed`, so every env re-seeded to
    the same stream and drew the same deal — **two distinct hands across
    twelve**, measured. A run like that trains to convergence, reports a
    falling loss, and has seen one hand.

    Seeded runs must still be reproducible, so this asks for variety within a
    run rather than for the seed to be ignored.
    """
    trainer = PPOTrainer(config)
    seeds = [trainer._next_hand()[1]["game_seed"] for _ in range(120)]

    distinct = len(set(seeds))
    assert distinct > 100, (
        f"{distinct} distinct deals in 120 hands — the environment is being "
        f"re-seeded per hand and the agent is seeing the same cards")


def test_episodes_terminate_and_report_a_consistent_delta(env):
    """
    `chip_delta` and `bb_per_100` describe the same outcome and must agree,
    since downstream reporting picks whichever is convenient.
    """
    rng = np.random.default_rng(3)
    env.reset()

    for _ in range(400):
        _obs, _r, terminated, _t, info = env.step(int(rng.integers(6)))
        if terminated:
            assert "chip_delta" in info and "bb_per_100" in info
            expected = (info["chip_delta"] / env.big_blind) * 100
            assert info["bb_per_100"] == pytest.approx(expected)
            env.reset()


# ---------------------------------------------------------------------------
# The agent and one real update
# ---------------------------------------------------------------------------

def test_the_agent_only_ever_names_a_legal_action(env, config):
    """
    `act` takes the legality mask, so the agent is responsible for respecting
    it — the env's substitution is a backstop, not the mechanism. Asserting the
    action is merely *in range* would pass on an agent that names an illegal
    action every hand and leans on the backstop, which is how the audit found
    the previous loop scoring broken agents as weak ones.
    """
    agent = PPOAgent(config)
    obs, _ = env.reset()

    for _ in range(80):
        mask = np.asarray(get_abstract_action_mask(env._game, env._agent_id))
        action = agent.act(obs, mask)
        index = int(action[0] if isinstance(action, tuple) else action)

        assert 0 <= index < PokerEnv.NUM_ACTIONS, index
        assert mask[index], (
            f"agent chose action {index}, which the mask {mask.tolist()} "
            f"marks illegal")

        obs, _r, terminated, _t, _i = env.step(index)
        if terminated:
            obs, _ = env.reset()


def test_one_ppo_update_produces_no_nan(config):
    """
    The failure this exists for is silent. A NaN in the policy does not raise —
    it makes the action distribution uniform and the run continues to
    completion, looking merely disappointing rather than broken.
    """
    import torch

    trainer = PPOTrainer(config)
    agent = trainer.train()

    parameters = list(agent.net.parameters())
    assert parameters, "agent exposes no parameters"
    for tensor in parameters:
        assert torch.isfinite(tensor).all(), "non-finite parameter after one update"


def test_a_saved_agent_reloads_to_the_same_policy(config, tmp_path):
    """
    Checkpoints are how a run's result survives the run. An agent that reloads
    to a different policy makes every downstream evaluation describe something
    that was never trained.
    """
    agent = PPOAgent(config)
    path = str(tmp_path / "agent.pt")
    agent.save(path)

    revived = PPOAgent.from_checkpoint(path, device="cpu")
    probe = np.zeros(PokerEnv.OBS_SIZE, dtype=np.float32)

    original = agent.get_value(probe)
    reloaded = revived.get_value(probe)
    assert reloaded == pytest.approx(original, abs=1e-6), (original, reloaded)


# ---------------------------------------------------------------------------
# Self-play through the snapshot pool
# ---------------------------------------------------------------------------

def _policy_on(agent, obs):
    """The agent's action distribution on one observation, as an array."""
    import torch

    mask = torch.ones(1, PokerEnv.NUM_ACTIONS)
    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
    with torch.no_grad():
        logits, _value = agent.net.forward(obs_t)
    logits = logits.masked_fill(mask == 0, -1e9)
    return torch.softmax(logits, dim=-1).numpy().ravel()


def test_a_snapshot_does_not_follow_the_policy_that_made_it(config):
    """
    The pool must hold *copies*. Storing a reference to the live agent gives a
    pool whose every member tracks the current policy — `len(pool)` grows, the
    log looks healthy, and the agent is playing itself in the present against
    itself in the present, which is not self-play in the sense the audit means
    and provides none of the opponent variety the pool exists for.

    Nothing about that is visible from outside, which is why it is asserted.
    """
    trainer = PPOTrainer(config)
    probe = np.linspace(0.0, 1.0, PokerEnv.OBS_SIZE).astype(np.float32)

    trainer.snapshots.add(trainer.agent, update_cycle=0)
    frozen = trainer.snapshots.sample()
    before = _policy_on(frozen, probe)

    trainer.train()          # the live policy moves

    after = _policy_on(frozen, probe)
    moved = _policy_on(trainer.agent, probe)

    np.testing.assert_allclose(after, before, atol=1e-9,
                               err_msg="the snapshot changed when the agent did")
    assert np.abs(moved - before).max() > 1e-9, (
        "the live policy did not move at all, so this test proves nothing "
        "about the snapshot")


def test_the_pool_keeps_the_most_recent_snapshots(config):
    """Capacity is a sliding window over the recent past, not a hard stop."""
    from rl import SnapshotPool

    agent = PPOAgent(config)
    pool = SnapshotPool(capacity=3, rng=np.random.default_rng(0))
    for cycle in range(1, 8):
        pool.add(agent, update_cycle=cycle)

    assert len(pool) == 3, f"pool holds {len(pool)}, capacity 3"
    assert pool.taken_at == [5, 6, 7], pool.taken_at
    assert pool.total_taken == 7


def test_the_trainer_snapshots_on_schedule_and_meets_the_pool(config):
    """
    A pool that never fills, or one the agent never actually faces, looks from
    the loss curve exactly like one that works. Both are checked here and both
    are logged during a run for the same reason.
    """
    config.total_hands = 3_000
    config.n_steps = 256
    config.snapshot_every = 2
    config.snapshot_pool_size = 4

    trainer = PPOTrainer(config)
    trainer.train()

    assert trainer.update_cycle >= 6, "too few updates to test the cadence"
    assert trainer.snapshots.total_taken == trainer.update_cycle // 2, (
        f"{trainer.snapshots.total_taken} snapshots over "
        f"{trainer.update_cycle} updates at a cadence of 2")
    assert len(trainer.snapshots) == 4, "the pool did not fill to capacity"

    faced = trainer._faced_current + trainer._faced_snapshot
    assert trainer._faced_snapshot > 0, "the agent never faced its own past"
    # Facing the live policy is capped by config, plus the warm-up hands
    # before the first snapshot exists, when the live policy is the only past
    # there is. So this is an upper bound rather than a match.
    assert trainer._faced_current / faced < 0.5, (
        f"faced the live policy on {trainer._faced_current / faced:.0%} of "
        f"hands, configured for {config.current_policy_prob:.0%}")


def test_the_critic_does_not_swamp_the_policy_through_the_shared_trunk(config):
    """
    Actor and critic share a trunk, so their gradients compete for it.

    Rewards in big blinds put the returns on a 100 BB stack's scale: standard
    deviation near 60, critic MSE in the thousands, and a trunk gradient of
    **55.2 against the policy's 0.0078** — a factor of 7,000. `max_grad_norm`
    then divides the sum by 160 and the policy stops moving. The run completes,
    the loss falls, and the result is a plateau — which is exactly what
    `docs/training-plan.md` predicts for PPO in advance. A predicted finding
    produced by an artefact is the worst outcome available here.

    So the returns are kept O(1). This asserts the property rather than the
    gradient ratio because the property is what holds throughout the run, and
    it is cheap enough to check every time.
    """
    trainer = PPOTrainer(config)
    trainer._collect_rollout()
    spread = float(trainer.buffer.returns[: trainer.buffer.size].std())

    assert spread < 5.0, (
        f"returns have a standard deviation of {spread:.1f}; the critic's "
        f"gradient will swamp the policy on the shared trunk")

    # And the same rollout without the scaling, to show the check has teeth.
    trainer.env.reward_scale = 1.0
    trainer._collect_rollout()
    unscaled = float(trainer.buffer.returns[: trainer.buffer.size].std())
    assert unscaled > 5.0, (
        f"unscaled returns spread only {unscaled:.1f}, so this test is no "
        f"longer measuring the thing it was written for")


def test_an_interrupted_run_resumes_where_it_stopped(config, tmp_path):
    """
    A rung of the ladder is four hours on a box that has been terminated twice
    under memory pressure, so resume is not a convenience.

    Restoring the policy alone is the failure worth guarding: it looks like a
    resume, the weights are right, and Adam restarts from zero moments against
    an empty opponent pool. The run continues as a *different* training process
    wearing the same weights, and nothing says so.
    """
    config.total_hands = 4_000
    config.snapshot_every = 2
    trainer = PPOTrainer(config)
    trainer.train()

    path = str(tmp_path / "state.pt")
    trainer.save_state(path)

    revived = PPOTrainer(config)
    revived.load_state(path)

    assert revived.total_hands == trainer.total_hands
    assert revived.update_cycle == trainer.update_cycle
    assert len(revived.snapshots) == len(trainer.snapshots), (
        "the opponent pool did not survive the resume")
    assert revived.snapshots.taken_at == trainer.snapshots.taken_at
    assert revived.snapshots.total_taken == trainer.snapshots.total_taken

    for restored, original in zip(revived.agent.net.parameters(),
                                  trainer.agent.net.parameters()):
        assert torch.equal(restored, original), "policy differs after resume"

    # Adam's moments are the part a policy-only checkpoint silently loses.
    revived_state = revived.optimizer.state_dict()["state"]
    assert revived_state, "optimiser resumed with no state at all"
    original_state = trainer.optimizer.state_dict()["state"]
    for key, values in original_state.items():
        assert torch.allclose(values["exp_avg"], revived_state[key]["exp_avg"])


def test_a_seeded_run_is_actually_reproducible(config):
    """
    `config.seed` reached the environment and nothing else. The network's
    orthogonal initialisation and every action sample came from torch's global
    RNG, and the minibatch shuffle from numpy's — all unseeded. Two runs at the
    same seed produced different policies, one folding 32% of the time and the
    other 64%.

    The endpoint test needs more than repeatability. It measures the trained
    policy against an untrained one *from the same initialisation*, and that
    baseline cannot be constructed at all unless the seed reaches the
    initialiser.
    """
    def weights_of(trainer):
        return np.concatenate([p.detach().cpu().numpy().ravel()
                               for p in trainer.agent.net.parameters()])

    config.seed = 7
    first = PPOTrainer(config)
    start = weights_of(first)
    first.train()
    trained = weights_of(first)

    second = PPOTrainer(config)
    assert np.array_equal(weights_of(second), start), (
        "same seed, different initialisation")
    second.train()
    assert np.array_equal(weights_of(second), trained), (
        "same seed, different policy after training")

    assert not np.array_equal(start, trained), (
        "training did not move the weights, so this proves nothing")
