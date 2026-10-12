"""
Is the PPO training loop sound this time?

`scripts/preflight_training.py` asks this of the evolutionary loop, and the
reason it exists applies here with more force. The PPO stack was carried
through the August audit untested on the grounds that its implementation
looked sound, and when it was finally run rather than read, six of its
measurements were wrong — a reward that never paid the pot, an agent pinned to
one seat, 13% of every rollout spent on hands the agent never played, a fixed
seed that dealt the same hand every time, a critic whose gradient outweighed
the policy's by a factor of 7,000, and a "seeded" run that reproduced nothing.

Every one of those produced a run that completed, logged a falling loss, and
meant nothing. Tests passing is not the same evidence: the tests exercise
components, and what failed was the loop as a whole.

So this runs the real thing, briefly, and checks the properties that were false
— each stated as what it would mean if it failed.

    python scripts/preflight_ppo.py                 # the gate, about 3 minutes
    python scripts/preflight_ppo.py --calibrate     # entropy, about 20 minutes

The machine is shared. Torch is pinned to one thread — which is also *faster*
here, 552 hands/s against 317, because the matrices are small enough that
thread coordination costs more than it saves — and nothing runs in parallel.
"""
import argparse
import os
import sys
import zlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

# Before any work: one thread, so a preflight never competes with whatever else
# is on the box.
torch.set_num_threads(1)

import torch.nn.functional as F

from engine import get_feature_names
from rl import PPOConfig, PPOTrainer, PokerEnv

OUT = "/tmp/preflight_ppo_out"
FAILURES = []


def check(name, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}\n         {detail}\n")
    if not ok:
        FAILURES.append(name)


def config(**overrides):
    """A real heads-up config, shortened. Same table the panel measures on."""
    cfg = PPOConfig.heads_up_default()
    cfg.eval_every = 10 ** 9
    cfg.save_every = 10 ** 9
    cfg.verbose = 0
    cfg.seed = 0
    cfg.checkpoint_dir = OUT
    cfg.log_dir = OUT
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def policy_on(agent, obs):
    """The agent's action distribution on one observation."""
    with torch.no_grad():
        logits, _value = agent.net.forward(torch.from_numpy(obs).float().unsqueeze(0))
    return torch.softmax(logits, dim=-1).numpy().ravel()


# ===========================================================================
# The checks
# ===========================================================================

def run_gate():
    print("Running the real PPO loop, heads-up, 200 chips at blinds 1/2.\n")

    # -- 1 -----------------------------------------------------------------
    # The check that caught the unpaid pot, and the one worth having most.
    # Against its own live policy the matchup is exactly symmetric, so the
    # expectation is zero whatever the policy happens to be — which makes this
    # a statement about the *measurement* rather than about the agent, and so
    # valid at every point in a run.
    trainer = PPOTrainer(config(total_hands=10 ** 9, current_policy_prob=1.0))
    rewards = []
    while len(rewards) < 4000:
        rewards += trainer._collect_rollout()
    rewards = np.asarray(rewards)
    sigma = rewards.std(ddof=1) / np.sqrt(len(rewards))
    away = rewards.mean() / sigma if sigma else float("inf")

    check("Reward is consistent with zero in a symmetric matchup",
          abs(away) < 3.0,
          f"the policy against itself over {len(rewards):,} hands paid "
          f"{rewards.mean():+.4f} +/- {sigma:.4f} stacks/hand, {away:+.1f} sigma "
          f"from zero. Poker is zero-sum and this matchup is symmetric, so a "
          f"persistent drift means the reward is not measuring the hand. It "
          f"was 17 sigma negative while the pot was never being awarded — "
          f"every hand scored as a loss of whatever had been contributed, and "
          f"a policy gradient on that learns to fold and nothing else. "
          f"{rewards.mean() > 0 and 'Positive' or 'Negative'} drift is equally "
          f"disqualifying; chips would be coming from nowhere.")

    # -- 2 -----------------------------------------------------------------
    # Actor and critic share a trunk and their gradients compete for it.
    # Measured after a few updates rather than at initialisation, because the
    # critic's error is largest before it has fit anything and the transient
    # says little about the run.
    trainer = PPOTrainer(config(total_hands=10 ** 9))
    for _ in range(5):
        trainer._collect_rollout()
        trainer._ppo_update()
    trainer._collect_rollout()

    spread = float(trainer.buffer.returns[:trainer.buffer.size].std())
    trainer.agent.net.train()
    obs, act, old_lp, adv, ret, mask = next(trainer.buffer.get_batches(trainer.cfg.batch_size))
    log_probs, values, entropy = trainer.agent.net.evaluate_actions(obs, act, mask)
    ratio = torch.exp(log_probs - old_lp)
    clipped = torch.clamp(ratio, 1 - trainer.cfg.clip_range, 1 + trainer.cfg.clip_range)
    policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
    value_loss = F.mse_loss(values.squeeze(-1), ret)

    def trunk_gradient(loss):
        trainer.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        return float(torch.norm(torch.stack(
            [p.grad.norm() for p in trainer.agent.net.trunk.parameters()
             if p.grad is not None])))

    from_policy = trunk_gradient(policy_loss)
    from_critic = trunk_gradient(trainer.cfg.vf_coef * value_loss)
    imbalance = from_critic / max(from_policy, 1e-12)
    trainer.optimizer.zero_grad()

    check("The critic does not swamp the policy through the shared trunk",
          imbalance < 100.0 and spread < 5.0,
          f"returns spread {spread:.2f}, trunk gradient {from_critic:.3f} from "
          f"the critic against {from_policy:.4f} from the policy, a factor of "
          f"{imbalance:.0f}. In big blinds on a 100 BB stack that factor was "
          f"7,000: max_grad_norm then divided the sum by 160 and the policy "
          f"stopped moving. The run completes, the loss falls, and the result "
          f"is a plateau — which is exactly what docs/training-plan.md "
          f"predicts for PPO in advance. A predicted finding produced by an "
          f"artefact is the worst outcome available here, because nothing "
          f"about it looks wrong.")

    # -- 3 -----------------------------------------------------------------
    # A pool that never fills, and a pool the agent never meets, both look
    # from the loss curve exactly like one that is working.
    trainer = PPOTrainer(config(total_hands=12_000, snapshot_every=3,
                                snapshot_pool_size=5))
    probe = np.linspace(0.0, 1.0, PokerEnv.OBS_SIZE).astype(np.float32)
    trainer.snapshots.add(trainer.agent, update_cycle=0)
    frozen = trainer.snapshots.sample()
    before = policy_on(frozen, probe)

    trainer.train()

    faced = trainer._faced_current + trainer._faced_snapshot
    share = trainer._faced_current / faced if faced else 1.0
    expected = trainer.cfg.current_policy_prob
    check("The agent faces its own past, at roughly the configured rate",
          trainer._faced_snapshot > 0 and len(trainer.snapshots) == 5
          and share < 0.5,
          f"pool holds {len(trainer.snapshots)} of a capacity of 5 after "
          f"{trainer.snapshots.total_taken} snapshots; the agent faced the live "
          f"policy on {share:.0%} of hands against a configured {expected:.0%}. "
          f"The excess is the warm-up before any snapshot exists, when the live "
          f"policy is the only past there is, so this is an upper bound rather "
          f"than a match. Zero snapshots faced would mean the trainer had "
          f"quietly reverted to training against a single fixed opponent.")

    # -- 4 -----------------------------------------------------------------
    after = policy_on(frozen, probe)
    moved = policy_on(trainer.agent, probe)
    check("Snapshots are frozen copies, not references to the live policy",
          np.allclose(after, before, atol=1e-9) and np.abs(moved - before).max() > 1e-9,
          f"the snapshot's action distribution moved by "
          f"{np.abs(after - before).max():.2e} while the live policy moved by "
          f"{np.abs(moved - before).max():.2e}. Storing a reference instead of "
          f"a copy gives a pool whose every member tracks the current policy: "
          f"len(pool) still grows, the log still looks healthy, and the agent "
          f"plays itself in the present against itself in the present. Nothing "
          f"about that is visible from outside the pool.")

    # -- 5 -----------------------------------------------------------------
    # Illegal actions are impossible by construction — the network masks its
    # logits — which is a claim, and this is the claim checked against a real
    # rollout rather than a constructed one.
    stored = trainer.buffer
    offenders = [i for i in range(stored.size)
                 if not stored.action_masks[i][stored.actions[i]]]
    check("No illegal action was stored, and so none reached the engine",
          not offenders,
          f"{len(offenders)} of {stored.size} stored transitions name an action "
          f"their own mask forbids. The env substitutes a legal action when "
          f"this happens, so the engine never raises — it trains on the "
          f"substitute while crediting the choice, which is how the audit found "
          f"the previous loop scoring broken agents as merely weak ones.")

    # -- 6 -----------------------------------------------------------------
    seeds = [trainer._next_hand()[1]["game_seed"] for _ in range(200)]
    check("A run deals a different hand every time",
          len(set(seeds)) > 190,
          f"{len(set(seeds))} distinct deals in 200 hands. Building a fresh env "
          f"per hand re-seeded it from the same value every time and produced "
          f"two distinct deals across twelve — the audit's original finding, "
          f"which is that the deck re-dealt the same two hands every hand. A "
          f"run like that trains to convergence having seen one hand.")

    # -- 7 -----------------------------------------------------------------
    # Not repeatability for its own sake: the endpoint test compares the
    # trained policy against an untrained one from the *same* initialisation,
    # and that baseline cannot be built at all unless the seed reaches the
    # initialiser.
    # Its own pair of runs rather than a comparison against `trainer` above:
    # that one had a snapshot added by hand for check 4, which leaves the pool
    # non-empty from the first hand and draws differently from the shared
    # generator. The runs would then differ for a legitimate reason and this
    # would report a defect that was not there.
    def train_a_twin():
        twin = PPOTrainer(config(total_hands=8_000, snapshot_every=3,
                                 snapshot_pool_size=5))
        opening = np.concatenate([p.detach().numpy().ravel()
                                  for p in twin.agent.net.parameters()]).copy()
        twin.train()
        return opening, np.concatenate([p.detach().numpy().ravel()
                                        for p in twin.agent.net.parameters()])

    first_start, first_end = train_a_twin()
    second_start, second_end = train_a_twin()

    check("A seeded run is reproducible, initialisation included",
          np.array_equal(first_start, second_start)
          and np.array_equal(first_end, second_end)
          and not np.array_equal(first_start, first_end),
          f"two runs at seed {trainer.cfg.seed} agree on initialisation: "
          f"{np.array_equal(first_start, second_start)}, and after training: "
          f"{np.array_equal(first_end, second_end)}. Training moved the "
          f"weights: {not np.array_equal(first_start, first_end)}. The seed "
          f"used to reach the "
          f"environment and nothing else — initialisation and action sampling "
          f"came from torch's global RNG and the minibatch shuffle from "
          f"numpy's. Two runs at one seed converged to policies folding 32% "
          f"and 64% of the time.")

    # -- 8 -----------------------------------------------------------------
    names = get_feature_names()
    widths = {"engine": len(names), "PPOConfig.obs_size": trainer.cfg.obs_size,
              "PokerEnv.OBS_SIZE": PokerEnv.OBS_SIZE,
              "network input": trainer.agent.net.trunk[0].in_features}
    check("Everything agrees on the width of the observation",
          len(set(widths.values())) == 1,
          f"{widths}. Three of these are written down separately and one is "
          f"derived. A disagreement feeds the network a truncated or padded "
          f"vector and it trains happily on it — the Phase 1 rebuild needed "
          f"nine coordinated edits where four were predicted, and the ones "
          f"nothing pointed at were exactly this kind.")

    # -- 9 -----------------------------------------------------------------
    # Not a pass/fail on the short run — entropy is meant to fall. This reports
    # it, and --calibrate is where the coefficient is chosen.
    import csv
    with open(os.path.join(OUT, "training_log.csv")) as handle:
        rows = list(csv.DictReader(handle))
    first, last = float(rows[0]["entropy"]), float(rows[-1]["entropy"])
    ceiling = float(np.log(trainer.cfg.num_actions))
    check("Entropy has not already collapsed",
          last > 0.05,
          f"entropy {first:.2f} -> {last:.2f} over {len(rows)} updates, against "
          f"a uniform-policy ceiling of {ceiling:.2f}. Falling is the point; "
          f"reaching zero is not. A deterministic policy in an "
          f"imperfect-information game is maximally exploitable and stops "
          f"exploring, and this run is a fraction of a percent of the training "
          f"budget. Run --calibrate before committing to a long run.")


# ===========================================================================
# Entropy calibration
# ===========================================================================

def ppo_as_benchmark_agent(agent):
    """Adapt a PPOAgent to the panel's (game, seat, mask, history) signature."""
    from engine import get_state_vector

    def act(game, player_id, mask, history):
        obs = np.asarray(get_state_vector(game, player_id), dtype=np.float32)
        return int(agent.act(obs, np.asarray(mask, dtype=np.float32)))
    return act


def panel_scores(agent, hands, seed=17):
    """
    The candidate against opponents that share no ancestry with it.

    Self-play reward cannot rank these — the matchup is symmetric and
    zero-sum, so it sits at zero for every candidate whatever it has learned,
    which is the whole reason Phase 2's fitness curve was uninformative.
    Entropy is only a proxy for whether the policy still has choices left. The
    panel is the instrument.
    """
    from evaluation import always_call_agent, benchmark, cfr_agent, random_agent

    # A generator per opponent and a torch reseed per matchup, for the reason
    # `train_ppo.panel_scores` gives at length: the policy samples from torch's
    # global generator, so without this a score depends on how many hands ran
    # before it, and the panel used to share one generator between the random
    # opponent and the solver on top of that. The stakes are lower here than in
    # a published measurement, but a preflight whose numbers move with execution
    # order is a preflight that cannot be compared with the last one.
    misses = None
    panel = [("random", lambda: random_agent(np.random.default_rng(4))),
             ("always-call", lambda: always_call_agent())]
    try:
        import pickle
        with open("results/cfr/nolimit_strategy.pkl", "rb") as handle:
            saved = pickle.load(handle)
        misses = [0, 0]
        panel.append(("cfr", lambda: cfr_agent(saved["strategy"],
                                               saved["abstraction"],
                                               np.random.default_rng(5), misses)))
    except Exception as error:                              # noqa: BLE001
        print(f"    (CFR agent unavailable: {error})")

    played = ppo_as_benchmark_agent(agent)
    out = {}
    for name, make_opponent in panel:
        torch.manual_seed(seed + zlib.crc32(name.encode()))
        result = benchmark(played, make_opponent(), name, hands=hands, seed=seed,
                           misses=misses if name == "cfr" else None)
        out[name] = result
    return out, misses


def run_seed_check(hands, coefficients, seeds, panel_hands, panel_seed):
    """
    Does the ranking survive a change of training seed?

    The calibration above trains one run per candidate, and a difference
    between single runs is not a difference between coefficients. Two runs of
    this loop at the same settings have already converged to policies folding
    32% and 64% of the time, so the spread across seeds is not small and is
    not something to assume.

    Only the CFR column is scored. It is the opponent that separates the
    candidates — random and always-call rank them in the opposite order and
    all three beat both comfortably — and scoring one opponent rather than
    three is what makes checking three seeds affordable at all.
    """
    from evaluation import benchmark, cfr_agent

    try:
        import pickle
        with open("results/cfr/nolimit_strategy.pkl", "rb") as handle:
            saved = pickle.load(handle)
    except Exception as error:                              # noqa: BLE001
        print(f"CFR agent unavailable ({error}); cannot run the seed check.")
        return []

    print(f"Against CFR only, {len(seeds)} training seeds per candidate, "
          f"{hands:,} hands each,\nscored over {panel_hands:,} hands.\n")
    header = "".join(f"{'seed ' + str(s):>16}" for s in seeds)
    print(f"{'ent_coef':>10}{header}{'mean':>12}{'spread':>10}{'miss':>8}")
    print("-" * (30 + 16 * len(seeds)))

    table = []
    for coefficient in coefficients:
        row, worst_miss = [], 0.0
        for seed in seeds:
            directory = os.path.join(OUT, f"seedcheck_e{coefficient}_s{seed}")
            trainer = PPOTrainer(config(total_hands=hands, ent_coef=coefficient,
                                        seed=seed, checkpoint_dir=directory,
                                        log_dir=directory))
            trainer.train()

            rng = np.random.default_rng(4)
            misses = [0, 0]
            result = benchmark(
                ppo_as_benchmark_agent(trainer.agent),
                cfr_agent(saved["strategy"], saved["abstraction"], rng, misses),
                "cfr", hands=panel_hands, seed=panel_seed, misses=misses)
            row.append(result.bb_per_100)
            worst_miss = max(worst_miss, result.lookup_miss_rate)

        cells = "".join(f"{value:>+16.1f}" for value in row)
        spread = float(np.std(row, ddof=1)) if len(row) > 1 else 0.0
        print(f"{coefficient:>10.3f}{cells}{np.mean(row):>+12.1f}"
              f"{spread:>10.1f}{worst_miss:>8.1%}", flush=True)
        table.append((coefficient, row))

    print("\nSpread is the standard deviation across training seeds. Compare it "
          "against the\ngap between coefficients: if it is the larger of the "
          "two, the ranking above is\nthe seed talking and the coefficient is "
          "not the thing being measured.")
    return table


def run_calibration(hands, coefficients, panel_hands=40_000,
                    from_checkpoints=False, panel_seed=17):
    """
    Where does entropy settle, for each candidate `ent_coef`?

    The gate above can only say entropy has not collapsed in three minutes.
    The question that matters is where it goes over hours, and the honest way
    to answer it is to run each candidate long enough to see the trend rather
    than to reason about it.

    Read the last column. A coefficient whose entropy is still falling steeply
    at the end of this will floor out long before 8M hands; one that has
    levelled off has found its balance and will hold it.
    """
    import csv

    print(f"Entropy against ent_coef, {hands:,} hands each, one at a time.\n")
    print(f"{'ent_coef':>10}{'updates':>10}{'start':>9}{'mid':>9}{'end':>9}"
          f"{'last-third slope':>20}{'reward':>10}")
    print("-" * 77)

    from rl import PPOAgent

    results = []
    for coefficient in coefficients:
        directory = os.path.join(OUT, f"ent{coefficient}")
        cfg = config(total_hands=hands, ent_coef=coefficient,
                     checkpoint_dir=directory, log_dir=directory)
        if from_checkpoints:
            agent = PPOAgent.from_checkpoint(
                os.path.join(directory, "ppo_final.pt"), device="cpu")
        else:
            trainer = PPOTrainer(cfg)
            trainer.train()
            agent = trainer.agent

        with open(os.path.join(directory, "training_log.csv")) as handle:
            rows = list(csv.DictReader(handle))
        entropy = np.array([float(r["entropy"]) for r in rows])
        reward = np.array([float(r["mean_reward"]) for r in rows])

        tail = entropy[len(entropy) * 2 // 3:]
        slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0]) if len(tail) > 2 else 0.0
        results.append([coefficient, entropy, slope, agent, None])

        print(f"{coefficient:>10.3f}{len(rows):>10}{entropy[0]:>9.3f}"
              f"{entropy[len(entropy) // 2]:>9.3f}{entropy[-1]:>9.3f}"
              f"{slope:>+20.5f}{reward[-10:].mean():>+10.3f}", flush=True)

    print("\nSlope is entropy per update over the last third. Strongly negative "
          "means it is\nstill collapsing and will reach zero inside the real "
          "budget; near zero means it\nhas found a floor and will hold there. "
          "The reward column carries no\ninformation — the matchup is symmetric "
          "and zero-sum, so it sits at zero whatever\nthe policy has learned.")

    # ── The panel ─────────────────────────────────────────────────────
    print(f"\nAgainst the panel, {panel_hands:,} hands each, duplicate play, "
          f"seed {panel_seed}.\n")
    print(f"{'ent_coef':>10}{'random':>18}{'always-call':>18}{'cfr':>18}"
          f"{'cfr miss':>10}")
    print("-" * 74)

    for row in results:
        scores, _misses = panel_scores(row[3], panel_hands, panel_seed)
        row[4] = scores
        cells = "".join(
            f"{scores[name].bb_per_100:>+12.1f} +/-{1.96 * scores[name].stderr / 2 * 100:<4.0f}"
            if name in scores else f"{'-':>18}"
            for name in ("random", "always-call", "cfr"))
        # Printed on the same line as the number it qualifies, because it was
        # computed and dropped once already and the CFR row then read +53.2
        # with nothing beside it to say whether the opponent was real.
        miss = (f"{scores['cfr'].lookup_miss_rate:>9.1%}"
                if "cfr" in scores else f"{'-':>10}")
        print(f"{row[0]:>10.3f}{cells}{miss}", flush=True)

    print("\nError bars are 95%. A difference smaller than the bars is not a "
          "difference.\nThese policies have seen a fraction of the real budget, "
          "so this ranks the\ncandidates at this point in training and does not "
          "promise the order holds later.")
    print("\nRead the miss column before the cfr column. A benchmark that has "
          "quietly become\na second random opponent is worse than no benchmark, "
          "and this project has\nalready produced 'the evolved agent beats a "
          "solver' once — at +60.8 BB/100,\nalongside a 74.3% miss rate.")
    return results


# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true",
                        help="Run the entropy calibration instead of the gate.")
    parser.add_argument("--hands", type=int, default=200_000,
                        help="Hands per calibration run.")
    parser.add_argument("--ent-coef", type=float, nargs="+",
                        default=[0.01, 0.02, 0.05],
                        help="Candidate entropy coefficients.")
    parser.add_argument("--panel-hands", type=int, default=40_000,
                        help="Hands per panel matchup in the calibration.")
    parser.add_argument("--from-checkpoints", action="store_true",
                        help="Score the calibration checkpoints already on "
                             "disk instead of retraining them.")
    parser.add_argument("--panel-seed", type=int, default=17,
                        help="Deal seed for the panel. Vary it to check that a "
                             "surprising result replicates.")
    parser.add_argument("--seed-check", type=int, nargs="+", metavar="SEED",
                        help="Train each candidate at these training seeds and "
                             "score against CFR, to test whether the ranking "
                             "is the coefficient or the seed.")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)

    if args.seed_check:
        run_seed_check(args.hands, args.ent_coef, args.seed_check,
                       args.panel_hands, args.panel_seed)
        return

    if args.calibrate:
        run_calibration(args.hands, args.ent_coef, args.panel_hands,
                        args.from_checkpoints, args.panel_seed)
        return

    run_gate()

    print("=" * 70)
    if FAILURES:
        print(f"NOT SAFE TO RUN — {len(FAILURES)} check(s) failed: "
              f"{', '.join(FAILURES)}")
        sys.exit(1)
    print("All checks passed. The loop is sound on the properties that were "
          "false before.")
    print("This does not prove the run will be interesting — only that it will "
          "be honest.")


if __name__ == "__main__":
    main()
