"""
Anytime-valid comparisons for arena bursts: a verdict that stays honest however often it is read.

The burst rule ("believe a set after 2 or 3 bursts") is a sequential test, because we look after
every burst and stop when the number looks convincing. A fixed-sample test read that way is not at
5 percent: Armitage, McPherson and Rowe (1969) put five looks at about 14 percent false positives
and ten at about 19. Everything here is instead a confidence sequence, an interval that covers the
true value at every sample size at once with probability 1 - alpha, so a verdict can be read after
each burst, or after each match, and the error rate is still alpha (Howard, Ramdas, McAuliffe and
Sekhon 2021, "Time-uniform, nonparametric, nonasymptotic confidence sequences", Ann. Statist. 49).
The price is width: at a fixed n the interval is wider than a t-interval, and that is what buys
the right to peek.

Match outcomes: the conjugate beta-binomial mixture
---------------------------------------------------
A match is won or lost, so the per-version rate is a Bernoulli parameter. For each candidate rate
p the mixture likelihood ratio

    M_n(p) = integral of L_n(q) dBeta(q; a, b) / L_n(p)

is a nonnegative martingale with mean 1 when p is true, so by Ville's inequality it crosses 1/alpha
with probability at most alpha, ever. The confidence sequence is {p : M_n(p) < 1/alpha}. This is
Robbins' (1970) mixture and the beta-binomial case of Howard et al. 2021 (their section 3), and
testing p = 0.5 with it is exactly the mixture SPRT of Johari, Koomen, Pekelis and Walsh (2017,
"Peeking at A/B tests"). It was chosen over the betting confidence sequence of Waudby-Smith and
Ramdas because for a Bernoulli the likelihood is known exactly, so the mixture is closed form,
deterministic (no grid, no predictable tuning) and already near-optimal; betting earns its keep on
bounded data whose distribution is unknown, which is the chips case below.

The prior is Beta(5, 5), fixed in advance from simulation only, never from match logs. Any fixed
prior keeps the guarantee; the prior only decides which effects are found fastest. Beta(5, 5)
spreads its mass over roughly 35 to 65 percent, where arena win rates actually sit, and on
simulated 200-match runs it found a true 60 percent rate in 57 percent of runs against 47 for the
uniform Beta(1, 1), with 70 percent found at a median of 44 matches under either.

Two versions are compared by giving each its own sequence at alpha / 2 and differencing the
intervals. By the union bound both cover their rates at all times with probability 1 - alpha, so
the difference interval does too. It is conservative, but it needs nothing about how the two
versions' matches interleave, which matters because bursts of different versions are played on
different days against different opponents, never paired.

Chips per decision hand: the hedged betting confidence sequence
---------------------------------------------------------------
Per-hand nets are bounded but very far from normal: most hands move a few hundred chips while an
all-in moves up to 10,000, the effective stack (the smaller stack can never exceed half of the
20,000 chips in play). Here the hedged-capital betting sequence of Waudby-Smith and Ramdas (2024,
"Estimating means of bounded random variables by betting", JRSS B 86) is used: for each candidate
mean m, two gamblers bet on the hands coming in above or below m with predictable stakes, and m is
rejected once their combined wealth reaches 1/alpha. It adapts to the observed variance, so a
stream of small pots tightens it quickly while an all-in costs only what its size warrants; a
sub-Gaussian bound would have to assume the all-in variance on every hand. Its assumptions are
boundedness and a constant conditional mean, E[X_t | past] = m, which is weaker than independence
but is still an assumption: within a match the mean shifts with stack depth and with an adapting
opponent, so the guarantee is on the average conditional mean, and that caveat is real.

Clipping changes the estimand. With the default bound of 10,000 nothing is clipped, because no
hand can move more than the effective stack, and the interval is for the true mean. A smaller clip
gives a tighter interval for the mean of the clipped nets, which is a different number: it
discounts the all-ins that decided v6. Callers must print the clip next to the interval.

Projections ("about N more matches") assume the future arrives at exactly the observed rate. They
are estimates for planning, not part of the guarantee, and at a small n the observed rate itself
is noisy, so the projection can be off by a factor of several in either direction.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.special import betaln, xlogy

ALPHA = 0.05
PRIOR = (5.0, 5.0)
NO_CLIP = 10_000.0          # the effective stack at Chipzen's 10,000 starting stacks, so nothing is clipped
PROJECTION_CAP = 1_000_000
HAND_CAP = 200_000          # chips projections recompute a grid per step, so they stop sooner


@dataclass
class Verdict:
    """One comparison, as printed after a burst. `projection` is an estimate, never a guarantee."""
    what: str
    n: tuple
    estimate: float
    interval: tuple
    verdict: str
    projection: str = ""
    notes: list = field(default_factory=list)


# Match outcomes ---------------------------------------------------------------------------------

def _log_mixture(s, n, prior):
    a, b = prior
    return betaln(a + s, b + n - s) - betaln(a, b)


def _loglik(p, s, n):
    return xlogy(s, p) + xlogy(n - s, 1.0 - p)


def log_mixture_ratio(p, s, n, prior=PRIOR):
    """log M_n(p); it reaching log(1/alpha) rejects p at any time."""
    return _log_mixture(s, n, prior) - _loglik(p, s, n)


def bernoulli_cs(s, n, alpha=ALPHA, prior=PRIOR):
    """The rates not yet rejected after s wins in n matches: an interval, since the log-likelihood is concave."""
    if n == 0:
        return 0.0, 1.0
    level = _log_mixture(s, n, prior) + np.log(alpha)
    phat = s / n
    f = lambda p: _loglik(p, s, n) - level           # positive inside the set
    lo = 0.0 if f(0.0) > 0 else brentq(f, 0.0, phat, xtol=1e-10) if phat > 0 else 0.0
    hi = 1.0 if f(1.0) > 0 else brentq(f, phat, 1.0, xtol=1e-10) if phat < 1 else 1.0
    return lo, hi


def _bisect(f, lo, hi, iters=60):
    """Vectorised bisection for f(lo) <= 0 < f(hi) elementwise; 60 halvings reach float resolution."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        inside = f(mid) > 0
        lo, hi = np.where(inside, lo, mid), np.where(inside, mid, hi)
    return hi


def bernoulli_cs_path(outcomes, alpha=ALPHA, prior=PRIOR):
    """
    The running intersection over every prefix, which a confidence sequence permits and which never widens.

    Every prefix's interval is solved at once by vectorised bisection, because simulations call
    this thousands of times and a scalar root-finder per prefix was a minute of test time.
    """
    w = np.asarray(outcomes, dtype=float)
    if w.size == 0:
        return 0.0, 1.0
    n = np.arange(1, w.size + 1, dtype=float)
    s = np.cumsum(w)
    level = _log_mixture(s, n, prior) + np.log(alpha)
    f = lambda p: _loglik(p, s, n) - level
    phat = s / n
    zero, one = np.zeros_like(n), np.ones_like(n)
    lo = np.where(f(zero) > 0, 0.0, _bisect(f, zero, phat))
    # the upper edge, found as a lower edge of the mirrored rate q = 1 - p
    hi = np.where(f(one) > 0, 1.0, 1.0 - _bisect(lambda q: f(1.0 - q), zero, 1.0 - phat))
    lo, hi = float(lo.max()), float(hi.min())
    if lo > hi:                                       # possible only after a coverage failure; keep it visible
        lo = hi = float(phat[-1])
    return lo, hi


def _first_n(excluded, n0, cap=PROJECTION_CAP):
    """Smallest n >= n0 at which excluded(n) holds, by doubling then bisection; None past the cap."""
    if excluded(n0):
        return n0
    lo, hi = n0, max(2 * n0, n0 + 20)
    while not excluded(hi):
        lo, hi = hi, hi * 2
        if hi > cap:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if excluded(mid) else (mid, hi)
    return hi


def match_verdict_one(outcomes, p0=0.5, alpha=ALPHA, prior=PRIOR, name="A"):
    """One version's match win rate against p0."""
    n, s = len(outcomes), int(sum(outcomes))
    v = Verdict(f"{name} match win rate vs {p0:.0%}", (n,), s / n if n else float("nan"), (0.0, 1.0), "undecided")
    if n == 0:
        v.projection = "no matches yet"
        return v
    v.interval = bernoulli_cs_path(outcomes, alpha, prior)
    if v.interval[0] > p0:
        v.verdict = f"{name} better than {p0:.0%}"
    elif v.interval[1] < p0:
        v.verdict = f"{name} worse than {p0:.0%}"
    else:
        phat = s / n
        if phat == p0:
            v.projection = "no end in sight: the observed rate equals the null"
        else:
            need = _first_n(lambda m: log_mixture_ratio(p0, phat * m, m, prior) >= -np.log(alpha), n)
            v.projection = _more(need, n, "matches")
    return v


def match_verdict_two(a, b, alpha=ALPHA, prior=PRIOR, names=("A", "B")):
    """Difference of two versions' win rates, each sequence at alpha / 2 so the pair holds at alpha."""
    na, nb = len(a), len(b)
    ea = sum(a) / na if na else float("nan")
    eb = sum(b) / nb if nb else float("nan")
    v = Verdict(f"{names[0]} minus {names[1]} match win rate", (na, nb), ea - eb, (-1.0, 1.0), "undecided")
    if na == 0 or nb == 0:
        v.projection = "a version has no matches yet"
        return v
    la, ha = bernoulli_cs_path(a, alpha / 2, prior)
    lb, hb = bernoulli_cs_path(b, alpha / 2, prior)
    v.interval = (la - hb, ha - lb)
    if v.interval[0] > 0:
        v.verdict = f"{names[0]} better"
    elif v.interval[1] < 0:
        v.verdict = f"{names[1]} better"
    elif ea == eb:
        v.projection = "no end in sight: the observed rates are equal"
    else:
        def excluded(k):                              # k more matches for each version, at the observed rates
            l1, h1 = bernoulli_cs(ea * (na + k), na + k, alpha / 2, prior)
            l2, h2 = bernoulli_cs(eb * (nb + k), nb + k, alpha / 2, prior)
            return l1 > h2 or l2 > h1
        need = _first_n(excluded, 0)
        v.projection = _more(need, 0, "matches for each version", already=False)
    return v


def _more(need, n, unit, already=True):
    if need is None:
        return f"estimate: more than {PROJECTION_CAP:,} {unit} at the observed rate, so effectively never"
    return f"estimate: about {max(need - n, 1) if already else max(need, 1):,} more {unit} at the observed rate"


# Chips per hand ---------------------------------------------------------------------------------

def _stakes(y, alpha):
    """Waudby-Smith and Ramdas' predictable plug-in stake, before the per-candidate truncation."""
    t = np.arange(1, y.size + 1, dtype=float)
    mu = (0.5 + np.cumsum(y)) / (t + 1)
    var = (0.25 + np.cumsum((y - mu) ** 2)) / (t + 1)
    var_prev = np.concatenate(([0.25], var[:-1]))     # predictable: the stake on hand t uses hands before t only
    return np.sqrt(2 * np.log(2 / alpha) / (var_prev * t * np.log1p(t)))


def betting_log_wealth(y, m, alpha=ALPHA, c=0.5, theta=0.5, chunk=2048):
    """
    Running maximum over time of the hedged log wealth for each candidate mean in m (all on [0, 1]).

    Taking the maximum makes the reported set the running intersection, so a candidate once rejected
    stays rejected; that is what lets a reader stop at whichever burst they like.
    """
    y = np.asarray(y, dtype=float)
    m = np.asarray(m, dtype=float)
    best = np.full(m.shape, -np.inf)
    if y.size == 0:
        return np.zeros(m.shape)
    lam = _stakes(y, alpha)
    cap_up, cap_dn = c / m, c / (1.0 - m)             # keeps every factor positive, so wealth never hits zero
    up = np.zeros(m.shape)
    dn = np.zeros(m.shape)
    for start in range(0, y.size, chunk):
        ys = y[start:start + chunk, None]
        ls = lam[start:start + chunk, None]
        a = np.cumsum(np.log1p(np.minimum(ls, cap_up) * (ys - m)), axis=0) + up
        b = np.cumsum(np.log1p(-np.minimum(ls, cap_dn) * (ys - m)), axis=0) + dn
        w = np.logaddexp(a + np.log(theta), b + np.log(1 - theta))
        best = np.maximum(best, w.max(axis=0))
        up, dn = a[-1], b[-1]
    return best


def mean_cs(x, bound=NO_CLIP, alpha=ALPHA, grid=1001):
    """
    Confidence sequence for the mean of x after clipping to [-bound, bound], in x's own units.

    Returns (lo, hi, clipped) where clipped counts the values the bound moved. The grid sets the
    resolution, 2 * bound / grid, which at the defaults is 20 chips, small against intervals of
    hundreds and five times cheaper than a 5-chip grid.
    """
    x = np.asarray(x, dtype=float)
    clipped = int(np.sum(np.abs(x) > bound))
    if x.size == 0:
        return -bound, bound, 0
    y = (np.clip(x, -bound, bound) + bound) / (2 * bound)
    m = (np.arange(grid) + 0.5) / grid
    keep = betting_log_wealth(y, m, alpha) < np.log(1 / alpha)
    if not keep.any():                                # every grid point rejected: report the cell nearest the mean
        j = int(np.argmin(np.abs(m - y.mean())))
        keep[j] = True
    lo, hi = m[keep].min() - 0.5 / grid, m[keep].max() + 0.5 / grid
    return lo * 2 * bound - bound, hi * 2 * bound - bound, clipped


def _tile(x, n):
    """The observed hands repeated to length n: the future at exactly the observed mean and spread."""
    return np.resize(np.asarray(x, dtype=float), n)


def chips_verdict_one(x, mu0=0.0, bound=NO_CLIP, alpha=ALPHA, name="A", per_match=None):
    """One version's chips per decision hand against mu0."""
    x = np.asarray(x, dtype=float)
    v = Verdict(f"{name} chips per decision hand vs {mu0:+.0f}", (x.size,),
                float(np.clip(x, -bound, bound).mean()) if x.size else float("nan"), (-bound, bound), "undecided")
    if x.size == 0:
        v.projection = "no hands yet"
        return v
    lo, hi, clipped = mean_cs(x, bound, alpha)
    v.interval = (lo, hi)
    v.notes.append(f"clip ±{bound:,.0f} chips, {clipped} of {x.size} hands clipped")
    if lo > mu0:
        v.verdict = f"{name} above {mu0:+.0f}"
    elif hi < mu0:
        v.verdict = f"{name} below {mu0:+.0f}"
    elif v.estimate == mu0:
        v.projection = "no end in sight: the observed mean equals the null"
    else:
        y0 = (mu0 + bound) / (2 * bound)
        need = _first_n(lambda n: betting_log_wealth((np.clip(_tile(x, n), -bound, bound) + bound) / (2 * bound),
                                                     np.array([y0]), alpha)[0] >= np.log(1 / alpha), x.size, HAND_CAP)
        v.projection = _more_hands(need, x.size, per_match, already=True)
    return v


def chips_verdict_two(a, b, bound=NO_CLIP, alpha=ALPHA, names=("A", "B"), per_match=None):
    """Difference of two versions' chips per decision hand, each at alpha / 2."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ca, cb = np.clip(a, -bound, bound), np.clip(b, -bound, bound)
    est = float(ca.mean() - cb.mean()) if a.size and b.size else float("nan")
    v = Verdict(f"{names[0]} minus {names[1]} chips per decision hand", (a.size, b.size), est,
                (-2 * bound, 2 * bound), "undecided")
    if a.size == 0 or b.size == 0:
        v.projection = "a version has no hands yet"
        return v
    la, ha, xa = mean_cs(a, bound, alpha / 2)
    lb, hb, xb = mean_cs(b, bound, alpha / 2)
    v.interval = (la - hb, ha - lb)
    v.notes.append(f"clip ±{bound:,.0f} chips, {xa} of {a.size} and {xb} of {b.size} hands clipped")
    if v.interval[0] > 0:
        v.verdict = f"{names[0]} better"
    elif v.interval[1] < 0:
        v.verdict = f"{names[1]} better"
    elif est == 0:
        v.projection = "no end in sight: the observed means are equal"
    else:
        def excluded(k):                              # k more hands for each, coarse grid since only the sign matters
            l1, h1, _ = mean_cs(_tile(a, a.size + k), bound, alpha / 2, grid=401)
            l2, h2, _ = mean_cs(_tile(b, b.size + k), bound, alpha / 2, grid=401)
            return l1 > h2 or l2 > h1
        need = _first_n(excluded, 0, HAND_CAP)
        v.projection = _more_hands(need, 0, per_match, already=False, each=True)
    return v


def _more_hands(need, n, per_match, already, each=False):
    unit = "decision hands" + (" for each version" if each else "")
    if need is None:
        return f"estimate: more than {HAND_CAP:,} {unit} at the observed rate, so effectively never"
    more = max(need - n if already else need, 1)
    text = f"estimate: about {more:,} more {unit} at the observed rate"
    if per_match:
        text += f", roughly {int(np.ceil(more / per_match)):,} matches at {per_match:.0f} decision hands a match"
    return text
