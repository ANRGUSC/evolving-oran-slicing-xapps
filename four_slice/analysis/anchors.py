"""Reference policies of the four-slice experiment (paper Table II and the eta anchors of Table III):
the oracle reference (best minimum-funded subset per tick with the true current channel) and the
weight-ordered greedy policy with the true current channel or one-tick-lagged observations.
Running this file prints the development-trace values (1.0555, 0.9104, -0.0974)."""
import math
from pathlib import Path
import statistics
import sys
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
import simulator as E


def min_viable(efficiency, *, funding_tolerance=None):
    """Smallest share that funds each slice at the given efficiencies.

    funding_tolerance=None (default) funds ceil((T_k - OK_EPS)/e_k), the attainment
    threshold of Eq. (4); this gives the oracle reference of Table II (1.0555).
    funding_tolerance=0.0 funds the full target, ceil(T_k/e_k); this is the reference
    policy behind the held-out normalization of Table III (1.0523). Scoring always
    uses OK_EPS.
    """
    tolerance = E.OK_EPS if funding_tolerance is None else funding_tolerance
    if not math.isfinite(tolerance) or not 0 <= tolerance < min(E.T):
        raise ValueError('funding_tolerance must be finite and smaller than every target')
    return [math.ceil((E.T[k] - tolerance) / efficiency[k]) if efficiency[k] > 0 else 10**9
            for k in range(E.NG)]


def _feasible_subset_allocs(efficiency, *, funding_tolerance=None):
    costs = min_viable(efficiency, funding_tolerance=funding_tolerance)
    choices = []
    for size in range(E.NG + 1):
        for funded in combinations(range(E.NG), size):
            if sum(costs[k] for k in funded) <= E.GCAP:
                shares = [0] * E.NG
                for k in funded:
                    shares[k] = costs[k]
                choices.append(shares)
    return choices


def _subset_episode(seed, *, funding_tolerance=None):
    """Choose the highest-reward minimum-funded subset using current channels.

    This reference only considers minimum-funded subsets. With switching costs
    it is not an optimum over all legal allocations or over the full episode.
    """
    efficiency = E.eff_series(seed)
    previous, scores = None, []
    for tick in range(E.TICKS):
        current = [efficiency[k][tick] for k in range(E.NG)]
        shares = max(_feasible_subset_allocs(current, funding_tolerance=funding_tolerance),
                     key=lambda candidate: E.composite_tick(current, candidate, previous))
        scores.append(E.composite_tick(current, shares, previous))
        previous = shares
    return statistics.fmean(scores)


def subset_reference(seeds=E.DEV, *, funding_tolerance=None):
    return min(_subset_episode(seed, funding_tolerance=funding_tolerance) for seed in seeds)


def _greedy_episode(seed, lag):
    efficiency = E.eff_series(seed)
    order = sorted(range(E.NG), key=lambda k: -E.W[k])
    previous, scores = None, []
    for tick in range(E.TICKS):
        observed_tick = tick - 1 if lag and tick > 0 else tick
        costs = min_viable([efficiency[k][observed_tick] for k in range(E.NG)])
        shares, remaining = [0] * E.NG, E.GCAP
        for k in order:
            if costs[k] <= remaining:
                shares[k] = costs[k]
                remaining -= costs[k]
        current = [efficiency[k][tick] for k in range(E.NG)]
        scores.append(E.composite_tick(current, shares, previous))
        previous = shares
    return statistics.fmean(scores)


def greedy_current(seeds=E.DEV):
    return min(_greedy_episode(seed, lag=False) for seed in seeds)


def greedy_lagged(seeds=E.DEV):
    return min(_greedy_episode(seed, lag=True) for seed in seeds)


if __name__ == '__main__':
    print(f'Subset reference: {subset_reference():.4f}')
    print(f'Weight-ordered, current channel: {greedy_current():.4f}')
    print(f'Weight-ordered, one-tick lag: {greedy_lagged():.4f}')
