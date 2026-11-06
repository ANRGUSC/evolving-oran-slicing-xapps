"""Hand-written expert rule (paper Sec. VI-B, Table II: fitness 0.541, eta 53.4%).

The rule sees the same one-tick-lagged observations as the evolved controllers and knows the
nominal efficiency curve of Eq. 11. Each tick it extrapolates every slice's CQI from the last two
observations (strength alpha), converts the forecast to an efficiency with a conservative
multiplier m, computes the smallest share that meets each target, and then evaluates every feasible
funded subset plus the option of keeping the previous allocation, choosing the one with the highest
predicted composite score (including the change penalty).

alpha in {0, 0.5, 1} and m in {1.0, 0.9, 0.8} were tuned on the 16 development traces used during
search; the best setting (alpha = 0, m = 0.9) scores 0.541 and is the module-level `controller`
that reproduce.py scores. Running this file directly re-runs the nine-setting grid:

    python expert_rule.py            (from inside controllers/)
"""
import math
import os
import statistics
import sys

try:
    import simulator as E
    import scorer as H
except ImportError:  # run directly: put ../engine on the path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
    import simulator as E
    import scorer as H

assert E.NG == 4 and E.GCAP == 70 and E.TICKS == 360 and E.DEV == list(range(16))


def policy_factory(alpha, scale):
    def controller(history, state, sla, predictors):
        observed = [history.slices[k].last().cqi for k in range(sla.n)]
        previous_cqi = state.get('prior_observed_cqi', None)
        pred = [min(E.CQI_HI, max(E.CQI_LO, c + (alpha * (c - previous_cqi[k])
                if previous_cqi is not None else 0.0))) for k, c in enumerate(observed)]
        est_eff = [E.e_of_cqi(c) * scale for c in pred]
        required = [max(0, math.ceil((sla.slices[k].target_mbps - E.OK_EPS) / est_eff[k]))
                    for k in range(sla.n)]
        prev = state.get('previous_action', None)
        candidates = []
        for mask in range(1 << sla.n):
            shares = [required[k] if mask & (1 << k) else 0 for k in range(sla.n)]
            if sum(shares) <= sla.guaranteed_cap:
                candidates.append(shares)
        if prev is not None:
            candidates.append(list(prev))
        # This is subset enumeration plus a hold candidate, not exact action-space optimization.
        selected = max(candidates, key=lambda s: E.composite_tick(est_eff, s, prev))
        assert len(selected) == sla.n and all(isinstance(x, int) and x >= 0 for x in selected)
        assert sum(selected) <= sla.guaranteed_cap
        state.set('prior_observed_cqi', list(observed))
        state.set('previous_action', list(selected))
        return list(selected)
    return controller


ALPHA, SCALE = 0.0, 0.9          # the setting reported in the paper
controller = policy_factory(ALPHA, SCALE)

GRID = [(a, s) for a in (0.0, 0.5, 1.0) for s in (1.0, 0.9, 0.8)]   # fixed before the probe was run


def main():
    print("expert rule: nine settings on the 16 development traces (worst-env = min over seeds)")
    rows = []
    for alpha, scale in GRID:
        vals = [H.eval_prompt_controller(policy_factory(alpha, scale), seed) for seed in E.DEV]
        rows.append((min(vals), statistics.fmean(vals), alpha, scale))
    rows.sort(reverse=True)
    print(f"  {'alpha':>5} {'m':>4} {'worst-env':>10} {'mean':>8}")
    for worst, mean, alpha, scale in rows:
        mark = "   <- paper (0.541)" if (alpha, scale) == (ALPHA, SCALE) else ""
        print(f"  {alpha:>5.1f} {scale:>4.1f} {worst:>10.4f} {mean:>8.4f}{mark}")


if __name__ == '__main__':
    main()
