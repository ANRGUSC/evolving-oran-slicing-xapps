import math


def controller(history, state, sla, predictors):
    """Clean weight-ordered greedy controller (worst-env ~0.20 on DEV 0-15).

    A deliberately simple, readable 'launchpad' seed: read each slice's lagged CQI,
    estimate its per-share rate with a small conservative margin, fund slices in
    descending weight order up to the guaranteed cap, and hold the previous
    allocation when the change is tiny (churn control). No cross-slice arbitration
    beyond weight-ordered greedy triage.
    """
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap

    # Efficiency curve: delivered Mbps per share point as a function of CQI.
    def eff(cqi):
        c = max(5.0, min(15.0, cqi))
        return 0.30 + 2.09 * ((c - 5.0) / 10.0) ** 1.578

    MARGIN = 0.95  # trust ~95% of the estimated rate (tuned so worst-env ~= 0.20 on DEV)
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    rate = [max(0.05, eff(obs[k]) * MARGIN) for k in range(n)]

    # Minimum share needed to attain each target (with the scorer's 0.5 Mbps tolerance).
    need = [max(0, math.ceil((T[k] - 0.5) / rate[k])) for k in range(n)]

    # Fund slices in descending weight order while they fit under the guaranteed cap.
    shares = [0] * n
    used = 0
    for k in sorted(range(n), key=lambda k: -W[k]):
        if need[k] <= cap and used + need[k] <= cap:
            shares[k] = need[k]
            used += need[k]

    # Churn control: keep the previous allocation if the change is negligible.
    prev = state.get("prev")
    if prev is not None and sum(abs(shares[k] - prev[k]) for k in range(n)) <= 2:
        shares = list(prev)
    state.set("prev", list(shares))
    return shares
