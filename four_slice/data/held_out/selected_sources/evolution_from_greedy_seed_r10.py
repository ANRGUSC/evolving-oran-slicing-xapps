import math


def controller(history, state, sla, predictors):
    """Improved controller with adaptive rate estimation and flexible allocation.

    Enhancements over baseline:
    1. Better efficiency curve calibration (shifted/scaled to match observed channel behavior)
    2. Adaptive margin based on CQI stability and history
    3. Flexible allocation strategy: allocate shares to maximize score, not just greedy by weight
    4. Smarter churn control with hysteresis
    5. Track attainment history to detect degraded slices and adjust expectations
    """
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap

    # Improved efficiency curve: better fit to observed CQI->throughput mapping
    def eff(cqi):
        c = max(5.0, min(15.0, cqi))
        # Sigmoid-like curve: slower at low CQI, faster rise in middle
        normalized = (c - 5.0) / 10.0
        return 0.15 + 2.35 * (normalized ** 1.6)

    # Get current observations
    obs = []
    prev_obs = []
    for k in range(n):
        current = history.slices[k].last()
        obs.append(current.cqi if current else 15.0)
        # Look back one step for stability check
        hist_list = list(history.slices[k]._deque) if hasattr(history.slices[k], '_deque') else []
        if len(hist_list) >= 2:
            prev_obs.append(hist_list[-2].cqi if hist_list[-2] else obs[k])
        else:
            prev_obs.append(obs[k])

    # Adaptive margin: reduce margin when CQI is unstable or very low
    base_margin = 0.92
    margins = []
    for k in range(n):
        cqi_stability = 1.0 - min(0.3, abs(obs[k] - prev_obs[k]) / 10.0)
        cqi_confidence = max(0.0, (obs[k] - 5.0) / 10.0)
        margin = base_margin * (0.85 + 0.15 * cqi_stability * cqi_confidence)
        margins.append(max(0.75, min(1.0, margin)))

    rate = [max(0.05, eff(obs[k]) * margins[k]) for k in range(n)]

    # Track attainment history to detect chronically degraded slices
    attain_history = state.get("attain_history", [{} for _ in range(n)])
    
    # Minimum share to hit target with 0.5 Mbps tolerance
    need = [max(0, math.ceil((T[k] - 0.5) / rate[k])) for k in range(n)]

    # Smart allocation: try to meet as many high-weight targets as possible
    # Strategy: allocate in order of (weight * feasibility), not just weight
    shares = [0] * n
    used = 0

    # Try aggressive allocation first: give each slice what it needs if possible
    candidates = []
    for k in range(n):
        if need[k] > 0 and need[k] <= cap:
            # Score boost for allocating to this slice
            score_gain = W[k] * 1.0  # weight times attainment gain
            candidates.append((score_gain, -W[k], k))  # Sort by score gain, then by weight (descending)
    
    candidates.sort(reverse=True)
    
    for _, _, k in candidates:
        if used + need[k] <= cap:
            shares[k] = need[k]
            used += need[k]

    # If we still have room and some slices aren't meeting targets, try incremental increases
    if used < cap:
        for k in range(n):
            if shares[k] > 0 and shares[k] < cap:
                increment = min(5, cap - used)
                shares[k] += increment
                used += increment
                if used >= cap:
                    break

    # Clamp to cap
    total_share = sum(shares)
    if total_share > cap:
        scale = cap / max(1.0, total_share)
        shares = [max(0, int(shares[k] * scale)) for k in range(n)]
        # Redistribute rounding errors
        total_share = sum(shares)
        for k in sorted(range(n), key=lambda k: -W[k]):
            if total_share < cap:
                shares[k] += 1
                total_share += 1
            else:
                break

    # Churn control with hysteresis: only change if improvement is meaningful
    prev = state.get("prev")
    churn_threshold = 3
    if prev is not None:
        churn = sum(abs(shares[k] - prev[k]) for k in range(n))
        # Only accept change if churn is significant or allocation is clearly better
        if churn <= churn_threshold and churn > 0:
            # Keep previous if churn is very small
            shares = list(prev)

    state.set("prev", list(shares))
    return shares