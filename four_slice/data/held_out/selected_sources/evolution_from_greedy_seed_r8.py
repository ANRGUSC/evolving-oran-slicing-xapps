import math


def controller(history, state, sla, predictors):
    """Improved controller with adaptive allocation and attainment-weighted fund ordering.
    
    Enhancements:
    1. Track attainment history to prioritize slices struggling to meet targets
    2. Use adaptive efficiency margins based on recent delivery performance
    3. Allocate remaining capacity after meeting critical needs to high-weight slices
    4. Smarter churn control with attainment-aware thresholds
    5. Better handling of deep degradations by not over-allocating to saturated channels
    """
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap
    
    # Efficiency curve: delivered Mbps per share point as a function of CQI
    def eff(cqi):
        c = max(5.0, min(15.0, cqi))
        return 0.30 + 2.09 * ((c - 5.0) / 10.0) ** 1.578
    
    # Read current CQI observations
    obs = [history.slices[k].last().cqi if history.slices[k].last() else 15.0 for k in range(n)]
    
    # Adaptive margin: if a slice recently underperformed, be more conservative
    attain_history = state.get("attain_history", [None] * n)
    margins = []
    for k in range(n):
        if attain_history[k] is not None and attain_history[k] < 0.5:
            # Slice struggling: be conservative
            margins.append(0.90)
        elif obs[k] < 7.0:
            # Very poor channel: be conservative
            margins.append(0.85)
        else:
            margins.append(0.93)
    
    rate = [max(0.05, eff(obs[k]) * margins[k]) for k in range(n)]
    
    # Minimum share needed to meet each target (with 0.5 Mbps tolerance)
    need = [max(0, math.ceil((T[k] - 0.5) / rate[k])) for k in range(n)]
    
    # Compute attainment for each slice based on current delivery
    current_mbps = [history.slices[k].latest_mbps() if history.slices[k].latest_mbps() else 0.0 for k in range(n)]
    attain = [1.0 if current_mbps[k] >= T[k] - 0.5 else 0.0 for k in range(n)]
    
    # Update attainment history with exponential smoothing
    new_attain_history = []
    for k in range(n):
        if attain_history[k] is None:
            new_attain_history.append(attain[k])
        else:
            # Smooth with weight 0.7 to current, 0.3 to history
            new_attain_history.append(0.7 * attain[k] + 0.3 * attain_history[k])
    state.set("attain_history", new_attain_history)
    
    # Phase 1: Fund slices in order of (not attaining) then (high weight)
    # Priority: struggling slices first, then high-weight slices
    shares = [0] * n
    used = 0
    
    # Sort by: (attainment ascending, then weight descending)
    # This prioritizes slices that are not meeting targets, weighted by importance
    priority_order = sorted(range(n), key=lambda k: (attain[k], -W[k]))
    
    for k in priority_order:
        if need[k] > 0 and need[k] <= cap - used:
            shares[k] = need[k]
            used += need[k]
    
    # Phase 2: Allocate remaining capacity to high-weight slices for best-effort benefit
    # and to increase chances of meeting targets
    remaining = cap - used
    if remaining > 0:
        # Sort by weight descending for secondary allocation
        secondary_order = sorted(range(n), key=lambda k: -W[k])
        for k in secondary_order:
            if shares[k] < cap and remaining > 0:
                # Allocate up to 30% more than need (but not exceeding cap per-slice)
                extra = min(remaining, max(0, math.ceil(need[k] * 0.3)))
                if extra > 0 and shares[k] + extra <= cap:
                    shares[k] += extra
                    remaining -= extra
    
    # Churn control: compare with previous allocation
    prev = state.get("prev")
    if prev is not None:
        churn = sum(1 for k in range(n) if shares[k] != prev[k])
        # Only hold if churn is very high (more than 2 changes) and we're already
        # meeting most targets
        avg_attain = sum(attain) / n
        if churn > 2 and avg_attain > 0.4:
            shares = list(prev)
    
    state.set("prev", list(shares))
    return shares