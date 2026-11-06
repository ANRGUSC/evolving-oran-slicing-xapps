import math

def controller(history, state, sla, predictors):
    """Improved weight-ordered controller with adaptive margin and demand awareness."""
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap
    
    def eff(cqi):
        c = max(5.0, min(15.0, cqi))
        return 0.30 + 2.09 * ((c - 5.0) / 10.0) ** 1.578
    
    # Read current CQI and delivered throughput from history
    obs = []
    last_mbps = []
    for k in range(n):
        sample = history.slices[k].last()
        if sample:
            obs.append(sample.cqi)
            last_mbps.append(history.slices[k].latest_mbps())
        else:
            obs.append(15.0)
            last_mbps.append(0.0)
    
    # Adaptive margin based on attainment history
    prev_attain = state.get("attain", [0.5] * n)
    
    # For slices that consistently miss targets, reduce margin; for those that hit, increase confidence
    margin = [0.92 for _ in range(n)]
    for k in range(n):
        if prev_attain[k] < 0.3:
            margin[k] = 0.88  # More aggressive allocation for struggling slices
        elif prev_attain[k] > 0.8:
            margin[k] = 0.95  # Conservative for stable slices
    
    rate = [max(0.05, eff(obs[k]) * margin[k]) for k in range(n)]
    
    # Calculate minimum shares needed with 0.5 Mbps tolerance
    need = [max(1, math.ceil((T[k] - 0.5) / rate[k])) for k in range(n)]
    
    # Two-phase allocation:
    # Phase 1: Fund high-weight slices that are struggling to meet targets
    shares = [0] * n
    used = 0
    
    # Sort by weight descending, but prioritize slices with low recent attainment
    def priority(k):
        attain_score = -prev_attain[k]  # Lower attainment = higher priority
        return (-W[k], attain_score)
    
    for k in sorted(range(n), key=priority):
        min_share = need[k]
        if used + min_share <= cap:
            shares[k] = min_share
            used += min_share
    
    # Phase 2: Distribute remaining capacity by weight to boost best-effort or improve margins
    remaining = cap - used
    if remaining > 0:
        # Give extra shares to high-weight slices proportionally
        weight_sum = sum(W[k] for k in range(n))
        if weight_sum > 0:
            for k in sorted(range(n), key=lambda k: -W[k]):
                if remaining <= 0:
                    break
                # Allocate at most 5 extra shares per slice
                extra = min(5, math.ceil(remaining * W[k] / weight_sum))
                shares[k] += extra
                remaining -= extra
    
    # Clamp to guaranteed cap
    total = sum(shares)
    if total > cap:
        scale = cap / total
        shares = [max(0, int(shares[k] * scale)) for k in range(n)]
        # Re-distribute any rounding loss
        total = sum(shares)
        deficit = cap - total
        for k in sorted(range(n), key=lambda k: -W[k]):
            if deficit <= 0:
                break
            add = min(deficit, 1)
            shares[k] += add
            deficit -= add
    
    # Churn control: smooth transitions for stability
    prev = state.get("prev")
    if prev is not None:
        change = sum(abs(shares[k] - prev[k]) for k in range(n))
        if change <= 3 and prev is not None:
            # Small change: blend toward previous to reduce churn penalty
            shares = [int(0.7 * shares[k] + 0.3 * prev[k]) for k in range(n)]
    
    state.set("prev", list(shares))
    
    # Update attainment tracking for next tick
    new_attain = []
    for k in range(n):
        # Estimate what we'll deliver with current share
        estimated_mbps = rate[k] * shares[k]
        attain = 1.0 if estimated_mbps >= T[k] - 0.5 else 0.0
        new_attain.append(attain)
    state.set("attain", new_attain)
    
    return shares