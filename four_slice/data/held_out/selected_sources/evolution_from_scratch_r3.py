def controller(history, state, sla, predictors):
    """
    PRB allocation controller for 4 guaranteed slices + 1 best-effort slice.
    Maximizes weighted score considering slice attainment, best-effort throughput, and churn.
    """
    
    # Initialize state tracking
    if state.get('prev_shares') is None:
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
    if state.get('cqi_history') is None:
        state.set('cqi_history', [[] for _ in range(4)])
    
    prev_shares = state.get('prev_shares', [0.0, 0.0, 0.0, 0.0])
    cqi_history = state.get('cqi_history', [[] for _ in range(4)])
    
    n = sla.n  # 4
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    pool_max = sla.pool_max  # 100
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Collect current metrics
    cqis = []
    last_mbps = []
    last_shares = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is not None:
            cqis.append(sample.cqi)
            last_mbps.append(history.slices[k].latest_mbps())
            last_shares.append(sample.prb_share)
            # Track CQI history
            cqi_history[k].append(sample.cqi)
            if len(cqi_history[k]) > 50:
                cqi_history[k].pop(0)
        else:
            cqis.append(10.0)  # neutral default
            last_mbps.append(0.0)
            last_shares.append(0.0)
    
    # Estimate rate per share for each slice based on CQI and recent delivery
    # rate_k = Mbps per unit share; use observed ratio when available
    rates = []
    for k in range(n):
        if last_shares[k] > 0.1 and last_mbps[k] > 0:
            # Observed rate per share
            rate_k = last_mbps[k] / last_shares[k]
        else:
            # Estimate from CQI: map CQI 5..15 to rate roughly 0.5..3.5 Mbps/share
            cqi_norm = (cqis[k] - 5.0) / 10.0
            cqi_norm = max(0.0, min(1.0, cqi_norm))
            rate_k = 0.5 + 3.0 * cqi_norm
        rates.append(rate_k)
    
    # Waterfilling-inspired allocation: prioritize slices needing more help
    # Compute "slack": how much share needed to reach target
    needs = []
    for k in range(n):
        if rates[k] > 0.1:
            needed_share = targets[k] / rates[k]
        else:
            needed_share = guaranteed_cap  # unreachable
        needs.append(needed_share)
    
    # Start with a proportional allocation weighted by weights and need
    shares = [0.0] * n
    total_need = sum(max(0.1, needs[k]) for k in range(n))
    
    # First pass: allocate proportionally, respecting targets
    for k in range(n):
        # Weight by importance and urgency
        urgency = weights[k] / max(0.1, rates[k])
        shares[k] = (urgency / sum(weights[i] / max(0.1, rates[i]) for i in range(n))) * guaranteed_cap
    
    # Second pass: refine by checking attainment
    # Try to meet all targets if possible
    sorted_indices = sorted(range(n), key=lambda k: (needs[k], -weights[k]))
    
    shares = [0.0] * n
    remaining = guaranteed_cap
    met_count = 0
    
    for k in sorted_indices:
        needed = needs[k]
        if needed <= remaining:
            shares[k] = needed
            remaining -= needed
            met_count += 1
        else:
            shares[k] = max(0, remaining / 2)  # partial allocation
            remaining -= shares[k]
    
    # Redistribute remaining capacity to high-weight slices with good channels
    if remaining > 0:
        value_per_share = [(weights[k] * rates[k], k) for k in range(n)]
        value_per_share.sort(reverse=True)
        for _, k in value_per_share:
            if remaining > 0:
                bonus = min(remaining, 5.0)
                shares[k] += bonus
                remaining -= bonus
    
    # Third pass: smooth against churn to avoid excessive changes
    smoothing_factor = 0.4
    smoothed_shares = []
    for k in range(n):
        smoothed = prev_shares[k] * (1 - smoothing_factor) + shares[k] * smoothing_factor
        smoothed_shares.append(smoothed)
    shares = smoothed_shares
    
    # Clamp and normalize
    shares = [max(0.0, s) for s in shares]
    total_share = sum(shares)
    
    if total_share > guaranteed_cap:
        scale = guaranteed_cap / max(total_share, 0.001)
        shares = [s * scale for s in shares]
    
    # Round to avoid floating point artifacts
    shares = [round(s, 2) for s in shares]
    
    # Update state
    state.set('prev_shares', shares)
    state.set('cqi_history', cqi_history)
    
    return shares