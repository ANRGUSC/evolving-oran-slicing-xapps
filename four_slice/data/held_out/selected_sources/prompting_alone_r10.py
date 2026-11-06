def controller(history, state, sla, predictors):
    """
    PRB allocation controller for 4 guaranteed slices + 1 best-effort slice.
    Maximizes score by allocating shares based on CQI, targets, and attainment.
    """
    
    # Initialize state on first call
    if state.get("initialized") is None:
        state.set("initialized", True)
        state.set("prev_shares", [0.0, 0.0, 0.0, 0.0])
        state.set("tick_count", 0)
    
    tick_count = state.get("tick_count", 0)
    state.set("tick_count", tick_count + 1)
    
    # Extract slice data
    n = sla.n  # 4
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    pool_max = sla.pool_max  # 100
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Gather current CQI, throughput, and rate estimates
    cqis = []
    last_mbps = []
    rate_estimates = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is not None:
            cqis.append(sample.cqi)
            last_mbps.append(history.slices[k].latest_mbps())
            # Estimate rate_k (Mbps per share) from last sample
            if sample.prb_share > 0.001:
                rate_k = last_mbps[k] / sample.prb_share
            else:
                # Fallback: assume CQI maps to rate; CQI 5-15 maps to 0.5-5 Mbps/share
                rate_k = 0.3 * (sample.cqi - 5.0) + 0.5
            rate_estimates.append(max(0.1, rate_k))
        else:
            # No history: assume neutral CQI and modest rate
            cqis.append(10.0)
            last_mbps.append(0.0)
            rate_estimates.append(1.0)
    
    # Normalize CQI to [0, 1] for weighting: lower CQI → lower priority for share growth
    cqi_norm = [(cqis[k] - 5.0) / 10.0 for k in range(n)]
    cqi_norm = [max(0.0, min(1.0, c)) for c in cqi_norm]
    
    # Compute feasibility: for each slice, what share is minimally needed to meet target?
    min_shares_for_target = []
    for k in range(n):
        if rate_estimates[k] > 0.1:
            min_share = targets[k] / rate_estimates[k]
        else:
            min_share = guaranteed_cap + 1  # Impossible to meet
        min_shares_for_target.append(min_share)
    
    # Strategy: allocate shares to maximize score
    # Priority: (1) meet targets where feasible, (2) weight by w_k, (3) favor higher CQI
    
    # Start with proportional allocation based on weight and CQI
    total_weight_cqi = sum(weights[k] * cqi_norm[k] for k in range(n))
    if total_weight_cqi < 0.01:
        total_weight_cqi = sum(weights)
    
    base_shares = [guaranteed_cap * weights[k] / sum(weights) for k in range(n)]
    
    # Adjust to try to meet targets where feasible
    shares = base_shares[:]
    unmet_deficit = 0.0
    
    # First pass: allocate minimum needed to meet targets
    temp_shares = [0.0] * n
    remaining = guaranteed_cap
    
    for k in range(n):
        min_needed = min_shares_for_target[k]
        if min_needed <= remaining and min_needed <= guaranteed_cap * 0.8:
            temp_shares[k] = min(min_needed, remaining)
            remaining -= temp_shares[k]
        else:
            unmet_deficit += weights[k]
    
    # Second pass: distribute remaining capacity weighted by weight, CQI, and target coverage
    if remaining > 0.01:
        weighted_residual = [weights[k] * cqi_norm[k] for k in range(n)]
        total_residual = sum(weighted_residual)
        if total_residual > 0.01:
            for k in range(n):
                additional = remaining * (weighted_residual[k] / total_residual)
                temp_shares[k] += additional
    
    shares = temp_shares
    
    # Ensure non-negative and sum check
    shares = [max(0.0, s) for s in shares]
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        shares = [s * scale for s in shares]
    
    # Calculate churn (number of shares that changed significantly)
    prev_shares = state.get("prev_shares", [0.0] * n)
    churn = sum(1 for k in range(n) if abs(shares[k] - prev_shares[k]) > 0.5)
    
    # Penalize excessive churn by smoothing
    if churn > 2:
        alpha = 0.3  # Conservative smoothing
        shares = [alpha * shares[k] + (1 - alpha) * prev_shares[k] for k in range(n)]
        shares = [max(0.0, s) for s in shares]
        total = sum(shares)
        if total > guaranteed_cap:
            scale = guaranteed_cap / total
            shares = [s * scale for s in shares]
    
    # Store for next tick
    state.set("prev_shares", shares)
    
    return shares