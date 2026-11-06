def controller(history, state, sla, predictors):
    """
    PRB allocation controller for four guaranteed slices and one best-effort slice.
    Maximizes score = sum_k [w_k * attain_k - 0.5 * w_k * (1 - attain_k)] + 0.6 * (be_mbps / 239) - 0.02 * churn
    where attain_k = 1 if delivered >= target - 0.5 else 0.
    """
    
    n = sla.n  # 4
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Extract current slice metrics
    targets = [sla.slices[k].target_mbps for k in range(n)]
    weights = [sla.slices[k].weight for k in range(n)]
    
    # Get current CQI and recent throughput per share for each slice
    cqis = []
    recent_mbps = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is not None:
            cqis.append(sample.cqi)
            recent_mbps.append(history.slices[k].latest_mbps())
        else:
            cqis.append(10.0)  # Default middle CQI
            recent_mbps.append(0.0)
    
    # Estimate throughput per share for each slice based on CQI
    # CQI ranges 5..15, assume roughly linear relationship to rate per share
    # At CQI=5, rate ≈ 0.1 Mbps/share; at CQI=15, rate ≈ 2.0 Mbps/share
    rates_per_share = []
    for k in range(n):
        cqi = cqis[k]
        # Linear interpolation: cqi 5->0.1, cqi 15->2.0
        rate = 0.1 + (cqi - 5.0) * (2.0 - 0.1) / (15.0 - 5.0)
        rate = max(0.05, min(2.5, rate))  # Clamp to reasonable bounds
        rates_per_share.append(rate)
    
    # Retrieve previous allocation if it exists
    prev_shares = state.get('prev_shares', None)
    if prev_shares is None:
        prev_shares = [0.0] * n
    
    # Greedy allocation strategy:
    # 1. For each slice, compute minimum share needed to meet target
    # 2. Allocate greedily by weight, prioritizing slices that can't meet target
    
    min_shares_for_target = []
    for k in range(n):
        if rates_per_share[k] > 0.01:
            min_share = targets[k] / rates_per_share[k]
        else:
            min_share = guaranteed_cap + 1  # Unreachable
        min_shares_for_target.append(min_share)
    
    # Try to meet all targets within guaranteed_cap
    total_min = sum(min_shares_for_target)
    
    if total_min <= guaranteed_cap:
        # All targets can be met
        new_shares = min_shares_for_target[:]
    else:
        # Cannot meet all targets; allocate proportionally by weight to those with unmet targets
        # Start with zero allocation
        new_shares = [0.0] * n
        remaining_cap = guaranteed_cap
        
        # First pass: allocate to meet targets for slices where it's feasible and prioritize by weight
        allocations = [(weights[k], min_shares_for_target[k], k) for k in range(n)]
        allocations.sort(reverse=True)  # Sort by weight descending
        
        for w, min_share, k in allocations:
            if min_share <= remaining_cap:
                new_shares[k] = min_share
                remaining_cap -= min_share
        
        # Second pass: distribute remaining capacity proportionally by weight among unmet slices
        unmet_indices = [k for k in range(n) if new_shares[k] < min_shares_for_target[k]]
        unmet_weights = [weights[k] for k in unmet_indices]
        
        if unmet_indices and sum(unmet_weights) > 0.01:
            weight_sum = sum(unmet_weights)
            for k in unmet_indices:
                additional = remaining_cap * (weights[k] / weight_sum)
                new_shares[k] += additional
            remaining_cap = 0.0
        elif remaining_cap > 0.01:
            # Distribute to all slices proportionally by weight
            weight_sum = sum(weights)
            for k in range(n):
                new_shares[k] += remaining_cap * (weights[k] / weight_sum)
    
    # Ensure shares are non-negative and sum to at most guaranteed_cap
    new_shares = [max(0.0, s) for s in new_shares]
    total = sum(new_shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        new_shares = [s * scale for s in new_shares]
    
    # Calculate churn (number of shares that changed significantly)
    churn = 0
    if prev_shares is not None:
        for k in range(n):
            if abs(new_shares[k] - prev_shares[k]) > 0.5:
                churn += 1
    
    # Store current shares for next tick
    state.set('prev_shares', new_shares[:])
    
    return new_shares