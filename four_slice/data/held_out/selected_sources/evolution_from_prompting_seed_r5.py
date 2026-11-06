def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller focusing on target attainment.
    
    Key improvements:
    1. More aggressive rate estimation with history-based learning
    2. Direct focus on meeting SLA targets with tighter share calculation
    3. Multi-phase allocation: must-meet targets first, then gap-filling
    4. Reduced smoothing for faster response to channel changes
    5. Per-slice momentum to avoid oscillation while staying responsive
    """
    
    n_slices = 4
    guaranteed_cap = 70
    
    # Extract SLA parameters
    weights = [sla.slices[k].weight for k in range(n_slices)]
    targets = [sla.slices[k].target_mbps for k in range(n_slices)]
    
    # Initialize state
    if state.get('prev_shares') is None:
        state.set('prev_shares', [17.5] * n_slices)
        state.set('rate_history', [[0.8] * 10 for _ in range(n_slices)])
        state.set('cqi_history', [[10.0] * 10 for _ in range(n_slices)])
    
    prev_shares = state.get('prev_shares', [17.5] * n_slices)
    rate_history = state.get('rate_history', [[0.8] * 10 for _ in range(n_slices)])
    cqi_history = state.get('cqi_history', [[10.0] * 10 for _ in range(n_slices)])
    
    # Estimate rate (Mbps per PRB share) and CQI for each slice
    rates = []
    cqis = []
    attainments = []
    
    for k in range(n_slices):
        last_sample = history.slices[k].last()
        cqi = 10.0
        rate = 0.8
        current_mbps = 0.0
        
        if last_sample is not None:
            cqi = last_sample.cqi
            current_mbps = last_sample.dl_bytes * 8 / 1e6
            
            # Robust empirical rate calculation
            if last_sample.prb_share > 0.5:
                empirical_rate = current_mbps / last_sample.prb_share
                # Trust empirical data more when share is substantial
                rate = 0.6 * empirical_rate + 0.4 * (0.7 + 0.095 * (cqi - 10.0))
            else:
                # Estimate from CQI when share is small/zero
                rate = 0.7 + 0.095 * (cqi - 10.0)
        else:
            rate = 0.7 + 0.095 * (cqi - 10.0)
        
        # Clamp to physically reasonable bounds
        rate = max(0.12, min(rate, 1.8))
        rates.append(rate)
        cqis.append(cqi)
        
        # Update rolling history
        rate_history[k] = (rate_history[k] + [rate])[-10:]
        cqi_history[k] = (cqi_history[k] + [cqi])[-10:]
        
        # Current attainment
        if current_mbps >= targets[k] - 0.5:
            attainments.append(1.0)
        else:
            attainments.append(max(0.0, current_mbps / targets[k]))
    
    # Detect channel conditions
    cqi_mean = sum(cqis) / len(cqis)
    min_cqi = min(cqis)
    cqi_variance = sum((c - cqi_mean) ** 2 for c in cqis) / len(cqis)
    
    # Adaptive smoothing based on stability
    if min_cqi < 6.5 or cqi_variance > 3.0:
        # Very unstable: respond aggressively
        alpha = 0.05
    elif min_cqi < 8.0 or cqi_variance > 2.0:
        # Degraded conditions: respond quickly
        alpha = 0.10
    elif cqi_mean > 12.0 and cqi_variance < 1.0:
        # Excellent stable conditions: can smooth
        alpha = 0.35
    else:
        # Normal conditions
        alpha = 0.15
    
    # Core allocation algorithm
    allocations = [0.0] * n_slices
    remaining = guaranteed_cap
    
    # Phase 1: Calculate exact shares needed for each target
    shares_needed = []
    for k in range(n_slices):
        needed = targets[k] / rates[k] if rates[k] > 0 else guaranteed_cap + 1
        shares_needed.append(needed)
    
    # Phase 2: Greedy allocation by weight-normalized cost
    # Slices with higher weight per unit cost get priority
    candidates = []
    for k in range(n_slices):
        if shares_needed[k] <= guaranteed_cap:
            # Cost-effectiveness: weight per share needed
            cost_effectiveness = weights[k] / max(shares_needed[k], 1.0)
            candidates.append((cost_effectiveness, shares_needed[k], k))
    
    # Sort by cost-effectiveness (descending)
    candidates.sort(reverse=True, key=lambda x: x[0])
    
    # Allocate to feasible targets first
    for _, needed, k in candidates:
        alloc = min(needed, remaining)
        allocations[k] = alloc
        remaining -= alloc
    
    # Phase 3: Distribute remaining capacity to reduce largest shortfalls
    if remaining > 0.5:
        shortfalls = []
        for k in range(n_slices):
            current_delivery = allocations[k] * rates[k]
            gap = max(0, targets[k] - current_delivery)
            if gap > 0.1:
                # Prioritize by weight * gap (weighted shortfall intensity)
                priority = weights[k] * gap / max(rates[k], 0.1)
                shortfalls.append((priority, k, gap))
        
        shortfalls.sort(reverse=True)
        
        for _, k, gap in shortfalls:
            if remaining < 0.2:
                break
            # Allocate just enough to close gap
            share_for_gap = gap / max(rates[k], 0.1)
            increment = min(share_for_gap, remaining)
            allocations[k] += increment
            remaining -= increment
    
    # Phase 4: Allocate remaining to highest-weight unmet slices
    if remaining > 0.3:
        by_weight = []
        for k in range(n_slices):
            if allocations[k] * rates[k] < targets[k] - 0.5:
                by_weight.append((weights[k], k))
        
        by_weight.sort(reverse=True)
        
        for weight, k in by_weight:
            if remaining < 0.2:
                break
            increment = min(3.0, remaining)
            allocations[k] += increment
            remaining -= increment
    
    # Phase 5: Use any final crumbs on best-performing slices
    if remaining > 0.1:
        by_weight = [(weights[k], k) for k in range(n_slices)]
        by_weight.sort(reverse=True)
        
        for _, k in by_weight:
            if remaining < 0.1:
                break
            increment = min(1.0, remaining)
            allocations[k] += increment
            remaining -= increment
    
    # Apply smoothing with per-slice momentum
    smoothed = []
    for k in range(n_slices):
        # Reduce oscillation with adaptive smoothing
        # Increase smoothing for slices close to targets
        if attainments[k] > 0.95:
            local_alpha = alpha * 1.5
        elif attainments[k] < 0.5:
            local_alpha = alpha * 0.5
        else:
            local_alpha = alpha
        
        local_alpha = min(local_alpha, 0.4)
        smoothed_val = local_alpha * prev_shares[k] + (1 - local_alpha) * allocations[k]
        smoothed.append(smoothed_val)
    
    allocations = smoothed
    
    # Enforce hard constraints
    allocations = [max(0, a) for a in allocations]
    total = sum(allocations)
    
    if total > guaranteed_cap:
        if total > 0:
            scale = guaranteed_cap / total
            allocations = [a * scale for a in allocations]
    
    # Store for next iteration
    state.set('prev_shares', allocations[:])
    state.set('rate_history', rate_history)
    state.set('cqi_history', cqi_history)
    
    return allocations