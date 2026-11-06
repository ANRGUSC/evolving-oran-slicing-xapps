def controller(history, state, sla, predictors):
    """
    PRB allocation controller for four guaranteed slices + one best-effort slice.
    Maximizes: sum_k [ w_k * attain_k - 0.5 * w_k * (1 - attain_k) ] + 0.6 * (best_effort_Mbps / 239) - 0.02 * churn
    """
    
    # Extract SLA parameters
    n = sla.n  # 4 guaranteed slices
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    pool_max = sla.pool_max  # 100
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Get previous allocation (for churn calculation)
    prev_shares = state.get('prev_shares', [0, 0, 0, 0])
    
    # Initialize shares
    shares = [0.0] * n
    
    # Gather current channel quality and throughput data
    current_rates = []  # Mbps per PRB share for each slice
    cqis = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is not None:
            cqi = sample.cqi
            cqis.append(cqi)
            # Estimate Mbps per PRB share from CQI (higher CQI -> higher rate)
            # CQI range: 5 (bad) to 15 (good)
            # Linear approximation: rate ≈ (cqi - 5) / 10 * max_per_share + base
            # Assume at CQI=15 we get ~1.5 Mbps/PRB, at CQI=5 we get ~0.1 Mbps/PRB
            normalized_cqi = max(0, min(1, (cqi - 5) / 10))
            rate_per_share = 0.1 + normalized_cqi * 1.4
            current_rates.append(rate_per_share)
        else:
            cqis.append(10)  # neutral default
            current_rates.append(0.75)  # neutral default Mbps/PRB
    
    # Estimate feasibility: can we meet all targets?
    total_required_share = 0
    for k in range(n):
        required_share = targets[k] / max(0.01, current_rates[k])
        total_required_share += required_share
    
    # Allocation strategy:
    if total_required_share <= guaranteed_cap:
        # All targets are feasible; allocate proportional to weight and target
        # while respecting individual channel capacity
        total_weight = sum(weights)
        for k in range(n):
            # Allocate based on weight
            base_share = (weights[k] / total_weight) * guaranteed_cap
            # Adjust for channel quality: give more to slices with better channels
            quality_factor = (cqis[k] - 5) / 10.0
            quality_adjusted = base_share * (0.5 + 0.5 * (1 + quality_factor))
            shares[k] = quality_adjusted
        
        # Normalize to cap
        share_sum = sum(shares)
        if share_sum > guaranteed_cap:
            scale = guaranteed_cap / max(0.01, share_sum)
            shares = [s * scale for s in shares]
    else:
        # Not all targets feasible; prioritize based on weight and deficit
        # Allocate shares to maximize score subject to constraints
        
        # Strategy: allocate proportional to weight * (target feasibility bonus)
        allocations = []
        for k in range(n):
            max_achievable = current_rates[k] * guaranteed_cap
            if max_achievable >= targets[k]:
                # This slice can meet target
                feasibility_score = 1.0
            else:
                # Partial feasibility
                feasibility_score = max_achievable / max(0.01, targets[k])
            
            priority = weights[k] * (1.0 + feasibility_score)
            allocations.append((priority, k))
        
        allocations.sort(reverse=True, key=lambda x: x[0])
        
        # Allocate greedily in priority order
        remaining_cap = guaranteed_cap
        shares = [0.0] * n
        
        for priority, k in allocations:
            # Try to give enough to meet target
            needed = targets[k] / max(0.01, current_rates[k])
            allocation = min(needed, remaining_cap)
            shares[k] = allocation
            remaining_cap -= allocation
        
        # Distribute any remaining capacity proportionally to weights
        if remaining_cap > 0.01:
            total_weight = sum(weights)
            for k in range(n):
                shares[k] += (weights[k] / total_weight) * remaining_cap
    
    # Clamp negative values
    shares = [max(0, s) for s in shares]
    
    # Ensure sum doesn't exceed guaranteed_cap (harness will do this but let's be safe)
    share_sum = sum(shares)
    if share_sum > guaranteed_cap:
        scale = guaranteed_cap / max(0.01, share_sum)
        shares = [s * scale for s in shares]
    
    # Store for next tick's churn calculation
    state.set('prev_shares', shares)
    
    return shares