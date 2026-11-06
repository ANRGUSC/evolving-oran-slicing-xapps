def controller(history, state, sla, predictors):
    """
    PRB allocation controller for 4 guaranteed slices + 1 best-effort slice.
    Maximizes: sum_k [w_k * attain_k - 0.5 * w_k * (1 - attain_k)] + 0.6 * (be_Mbps / 239) - 0.02 * churn
    """
    
    # Extract slice parameters
    weights = [sla.slices[k].weight for k in range(sla.n)]
    targets = [sla.slices[k].target_mbps for k in range(sla.n)]
    pool_max = sla.pool_max
    guaranteed_cap = sla.guaranteed_cap
    
    # Initialize state for tracking previous allocation and rates
    if not state.get('initialized', False):
        state.set('initialized', True)
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
        state.set('rates', [1.0, 1.0, 1.0, 1.0])  # Mbps per PRB share
        state.set('demand_tick_count', [0, 0, 0, 0])  # Ticks meeting demand
    
    # Get current measurements
    current_rates = state.get('rates', [1.0, 1.0, 1.0, 1.0])
    prev_shares = state.get('prev_shares', [0.0, 0.0, 0.0, 0.0])
    
    # Update rate estimates from history
    for k in range(sla.n):
        sample = history.slices[k].last()
        if sample is not None and sample.prb_share > 0:
            # Estimate Mbps per PRB share: delivered_Mbps / prb_share
            delivered_mbps = history.slices[k].latest_mbps()
            if delivered_mbps is not None:
                rate_est = delivered_mbps / sample.prb_share
                # Smooth rate estimate
                current_rates[k] = 0.7 * current_rates[k] + 0.3 * rate_est
    
    state.set('rates', current_rates)
    
    # Get CQI values and assess slice health
    cqis = []
    for k in range(sla.n):
        sample = history.slices[k].last()
        cqi = sample.cqi if sample is not None else 10.0
        cqis.append(cqi)
    
    # Estimate achievable throughput per share for each slice
    # Lower CQI => lower rate
    achievable_rates = []
    for k in range(sla.n):
        cqi = cqis[k]
        # Map CQI (5..15) to rate multiplier (0.1 .. 1.0)
        cqi_factor = max(0.05, (cqi - 5.0) / 10.0)
        achievable_rate = current_rates[k] * cqi_factor
        achievable_rates.append(achievable_rate)
    
    # Allocate shares to maximize weighted score
    # Use greedy allocation: prioritize slices with higher marginal benefit
    shares = [0.0, 0.0, 0.0, 0.0]
    remaining_budget = guaranteed_cap
    
    # Compute priority scores for each slice based on attainment gap
    priorities = []
    for k in range(sla.n):
        delivered = achievable_rates[k] * prev_shares[k] if prev_shares[k] > 0 else 0
        gap_to_target = max(0, targets[k] - delivered)
        # Priority: how much we benefit from additional shares
        # Higher weight and higher gap = higher priority
        priority = weights[k] * (1.0 + gap_to_target / (targets[k] + 0.1))
        priorities.append((priority, k))
    
    priorities.sort(reverse=True)
    
    # Greedy allocation: allocate in priority order
    allocated = [False] * sla.n
    for _ in range(sla.n):
        best_k = -1
        best_marginal = -1.0
        
        for _, k in priorities:
            if allocated[k]:
                continue
            
            # Current throughput at current share
            current_throughput = achievable_rates[k] * shares[k]
            
            # Try adding 1 more PRB share
            if shares[k] + 1 <= remaining_budget:
                new_throughput = achievable_rates[k] * (shares[k] + 1)
                
                # Marginal benefit: approaching target matters
                if current_throughput >= targets[k] - 0.5:
                    marginal_benefit = 0.0  # Already met target
                else:
                    marginal_benefit = weights[k] * min(1.0, (new_throughput - current_throughput) / (targets[k] + 0.1))
                
                if marginal_benefit > best_marginal:
                    best_marginal = marginal_benefit
                    best_k = k
        
        if best_k >= 0 and remaining_budget > 0:
            # Allocate to fill gap to target, up to remaining budget
            needed_throughput = max(targets[best_k], achievable_rates[best_k] * shares[best_k])
            needed_shares = min(remaining_budget, 
                              int((needed_throughput - achievable_rates[best_k] * shares[best_k]) / (achievable_rates[best_k] + 0.1)) + 1)
            shares[best_k] += needed_shares
            remaining_budget -= needed_shares
            allocated[best_k] = True
        else:
            # All slices allocated or no marginal benefit
            break
    
    # Distribute remaining budget by weighted priority
    if remaining_budget > 0:
        total_weight = sum(weights)
        for k in range(sla.n):
            share_of_remaining = int(remaining_budget * weights[k] / total_weight)
            shares[k] += share_of_remaining
            remaining_budget -= share_of_remaining
        
        # Distribute any leftovers to highest weight
        if remaining_budget > 0:
            max_k = weights.index(max(weights))
            shares[max_k] += remaining_budget
    
    # Ensure shares don't exceed budget
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.1)
        shares = [s * scale for s in shares]
    
    # Track churn for scoring
    churn = sum(1 for k in range(sla.n) if abs(shares[k] - prev_shares[k]) > 0.5)
    
    state.set('prev_shares', shares)
    
    return shares
