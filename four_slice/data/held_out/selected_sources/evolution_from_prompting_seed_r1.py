def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller for 4 guaranteed slices + 1 best-effort slice.
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
        state.set('rates', [1.0, 1.0, 1.0, 1.0])
        state.set('history_window', 10)
    
    current_rates = state.get('rates', [1.0, 1.0, 1.0, 1.0])
    prev_shares = state.get('prev_shares', [0.0, 0.0, 0.0, 0.0])
    
    # Update rate estimates from history with exponential smoothing
    for k in range(sla.n):
        sample = history.slices[k].last()
        if sample is not None and sample.prb_share > 0.1:
            delivered_mbps = history.slices[k].latest_mbps()
            if delivered_mbps is not None:
                rate_est = delivered_mbps / sample.prb_share
                # More aggressive smoothing for better responsiveness
                current_rates[k] = 0.6 * current_rates[k] + 0.4 * rate_est
            current_rates[k] = max(0.1, current_rates[k])
    
    state.set('rates', current_rates)
    
    # Get CQI values
    cqis = []
    for k in range(sla.n):
        sample = history.slices[k].last()
        cqi = sample.cqi if sample is not None else 10.0
        cqis.append(cqi)
    
    # Estimate achievable throughput per share for each slice
    achievable_rates = []
    for k in range(sla.n):
        cqi = cqis[k]
        # Better CQI mapping: floor at 5, saturate near 15
        cqi_factor = max(0.08, min(1.2, (cqi - 5.0) / 10.0))
        achievable_rate = current_rates[k] * cqi_factor
        achievable_rates.append(achievable_rate)
    
    # Compute current delivered throughput
    current_delivered = []
    for k in range(sla.n):
        delivered = achievable_rates[k] * prev_shares[k] if prev_shares[k] > 0 else 0
        current_delivered.append(delivered)
    
    # Identify which slices are meeting targets and which are not
    attainments = []
    for k in range(sla.n):
        attain = 1.0 if current_delivered[k] >= targets[k] - 0.5 else 0.0
        attainments.append(attain)
    
    # Compute benefit of adding one more share to each slice
    def marginal_benefit(k, current_share, delivered):
        """Compute marginal benefit of adding 1 more share to slice k"""
        new_delivered = achieved_rates[k] * (current_share + 1.0)
        
        # If already meeting target, benefit is nearly zero but small penalty on other slices
        if delivered >= targets[k] - 0.5:
            # Slight benefit for margin above target
            excess = delivered - (targets[k] - 0.5)
            return 0.05 * weights[k] if excess < 2.0 else 0.0
        
        # If not meeting target, benefit is the weight
        deficit = targets[k] - 0.5 - delivered
        gained = min(deficit, new_delivered - delivered)
        
        # Weighted benefit scaled by how much we gain toward target
        benefit = weights[k] * (gained / (targets[k] + 0.1))
        return benefit
    
    # Water-filling allocation: prioritize slices not meeting targets
    shares = [0.0] * sla.n
    remaining_budget = guaranteed_cap
    
    # Iterative water-filling: allocate one share at a time to the slice with highest marginal benefit
    achieved_rates = achievable_rates[:]
    
    while remaining_budget > 0.5:
        best_k = -1
        best_benefit = -1.0
        
        for k in range(sla.n):
            mb = marginal_benefit(k, shares[k], achieved_rates[k] * shares[k])
            if mb > best_benefit:
                best_benefit = mb
                best_k = k
        
        if best_k < 0 or best_benefit <= 0:
            break
        
        shares[best_k] += 1.0
        remaining_budget -= 1.0
    
    # Distribute remaining budget by weight proportional allocation
    if remaining_budget > 0.1:
        total_weight = sum(weights)
        for k in range(sla.n):
            extra_shares = remaining_budget * weights[k] / total_weight
            shares[k] += extra_shares
            remaining_budget -= extra_shares
    
    # Enforce guaranteed cap
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.1)
        shares = [s * scale for s in shares]
    
    # Smooth transitions to reduce churn
    # Use exponential smoothing with small alpha to reduce oscillation
    smoothing_alpha = 0.15
    smoothed_shares = [
        smoothing_alpha * shares[k] + (1.0 - smoothing_alpha) * prev_shares[k]
        for k in range(sla.n)
    ]
    
    # Only apply smoothing if we're not in critical deficit
    critical_deficit = sum(1 for k in range(sla.n) if current_delivered[k] < targets[k] - 5.0)
    if critical_deficit <= 1:
        shares = smoothed_shares
    
    # Clamp to non-negative
    shares = [max(0.0, s) for s in shares]
    
    # Final normalization
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.1)
        shares = [s * scale for s in shares]
    
    state.set('prev_shares', shares)
    
    return shares