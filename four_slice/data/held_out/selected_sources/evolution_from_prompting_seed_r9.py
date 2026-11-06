def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller for 4 guaranteed slices + 1 best-effort slice.
    Strategy:
    - Track per-slice rate estimates and demand satisfaction history
    - Use adaptive allocation that prioritizes meeting targets when feasible
    - Balance aggressive target pursuit with best-effort maximization
    - Minimize churn through hysteresis and smooth transitions
    """
    
    # Extract slice parameters
    weights = [sla.slices[k].weight for k in range(sla.n)]
    targets = [sla.slices[k].target_mbps for k in range(sla.n)]
    guaranteed_cap = sla.guaranteed_cap
    
    # Initialize persistent state
    if not state.get('initialized', False):
        state.set('initialized', True)
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
        state.set('rates', [1.5, 1.5, 1.5, 1.5])
        state.set('met_target_count', [0, 0, 0, 0])
        state.set('unmet_target_count', [0, 0, 0, 0])
        state.set('rate_history', [[], [], [], []])
    
    prev_shares = state.get('prev_shares', [0.0, 0.0, 0.0, 0.0])
    current_rates = state.get('rates', [1.5, 1.5, 1.5, 1.5])
    met_counts = state.get('met_target_count', [0, 0, 0, 0])
    unmet_counts = state.get('unmet_target_count', [0, 0, 0, 0])
    rate_histories = state.get('rate_history', [[], [], [], []])
    
    # Update rate estimates from delivered throughput
    for k in range(sla.n):
        sample = history.slices[k].last()
        if sample is not None and sample.prb_share > 0.1:
            delivered_mbps = history.slices[k].latest_mbps()
            if delivered_mbps is not None and delivered_mbps > 0:
                rate_est = delivered_mbps / sample.prb_share
                # Keep rate history for stability analysis
                rate_histories[k].append(rate_est)
                if len(rate_histories[k]) > 10:
                    rate_histories[k].pop(0)
                # Smooth exponential moving average with adaptive weight
                alpha = 0.4 if len(rate_histories[k]) > 5 else 0.5
                current_rates[k] = alpha * current_rates[k] + (1 - alpha) * rate_est
                current_rates[k] = max(0.2, min(3.0, current_rates[k]))
    
    state.set('rates', current_rates)
    state.set('rate_history', rate_histories)
    
    # Get CQI and channel quality indicators
    cqis = []
    for k in range(sla.n):
        sample = history.slices[k].last()
        cqi = sample.cqi if sample is not None else 10.0
        cqis.append(cqi)
    
    # Compute CQI-adjusted rates
    adjusted_rates = []
    for k in range(sla.n):
        cqi = cqis[k]
        # Smoother CQI-to-rate mapping
        if cqi < 7:
            cqi_factor = 0.1
        elif cqi < 9:
            cqi_factor = 0.3
        elif cqi < 11:
            cqi_factor = 0.6
        elif cqi < 13:
            cqi_factor = 0.85
        else:
            cqi_factor = 1.0
        adjusted_rates.append(current_rates[k] * cqi_factor)
    
    # Estimate minimum share needed for each slice to meet target
    min_shares_for_target = []
    for k in range(sla.n):
        if adjusted_rates[k] > 0.1:
            min_share = targets[k] / adjusted_rates[k]
            min_shares_for_target.append(min_share)
        else:
            min_shares_for_target.append(guaranteed_cap + 1)  # Unreachable
    
    # Check which slices can realistically meet targets
    achievable = [min_shares_for_target[k] <= guaranteed_cap for k in range(sla.n)]
    
    # Update satisfaction tracking
    for k in range(sla.n):
        sample = history.slices[k].last()
        if sample is not None:
            delivered = history.slices[k].latest_mbps()
            if delivered is not None:
                if delivered >= targets[k] - 0.5:
                    met_counts[k] += 1
                    unmet_counts[k] = 0
                else:
                    unmet_counts[k] += 1
                    met_counts[k] = max(0, met_counts[k] - 1)
    
    state.set('met_target_count', met_counts)
    state.set('unmet_target_count', unmet_counts)
    
    # Allocation strategy
    shares = [0.0, 0.0, 0.0, 0.0]
    
    # Phase 1: Allocate minimum shares to meet targets where achievable
    allocated_budget = 0.0
    for k in range(sla.n):
        if achievable[k]:
            alloc = min_shares_for_target[k]
            shares[k] = alloc
            allocated_budget += alloc
    
    remaining = guaranteed_cap - allocated_budget
    
    # Phase 2: Distribute remaining budget
    if remaining > 0:
        # Prioritize slices that are struggling (high unmet count) but achievable
        # and slices with high weight
        priorities = []
        for k in range(sla.n):
            if achievable[k]:
                # Weight: prioritize high-weight slices and those with recent unmet targets
                priority_score = weights[k] * (1.0 + 0.5 * min(1.0, unmet_counts[k] / 5.0))
            else:
                # Non-achievable slices get lower priority but still get some share
                priority_score = 0.3 * weights[k]
            priorities.append((priority_score, k))
        
        priorities.sort(reverse=True)
        
        # Distribute remaining budget proportionally to priorities
        total_priority = sum(p[0] for p in priorities)
        for priority, k in priorities:
            if remaining > 0 and total_priority > 0:
                allocation = (priority / total_priority) * remaining
                shares[k] += allocation
                remaining -= allocation
    
    # Clamp and normalize
    shares = [max(0.0, s) for s in shares]
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.1)
        shares = [s * scale for s in shares]
    
    # Apply hysteresis to reduce churn: don't change if benefit is marginal
    hysteresis_threshold = 1.5
    for k in range(sla.n):
        if abs(shares[k] - prev_shares[k]) < hysteresis_threshold:
            shares[k] = 0.9 * shares[k] + 0.1 * prev_shares[k]
    
    # Re-normalize after hysteresis
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.1)
        shares = [s * scale for s in shares]
    
    shares = [max(0.0, s) for s in shares]
    
    state.set('prev_shares', shares)
    
    return shares