def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller using adaptive rate estimation,
    demand-aware prioritization, and direct score optimization.
    """
    
    # Extract slice parameters
    weights = [sla.slices[k].weight for k in range(sla.n)]
    targets = [sla.slices[k].target_mbps for k in range(sla.n)]
    guaranteed_cap = sla.guaranteed_cap
    
    # Initialize state
    if not state.get('initialized', False):
        state.set('initialized', True)
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
        state.set('rates', [1.0, 1.0, 1.0, 1.0])
        state.set('rate_history', [[] for _ in range(sla.n)])
        state.set('cqi_history', [[] for _ in range(sla.n)])
    
    current_rates = state.get('rates', [1.0, 1.0, 1.0, 1.0])
    prev_shares = state.get('prev_shares', [0.0, 0.0, 0.0, 0.0])
    rate_history = state.get('rate_history', [[] for _ in range(sla.n)])
    cqi_history = state.get('cqi_history', [[] for _ in range(sla.n)])
    
    # Update rate and CQI estimates from history
    cqis = []
    for k in range(sla.n):
        sample = history.slices[k].last()
        if sample is not None:
            cqi = sample.cqi
            cqis.append(cqi)
            cqi_history[k].append(cqi)
            if len(cqi_history[k]) > 20:
                cqi_history[k].pop(0)
            
            # Update rate estimate: use actual delivered throughput
            if sample.prb_share > 0:
                delivered_mbps = history.slices[k].latest_mbps()
                if delivered_mbps is not None and delivered_mbps > 0:
                    rate_est = delivered_mbps / sample.prb_share
                    rate_history[k].append(rate_est)
                    if len(rate_history[k]) > 20:
                        rate_history[k].pop(0)
                    # Adaptive smoothing: heavier weight to recent samples
                    current_rates[k] = 0.6 * current_rates[k] + 0.4 * rate_est
        else:
            cqis.append(10.0)
    
    state.set('rates', current_rates)
    state.set('rate_history', rate_history)
    state.set('cqi_history', cqi_history)
    
    # Compute stable rate estimates using historical medians
    stable_rates = []
    for k in range(sla.n):
        if rate_history[k]:
            # Use median to reduce outlier impact
            sorted_rates = sorted(rate_history[k])
            median_rate = sorted_rates[len(sorted_rates) // 2]
            stable_rates.append(0.5 * current_rates[k] + 0.5 * median_rate)
        else:
            stable_rates.append(current_rates[k])
    
    # Compute achievable rates factoring in CQI
    achievable_rates = []
    for k in range(sla.n):
        cqi = cqis[k]
        # Map CQI (5..15) to multiplier (0.05 .. 1.0) with non-linear response
        cqi_normalized = (cqi - 5.0) / 10.0
        cqi_factor = max(0.05, min(1.0, cqi_normalized ** 1.2))
        achievable_rate = stable_rates[k] * cqi_factor
        achievable_rates.append(achievable_rate)
    
    # Direct optimization: try to find the allocation that maximizes score
    # Use a more sophisticated allocation strategy
    
    shares = [0.0, 0.0, 0.0, 0.0]
    remaining_budget = guaranteed_cap
    
    # Phase 1: Allocate minimum shares to meet targets (if possible)
    deficit_slices = []
    for k in range(sla.n):
        if achievable_rates[k] > 0:
            needed_share = (targets[k] - 0.5) / achievable_rates[k]
            deficit_slices.append((needed_share, k))
    
    deficit_slices.sort()
    
    for needed_share, k in deficit_slices:
        if needed_share > 0 and remaining_budget > 0:
            alloc = min(needed_share, remaining_budget)
            shares[k] = alloc
            remaining_budget -= alloc
    
    # Phase 2: Allocate remaining budget to maximize marginal score
    # Marginal score from an additional share to slice k:
    # w_k * (attain becomes 1 if throughput >= target - 0.5)
    # For slices already meeting target: marginal benefit is minimal
    # For slices not meeting target: high priority
    
    iteration_limit = int(remaining_budget) + 5
    iterations = 0
    
    while remaining_budget > 0.1 and iterations < iteration_limit:
        best_marginal_score = -1.0
        best_k = -1
        increment = remaining_budget / max(1, int(remaining_budget) + 1)
        
        for k in range(sla.n):
            current_throughput = achievable_rates[k] * shares[k]
            
            # Marginal benefit of adding a small share
            marginal_share = min(1.0, remaining_budget / guaranteed_cap)
            new_throughput = achievable_rates[k] * (shares[k] + marginal_share)
            
            # Compute score contribution change
            current_attain = 1.0 if current_throughput >= targets[k] - 0.5 else 0.0
            new_attain = 1.0 if new_throughput >= targets[k] - 0.5 else 0.0
            
            current_score_k = weights[k] * current_attain - 0.5 * weights[k] * (1 - current_attain)
            new_score_k = weights[k] * new_attain - 0.5 * weights[k] * (1 - new_attain)
            marginal_score = new_score_k - current_score_k
            
            # Prefer slices further from target with higher weight
            if new_attain < 1.0:
                gap = targets[k] - current_throughput
                marginal_score += 0.1 * weights[k] * (gap / (targets[k] + 1.0))
            
            if marginal_score > best_marginal_score:
                best_marginal_score = marginal_score
                best_k = k
        
        if best_k >= 0 and best_marginal_score > -0.9:
            alloc = min(2.0, remaining_budget)
            shares[best_k] += alloc
            remaining_budget -= alloc
        else:
            # Distribute remaining evenly by weight if no clear winner
            total_weight = sum(weights)
            for k in range(sla.n):
                alloc = remaining_budget * weights[k] / total_weight / sla.n
                shares[k] += alloc
            remaining_budget = 0
        
        iterations += 1
    
    # Ensure shares respect budget
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.1)
        shares = [s * scale for s in shares]
    
    state.set('prev_shares', shares)
    
    return shares