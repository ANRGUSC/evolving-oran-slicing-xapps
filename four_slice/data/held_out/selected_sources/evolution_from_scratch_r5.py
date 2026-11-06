def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller for 4 guaranteed slices + 1 best-effort slice.
    
    Strategy:
    1. Estimate each slice's rate (Mbps/share) from recent history with robust smoothing
    2. Detect when targets are unattainable and gracefully degrade
    3. Use weighted proportional-fair with target-aware prioritization
    4. Minimize churn via adaptive hysteresis based on rate stability
    5. Favor meeting targets over maximizing best-effort when targets are close
    """
    
    n = sla.n
    pool_max = sla.pool_max
    guaranteed_cap = sla.guaranteed_cap
    
    # Get SLA parameters
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    
    # Initialize state tracking
    if state.get('prev_shares') is None:
        state.set('prev_shares', [guaranteed_cap / n] * n)
    if state.get('rate_estimates') is None:
        state.set('rate_estimates', [1.0] * n)
    if state.get('rate_history') is None:
        state.set('rate_history', [[] for _ in range(n)])
    if state.get('tick') is None:
        state.set('tick', 0)
    
    prev_shares = state.get('prev_shares')
    rate_estimates = state.get('rate_estimates')
    rate_history = state.get('rate_history')
    
    # Update rate estimates from history with robust averaging
    for k in range(n):
        samples = history.slices[k]
        if samples.last() is not None:
            recent_mbps = samples.latest_mbps()
            recent_share = samples.last().prb_share
            cqi = samples.last().cqi
            
            # Estimate Mbps per share, with CQI-aware adjustment
            if recent_share > 0.1:
                estimated_rate = recent_mbps / recent_share
                # Cap estimate based on CQI (higher CQI = more capacity)
                cqi_factor = max(0.1, (cqi - 5.0) / 10.0)  # Normalize CQI 5-15 to 0-1
                estimated_rate = estimated_rate * (0.5 + 0.5 * cqi_factor)
                
                rate_history[k].append(estimated_rate)
                if len(rate_history[k]) > 10:
                    rate_history[k].pop(0)
                
                # Use median for robustness against outliers
                sorted_hist = sorted(rate_history[k])
                median_rate = sorted_hist[len(sorted_hist) // 2]
                
                # Smooth update with adaptive alpha based on rate stability
                rate_variance = max(0.01, sum((r - median_rate) ** 2 for r in rate_history[k]) / len(rate_history[k]))
                alpha = 0.2 if rate_variance < 0.5 else 0.4
                
                rate_estimates[k] = alpha * estimated_rate + (1 - alpha) * rate_estimates[k]
                rate_estimates[k] = max(0.1, rate_estimates[k])
    
    state.set('rate_estimates', rate_estimates)
    state.set('rate_history', rate_history)
    
    # Calculate minimum shares needed to meet each target
    min_shares = []
    for k in range(n):
        if rate_estimates[k] > 0.01:
            min_share = targets[k] / rate_estimates[k]
        else:
            min_share = guaranteed_cap + 1  # Impossible to meet
        min_shares.append(min_share)
    
    # Count how many targets can be met
    feasible = sum(1 for ms in min_shares if 0 < ms <= guaranteed_cap)
    
    shares = [0.0] * n
    
    if sum(min_shares) <= guaranteed_cap:
        # All targets attainable
        for k in range(n):
            shares[k] = min_shares[k]
        remaining = guaranteed_cap - sum(shares)
        
        # Distribute remainder proportionally by weight
        total_weight = sum(weights)
        if total_weight > 0:
            for k in range(n):
                shares[k] += remaining * (weights[k] / total_weight)
    else:
        # Not all targets attainable; prioritize by weight and target difficulty
        # Slices with easier targets (lower min_share) and higher weight get priority
        
        priority = []
        for k in range(n):
            if min_shares[k] <= 0 or min_shares[k] > guaranteed_cap:
                # Unattainable; deprioritize
                priority.append(weights[k] * 0.1)
            else:
                # Attainable; give high priority
                # Weight by how close to target and absolute weight
                priority.append(weights[k] * (1.0 + 2.0 / (1.0 + min_shares[k])))
        
        total_priority = sum(priority)
        if total_priority > 0:
            for k in range(n):
                shares[k] = (priority[k] / total_priority) * guaranteed_cap
        else:
            for k in range(n):
                shares[k] = guaranteed_cap / n
    
    # Adaptive hysteresis: reduce churn by smoothing small changes
    # Use rate stability to adjust hysteresis threshold
    avg_rate_variance = 0
    for k in range(n):
        if rate_history[k]:
            sorted_hist = sorted(rate_history[k])
            median = sorted_hist[len(sorted_hist) // 2]
            var = sum((r - median) ** 2 for r in rate_history[k]) / len(rate_history[k])
            avg_rate_variance += var / n
    
    hysteresis_threshold = max(0.5, min(2.5, 1.5 * (1.0 + avg_rate_variance)))
    
    for k in range(n):
        if abs(shares[k] - prev_shares[k]) < hysteresis_threshold:
            shares[k] = prev_shares[k]
    
    # Clamp and normalize
    shares = [max(0.0, s) for s in shares]
    total = sum(shares)
    
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        shares = [s * scale for s in shares]
    
    # Update state
    state.set('prev_shares', shares[:])
    state.set('tick', state.get('tick', 0) + 1)
    
    return shares