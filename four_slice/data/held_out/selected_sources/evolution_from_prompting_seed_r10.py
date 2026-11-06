def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller with enhanced rate estimation and adaptive targeting.
    Key improvements:
    - Better rate estimation using longer history and CQI correlation
    - Detect when targets are unachievable and reallocate to meeting targets elsewhere
    - Dynamic threshold for attainment vs pushing best-effort
    - Reduced churn via softer hysteresis and change thresholding
    """
    
    # Initialize state
    if not state.get('initialized'):
        state.set('initialized', True)
        state.set('prev_shares', [0.0] * 4)
        state.set('rate_ewma', [3.0] * 4)
        state.set('cqi_history', [[] for _ in range(4)])
        state.set('tick_count', 0)
    
    n_slices = sla.n
    pool_max = sla.pool_max
    guaranteed_cap = sla.guaranteed_cap
    
    tick_count = state.get('tick_count', 0)
    state.set('tick_count', tick_count + 1)
    
    # Extract per-slice telemetry
    cqis = []
    rates = []
    targets = []
    weights = []
    
    cqi_hist = state.get('cqi_history', [[] for _ in range(4)])
    
    for k in range(n_slices):
        sla_slice = sla.slices[k]
        weights.append(sla_slice.weight)
        targets.append(sla_slice.target_mbps)
        
        sample = history.slices[k].last()
        cqi = 10.0
        rate = 3.0
        
        if sample is not None:
            cqi = sample.cqi
            cqi_hist[k].append(cqi)
            if len(cqi_hist[k]) > 20:
                cqi_hist[k].pop(0)
            
            # Rate estimation: use EWMA with better filtering
            if sample.prb_share > 0.3:
                delivered_mbps = sample.dl_bytes * 8 / 1e6
                observed_rate = delivered_mbps / sample.prb_share
                # Clamp outliers
                observed_rate = max(0.1, min(observed_rate, 15.0))
                ewma_rate = state.get('rate_ewma', [3.0]*4)[k]
                # Use 0.25 alpha for smoother estimation
                ewma_rate = 0.75 * ewma_rate + 0.25 * observed_rate
                rate = ewma_rate
                
                rate_list = state.get('rate_ewma', [3.0]*4)
                rate_list[k] = ewma_rate
                state.set('rate_ewma', rate_list)
        else:
            rate = state.get('rate_ewma', [3.0]*4)[k]
        
        cqis.append(cqi)
        rates.append(rate)
    
    state.set('cqi_history', cqi_hist)
    
    # CQI quality factor with better sensitivity
    cqi_quality = []
    for k, cqi in enumerate(cqis):
        # Map [5, 15] -> [0.1, 1.0]
        q = (min(max(cqi, 5.0), 15.0) - 5.0) / 10.0
        q = max(0.08, q)  # Aggressive floor
        cqi_quality.append(q)
    
    # Compute shares needed to meet targets
    shares_for_target = []
    achievable = []
    
    for k in range(n_slices):
        rate_adj = rates[k] * cqi_quality[k]
        if rate_adj > 0.15:
            needed = targets[k] / rate_adj
            shares_for_target.append(needed)
            # Check if achievable at any reasonable allocation
            achievable.append(needed <= guaranteed_cap)
        else:
            shares_for_target.append(guaranteed_cap + 1)
            achievable.append(False)
    
    # Count how many targets can realistically be met
    achievable_count = sum(achievable)
    
    prev_shares = state.get('prev_shares', [0.0] * 4)
    
    # Main allocation strategy
    if achievable_count == 4:
        # All targets achievable: allocate to meet them
        test_shares = [shares_for_target[k] for k in range(n_slices)]
        total = sum(test_shares)
        if total > guaranteed_cap:
            # Over budget: scale down proportionally
            scale = guaranteed_cap / total
            test_shares = [s * scale for s in test_shares]
    elif achievable_count >= 2:
        # Some targets achievable: prioritize by weight * quality, meet what we can
        priority = [(weights[k] * cqi_quality[k] * (1.0 if achievable[k] else 0.3), k) 
                    for k in range(n_slices)]
        priority.sort(reverse=True)
        
        test_shares = [0.0] * 4
        remaining = guaranteed_cap
        
        for _, k in priority:
            if achievable[k]:
                needed = shares_for_target[k]
                if needed <= remaining:
                    test_shares[k] = needed
                    remaining -= needed
                else:
                    test_shares[k] = remaining
                    remaining = 0
        
        # Distribute remainder to high-priority slices
        if remaining > 0:
            for _, k in priority:
                if remaining <= 0.1:
                    break
                extra = min(remaining, 5.0)
                test_shares[k] += extra
                remaining -= extra
    else:
        # Few or no targets achievable: use CQI-weighted best-effort
        priority = [(weights[k] * cqi_quality[k], k) for k in range(n_slices)]
        priority.sort(reverse=True)
        
        test_shares = [0.0] * 4
        remaining = guaranteed_cap
        
        for _, k in priority:
            alloc = remaining * (weights[k] * cqi_quality[k] / (sum(weights[i] * cqi_quality[i] for i in range(n_slices)) + 0.001))
            test_shares[k] = max(1.0, alloc)
            remaining -= test_shares[k]
    
    test_shares = [max(0.0, s) for s in test_shares]
    total = sum(test_shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / (total + 0.001)
        test_shares = [s * scale for s in test_shares]
    
    # Apply hysteresis to reduce churn: only move if delta > threshold
    final_shares = []
    churn = 0
    hysteresis_threshold = 0.8
    
    for k in range(n_slices):
        delta = abs(test_shares[k] - prev_shares[k])
        if delta > hysteresis_threshold or prev_shares[k] == 0:
            final_shares.append(test_shares[k])
            if delta > 0.5:
                churn += 1
        else:
            # Keep previous share
            final_shares.append(prev_shares[k])
    
    # Final normalization
    final_shares = [max(0.0, s) for s in final_shares]
    total = sum(final_shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / (total + 0.001)
        final_shares = [s * scale for s in final_shares]
    
    state.set('prev_shares', final_shares)
    
    return final_shares