def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller with better rate estimation and adaptive weighting.
    
    Key improvements:
    1. More accurate CQI-to-rate mapping with calibration from observed delivery
    2. Target-aware allocation: prioritize slices farthest from targets
    3. Demand-based rate capping: account for actual throughput constraints
    4. Dynamic weight adjustment based on channel quality and attainment history
    5. Better smoothing strategy: less smoothing when channels change, more when stable
    6. Score-based marginal allocation: directly optimize the scoring function
    """
    
    n = sla.n  # 4
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    targets = [sla.slices[k].target_mbps for k in range(n)]
    weights = [sla.slices[k].weight for k in range(n)]
    
    prev_shares = state.get('prev_shares', [0.0] * n)
    rate_estimates = state.get('rate_estimates', [0.5] * n)
    cqi_history = state.get('cqi_history', [[] for _ in range(n)])
    
    # Collect current observations
    latest_mbps = []
    cqis = []
    prb_shares = []
    
    for k in range(n):
        sample = history.slices[k].last()
        latest = history.slices[k].latest_mbps()
        latest_mbps.append(latest if latest is not None else 0.0)
        
        if sample is None:
            cqis.append(10.0)
            prb_shares.append(1.0)
        else:
            cqi = sample.cqi if sample.cqi is not None else 10.0
            prb = sample.prb_share if sample.prb_share is not None and sample.prb_share > 0 else 1.0
            cqis.append(cqi)
            prb_shares.append(prb)
            
            # Update CQI history for trend detection
            cqi_history[k].append(cqi)
            if len(cqi_history[k]) > 15:
                cqi_history[k].pop(0)
    
    state.set('cqi_history', cqi_history)
    
    # Refine rate estimates using observed delivery
    for k in range(n):
        prb = prb_shares[k]
        if prb > 0.1 and latest_mbps[k] > 0.1:
            # Observed rate per PRB share
            obs_rate = latest_mbps[k] / prb
            rate_estimates[k] = 0.5 * obs_rate + 0.5 * rate_estimates[k]
        else:
            # Fall back to CQI-based estimate
            cqi = cqis[k]
            cqi_normalized = max(0.0, min(1.0, (cqi - 5.0) / 10.0))
            # More aggressive curve: CQI 5->0.01, CQI 10->1.0, CQI 15->12.0
            cqi_rate = 0.01 + 11.99 * (cqi_normalized ** 2.2)
            rate_estimates[k] = 0.3 * cqi_rate + 0.7 * rate_estimates[k]
        
        rate_estimates[k] = max(0.01, min(15.0, rate_estimates[k]))
    
    state.set('rate_estimates', rate_estimates)
    
    # Calculate current attainment and shortfall
    attainments = []
    shortfalls = []  # How much Mbps short of target
    
    for k in range(n):
        mbps = latest_mbps[k]
        attain = 1.0 if mbps >= targets[k] - 0.5 else 0.0
        shortfall = max(0.0, targets[k] - mbps)
        attainments.append(attain)
        shortfalls.append(shortfall)
    
    # Allocate PRBs greedily by marginal score contribution
    shares = [0.0] * n
    
    for prb_idx in range(guaranteed_cap):
        best_idx = -1
        best_score_delta = -float('inf')
        
        for k in range(n):
            if shares[k] >= guaranteed_cap:
                continue
            
            # Delivery with current and +1 PRB allocation
            rate_k = rate_estimates[k]
            demand_k = latest_mbps[k]  # Capped by demand
            
            curr_delivery = min(demand_k, rate_k * shares[k])
            next_delivery = min(demand_k, rate_k * (shares[k] + 1.0))
            
            # Attainment change
            curr_attain = 1.0 if curr_delivery >= targets[k] - 0.5 else 0.0
            next_attain = 1.0 if next_delivery >= targets[k] - 0.5 else 0.0
            
            # Score delta from allocation to slice k
            # Main term: weight * attainment improvement
            score_delta = weights[k] * (next_attain - curr_attain)
            
            # Bonus for reducing shortfall (getting closer to target)
            if shortfalls[k] > 0:
                shortfall_reduction = max(0.0, shortfalls[k] - max(0.0, targets[k] - next_delivery))
                score_delta += 0.05 * weights[k] * min(1.0, shortfall_reduction / (targets[k] + 0.1))
            
            # Penalty if already meeting target (avoid over-allocation)
            if curr_attain >= 1.0:
                score_delta -= 0.05 * weights[k]
            
            # Channel quality bonus: allocate more to slices with better CQI
            if curr_attain < 1.0:  # Only apply if not yet meeting target
                cqi_factor = max(0.0, (cqis[k] - 5.0) / 10.0)
                score_delta += 0.008 * weights[k] * cqi_factor
            
            if score_delta > best_score_delta:
                best_score_delta = score_delta
                best_idx = k
        
        # Stop if marginal gain is negligible
        if best_idx == -1 or best_score_delta < -0.01:
            break
        
        shares[best_idx] += 1.0
    
    # Compute CQI stability (detect channel changes)
    cqi_changes = 0
    for k in range(n):
        if len(cqi_history[k]) >= 2:
            recent_cqi_var = max(cqi_history[k]) - min(cqi_history[k])
            if recent_cqi_var > 3.0:
                cqi_changes += 1
    
    # Adaptive smoothing based on stability and attainment
    avg_attain = sum(attainments) / n
    stability_score = avg_attain + (1.0 - cqi_changes / n) * 0.3
    
    # Less smoothing when channels are unstable or attainment is poor
    # More smoothing when stable and attainment is good
    alpha = 0.25 + 0.45 * stability_score  # Range: 0.25 to 0.7
    
    smoothed_shares = [
        alpha * shares[k] + (1.0 - alpha) * prev_shares[k]
        for k in range(n)
    ]
    
    # Enforce hard constraints
    smoothed_shares = [max(0.0, s) for s in smoothed_shares]
    total = sum(smoothed_shares)
    
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        smoothed_shares = [s * scale for s in smoothed_shares]
    
    state.set('prev_shares', smoothed_shares)
    
    return smoothed_shares