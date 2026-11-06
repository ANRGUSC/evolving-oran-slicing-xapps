def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller with better attainment targeting.
    Strategy: Aggressive attainment maximization with intelligent capacity management.
    """
    
    # Initialize state tracking
    if state.get('prev_shares') is None:
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
    if state.get('tick_count') is None:
        state.set('tick_count', 0)
    if state.get('attainment_history') is None:
        state.set('attainment_history', [])
    
    prev_shares = state.get('prev_shares', [0.0, 0.0, 0.0, 0.0])
    tick_count = state.get('tick_count', 0)
    state.set('tick_count', tick_count + 1)
    
    # Extract SLA parameters
    targets = [sla.slices[k].target_mbps for k in range(4)]
    weights = [sla.slices[k].weight for k in range(4)]
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Estimate Mbps per share for each slice from history
    rates = []
    cqis = []
    latest_mbps = []
    for k in range(4):
        latest = history.slices[k].last()
        if latest is None:
            cqis.append(7.5)
            rates.append(0.5)
            latest_mbps.append(0.0)
        else:
            cqis.append(latest.cqi)
            latest_mbps.append(history.slices[k].latest_mbps())
            
            # CQI-to-rate mapping: CQI 5 → ~0.2 Mbps/share, CQI 15 → ~2.2 Mbps/share
            cqi = latest.cqi
            if cqi <= 5:
                base_rate = 0.15
            elif cqi >= 15:
                base_rate = 2.2
            else:
                # Exponential curve for better sensitivity
                base_rate = 0.15 * (2.0 ** ((cqi - 5) / 5.0))
            
            # Empirical rate from recent delivery
            estimated_rate = base_rate
            if latest.prb_share > 0.01:
                mbps = latest.dl_bytes * 8 / 1e6
                empirical_rate = mbps / latest.prb_share
                # Higher weight on empirical data (it's most recent)
                estimated_rate = 0.75 * empirical_rate + 0.25 * base_rate
            
            rates.append(max(0.1, estimated_rate))
    
    # Calculate shares needed to meet targets (with safety margin)
    shares_for_target = []
    attainable = []  # Can this target be met?
    for k in range(4):
        if rates[k] > 0.05:
            needed = targets[k] / rates[k]
        else:
            needed = guaranteed_cap + 1
        shares_for_target.append(needed)
        # A target is attainable if it needs ≤ guaranteed_cap
        attainable.append(needed <= guaranteed_cap)
    
    # Strategy: Maximize weighted attainment score
    # Score component for slice k: w_k * attain_k - 0.5 * w_k * (1 - attain_k)
    # = w_k * (1.5 * attain_k - 0.5)
    # Attainment threshold: delivered >= target - 0.5
    
    # Build priority with dual objectives:
    # 1. Protect high-weight slices (they have larger weight impact)
    # 2. Prefer slices with attainable targets (feasible allocation)
    # 3. Favor slices with better channels (more likely to maintain attainment)
    
    priority = []
    for k in range(4):
        # Base priority from weight (higher weight = more important)
        weight_priority = weights[k]
        
        # Feasibility bonus (attainable targets are easier to maintain)
        feas_mult = 1.8 if attainable[k] else 0.6
        
        # Channel quality bonus (CQI 15 = max, CQI 5 = min)
        cqi_mult = 1.0 + 0.4 * (cqis[k] - 5) / 10.0
        
        # Recent performance bonus (currently meeting target?)
        perf_mult = 1.3 if latest_mbps[k] >= targets[k] - 1.0 else 1.0
        
        combined_priority = weight_priority * feas_mult * cqi_mult * perf_mult
        priority.append((combined_priority, k))
    
    priority.sort(reverse=True)
    
    shares = [0.0] * 4
    remaining = guaranteed_cap
    
    # Phase 1: Guarantee all attainable targets
    for _, k in priority:
        if remaining < 0.1:
            break
        if attainable[k]:
            allocate = min(shares_for_target[k], remaining)
            shares[k] = allocate
            remaining -= allocate
    
    # Phase 2: For unattainable targets, allocate proportionally to weight
    # This gives best effort to high-weight slices even when target unattainable
    unattainable_indices = [k for k in range(4) if not attainable[k]]
    if unattainable_indices and remaining > 1.0:
        unatt_weights = [weights[k] for k in unattainable_indices]
        total_unatt_weight = sum(unatt_weights)
        for k in unattainable_indices:
            if remaining > 0.1:
                share_of_remaining = (weights[k] / total_unatt_weight) * remaining
                shares[k] = share_of_remaining
                remaining -= share_of_remaining
    
    # Phase 3: Aggressive reallocation to maximize attainment
    # If a high-weight slice is close to attainment but not quite there, 
    # steal from lower-weight slices if needed
    attainment = []
    for k in range(4):
        if rates[k] > 0.05:
            delivered = rates[k] * shares[k]
            att = 1.0 if delivered >= targets[k] - 0.5 else 0.0
        else:
            att = 0.0
        attainment.append(att)
    
    # Find high-weight slices not meeting attainment
    for attempt in range(2):
        unattained_high_weight = [
            (weights[k], k) for k in range(4) 
            if attainment[k] < 0.5 and weights[k] >= 0.8
        ]
        if not unattained_high_weight:
            break
        
        unattained_high_weight.sort(reverse=True)
        low_weight_with_slack = [
            (weights[k], k) for k in range(4)
            if attainment[k] > 0.5 and weights[k] < 0.9
        ]
        low_weight_with_slack.sort()
        
        # Rebalance: steal from low-weight slices with attainment
        for _, k_high in unattained_high_weight[:2]:
            for _, k_low in low_weight_with_slack:
                if shares[k_low] > shares_for_target[k_low] * 1.1:
                    transfer = min(2.0, shares[k_low] - shares_for_target[k_low])
                    shares[k_low] -= transfer
                    shares[k_high] += transfer
                    
                    # Recompute attainment
                    if rates[k_high] > 0.05:
                        delivered = rates[k_high] * shares[k_high]
                        attainment[k_high] = 1.0 if delivered >= targets[k_high] - 0.5 else 0.0
                    if rates[k_low] > 0.05:
                        delivered = rates[k_low] * shares[k_low]
                        attainment[k_low] = 1.0 if delivered >= targets[k_low] - 0.5 else 0.0
                    break
    
    # Smooth to reduce churn but keep responsiveness on big changes
    cqi_change = 0
    if state.get('prev_cqis') is not None:
        prev_cqis = state.get('prev_cqis')
        cqi_change = sum(abs(cqis[k] - prev_cqis[k]) for k in range(4))
    
    state.set('prev_cqis', cqis)
    
    # Early ticks: adapt quickly; later: smooth more to reduce churn
    if tick_count < 8:
        base_smoothing = 0.15
    else:
        base_smoothing = 0.30
    
    # Increase smoothing on stable channels; decrease on volatile
    stability_mult = min(1.0, max(0.3, 1.0 - cqi_change / 25.0))
    smoothing = base_smoothing * stability_mult
    
    smoothed = [
        (1.0 - smoothing) * shares[k] + smoothing * prev_shares[k]
        for k in range(4)
    ]
    
    # Ensure non-negative and within cap
    smoothed = [max(0, s) for s in smoothed]
    total = sum(smoothed)
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        smoothed = [s * scale for s in smoothed]
    
    state.set('prev_shares', smoothed)
    
    return smoothed