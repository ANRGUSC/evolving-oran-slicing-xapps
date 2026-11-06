def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller with better attainment and stability.
    Key improvements:
    1. More accurate rate estimation using CQI-to-rate mapping
    2. Adaptive allocation based on attainment status (not just required shares)
    3. Demand-aware allocation to avoid over-provisioning
    4. Better prioritization that considers both weight and attainment gaps
    5. Reduced churn through selective smoothing
    """
    
    # Extract SLA parameters
    weights = [sla.slices[k].weight for k in range(sla.n)]
    targets = [sla.slices[k].target_mbps for k in range(sla.n)]
    guaranteed_cap = sla.guaranteed_cap
    n_slices = sla.n
    
    # Initialize state
    if state.get('prev_shares') is None:
        state.set('prev_shares', [guaranteed_cap / n_slices] * n_slices)
    if state.get('rate_estimates') is None:
        state.set('rate_estimates', [2.0] * n_slices)
    if state.get('attain_history') is None:
        state.set('attain_history', [[] for _ in range(n_slices)])
    
    prev_shares = state.get('prev_shares', [guaranteed_cap / n_slices] * n_slices)
    rate_estimates = state.get('rate_estimates', [2.0] * n_slices)
    attain_history = state.get('attain_history', [[] for _ in range(n_slices)])
    
    # Collect current data and update rate estimates
    current_cqis = []
    current_delivered = []
    
    for k in range(n_slices):
        sample = history.slices[k].last()
        if sample is not None:
            cqi = sample.cqi
            current_cqis.append(cqi)
            delivered_mbps = sample.dl_bytes * 8 / 1e6
            current_delivered.append(delivered_mbps)
            
            # Update rate estimate: Mbps per unit share
            if sample.prb_share > 0.05:
                estimated_rate = delivered_mbps / sample.prb_share
                # More conservative EMA to smooth out noise
                rate_estimates[k] = 0.7 * rate_estimates[k] + 0.3 * max(0.1, estimated_rate)
            else:
                # CQI-based fallback: map CQI [5, 15] to rate [0.3, 4.5]
                cqi_normalized = max(0, min(1, (cqi - 5) / 10))
                # Nonlinear mapping: better channels get disproportionately higher rates
                estimated_rate = 0.3 + 4.2 * (cqi_normalized ** 1.5)
                rate_estimates[k] = 0.7 * rate_estimates[k] + 0.3 * estimated_rate
        else:
            current_cqis.append(10.0)
            current_delivered.append(0.0)
    
    # Compute attainment for each slice (for priority weighting)
    attainments = []
    for k in range(n_slices):
        if current_delivered[k] >= targets[k] - 0.5:
            attainments.append(1.0)
        else:
            attainments.append(0.0)
    
    # Track attainment history for trend detection
    for k in range(n_slices):
        attain_history[k].append(attainments[k])
        if len(attain_history[k]) > 10:
            attain_history[k].pop(0)
    
    # Compute recent attainment trends
    attain_trends = []
    for k in range(n_slices):
        if len(attain_history[k]) >= 3:
            trend = sum(attain_history[k][-3:]) / 3.0
        else:
            trend = attainments[k]
        attain_trends.append(trend)
    
    # Calculate required shares with some headroom for variations
    required_shares = []
    for k in range(n_slices):
        rate = max(0.05, rate_estimates[k])
        # Add small headroom (1.1x) to account for channel variations
        required = (targets[k] * 1.05) / rate
        required_shares.append(required)
    
    total_required = sum(required_shares)
    shares = [0.0] * n_slices
    
    if total_required <= guaranteed_cap:
        # All targets can be met with headroom
        shares = required_shares[:]
    else:
        # Need to prioritize: use composite priority = weight * (1 - attain_trend) / required_share
        # This gives priority to:
        # - Slices with high weight
        # - Slices not currently being met (low attain_trend)
        # - Slices that don't require much share
        
        priorities = []
        for k in range(n_slices):
            # High weight, low current attainment, and low requirement = high priority
            gap = max(0.1, 1.0 - attain_trends[k])  # 0.1 to 1.0
            priority = weights[k] * gap / max(0.1, required_shares[k])
            priorities.append((k, priority, required_shares[k]))
        
        # Sort by priority (descending)
        priorities.sort(key=lambda x: -x[1])
        
        # Two-pass allocation
        # Pass 1: Try to allocate to meet all targets
        remaining = guaranteed_cap
        allocation_order = [k for k, _, _ in priorities]
        
        for k in allocation_order:
            required = required_shares[k]
            if remaining >= required:
                shares[k] = required
                remaining -= required
        
        # Pass 2: Distribute remaining among under-attaining slices
        # Prioritize those with highest weight and lowest current attainment
        under_attaining = [(k, weights[k], attain_trends[k]) for k in range(n_slices) if attain_trends[k] < 0.9]
        
        if remaining > 0 and under_attaining:
            # Sort by weight (descending)
            under_attaining.sort(key=lambda x: -x[1])
            for k, w, _ in under_attaining:
                if remaining > 0.1:
                    bonus = remaining * (w / sum(weights[i] for i, _, _ in under_attaining))
                    shares[k] += bonus
                    remaining -= bonus
        
        # If still remaining, distribute to all slices proportionally to weight
        if remaining > 0.5:
            total_weight = sum(weights)
            for k in range(n_slices):
                shares[k] += remaining * weights[k] / total_weight
    
    # Normalize to guaranteed_cap
    total = sum(shares)
    if total > guaranteed_cap * 1.01:
        scale = guaranteed_cap / total
        shares = [s * scale for s in shares]
    
    # Selective smoothing: smooth more when stable, less when changing
    # Reduce churn by applying smoothing only to shares that are relatively stable
    smoothing = 0.15  # Very low to allow rapid adaptation
    final_shares = []
    for k in range(n_slices):
        # Increase smoothing for slices already at target
        if attain_trends[k] > 0.8:
            smooth_factor = 0.25  # More smoothing when meeting target
        else:
            smooth_factor = 0.10  # Less smoothing when not meeting target
        
        final_share = smooth_factor * prev_shares[k] + (1 - smooth_factor) * shares[k]
        final_shares.append(final_share)
    
    # Final clamp and normalize
    final_shares = [max(0.0, s) for s in final_shares]
    total = sum(final_shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        final_shares = [s * scale for s in final_shares]
    
    # Ensure minimum allocation to slices with high weight
    min_share = 0.5
    for k in range(n_slices):
        if final_shares[k] < min_share and weights[k] > 0.7:
            final_shares[k] = min_share
    
    total = sum(final_shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        final_shares = [s * scale for s in final_shares]
    
    # Update state
    state.set('prev_shares', final_shares)
    state.set('rate_estimates', rate_estimates)
    state.set('attain_history', attain_history)
    
    return final_shares