def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller for guaranteed slices.
    
    Strategy:
    1. Estimate each slice's rate (Mbps per share) from CQI and recent delivery.
    2. Compute minimum shares needed to meet targets.
    3. Allocate greedily by weighted efficiency: prioritize high-weight, high-efficiency slices.
    4. Use aggressive target pursuit with stability damping only when churn cost is high.
    5. Balance remaining capacity to maximize best-effort throughput while maintaining SLAs.
    """
    
    n = sla.n  # 4
    targets = [sla.slices[k].target_mbps for k in range(n)]
    weights = [sla.slices[k].weight for k in range(n)]
    pool_max = sla.pool_max  # 100
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Get previous allocation and state
    prev_shares = state.get('prev_shares', [0.0] * n)
    tick_count = state.get('tick_count', 0)
    state.set('tick_count', tick_count + 1)
    
    # Collect current observations
    current_cqis = []
    current_mbps = []
    current_shares = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is not None:
            current_cqis.append(sample.cqi if hasattr(sample, 'cqi') else 10.0)
            current_mbps.append(history.slices[k].latest_mbps() or 0.0)
            current_shares.append(sample.prb_share if hasattr(sample, 'prb_share') else 0.0)
        else:
            current_cqis.append(10.0)
            current_mbps.append(0.0)
            current_shares.append(0.0)
    
    # Estimate rate_k (Mbps per PRB share) using both CQI and observed delivery
    rates = []
    for k in range(n):
        # Use recent delivery if available, else estimate from CQI
        if current_shares[k] > 0.5:
            # Recent observation: actual Mbps per share
            rate_k = current_mbps[k] / current_shares[k]
        else:
            # CQI-based estimation: CQI 5..15 maps to 0.2..12 Mbps/share
            cqi = current_cqis[k]
            rate_k = max(0.2, (cqi - 5.0) / 10.0 * 11.8 + 0.2)
        
        # Apply smoothing to avoid noise
        prev_rate = state.get(f'rate_{k}', rate_k)
        smoothed_rate = 0.7 * rate_k + 0.3 * prev_rate
        rates.append(smoothed_rate)
        state.set(f'rate_{k}', smoothed_rate)
    
    # Compute minimum and ideal shares for each slice
    min_shares = []
    ideal_shares = []
    for k in range(n):
        if rates[k] > 0.05:
            min_share = targets[k] / rates[k]
            # Ideal share: allow some headroom (20%) for demand spikes
            ideal_share = min_share * 1.2
        else:
            min_share = guaranteed_cap + 1  # Unachievable
            ideal_share = guaranteed_cap + 1
        min_shares.append(min_share)
        ideal_shares.append(ideal_share)
    
    # Phase 1: Compute weighted efficiency scores
    # Higher weight + lower minimum share = higher priority
    efficiencies = []
    for k in range(n):
        if min_shares[k] < guaranteed_cap:
            # Efficiency: weight / min_share (normalized)
            eff = weights[k] / max(min_shares[k], 1.0)
        else:
            eff = 0.0
        efficiencies.append(eff)
    
    # Phase 2: Greedy allocation by efficiency
    # Sort by efficiency (descending), then by weight (descending)
    priority_order = sorted(range(n), 
                           key=lambda k: (-efficiencies[k], -weights[k]))
    
    shares = [0.0] * n
    remaining = guaranteed_cap
    
    # First pass: allocate to meet minimum targets for slices in priority order
    for k in priority_order:
        if min_shares[k] <= remaining and min_shares[k] < guaranteed_cap:
            shares[k] = min_shares[k]
            remaining -= shares[k]
    
    # Second pass: allocate remaining capacity
    # Prefer slices close to their ideal (give more to those with better channel)
    for k in priority_order:
        if remaining > 0.1 and shares[k] > 0:
            # Try to reach closer to ideal share
            can_add = min(ideal_shares[k] - shares[k], remaining)
            if can_add > 0.1:
                shares[k] += can_add
                remaining -= can_add
    
    # Third pass: distribute any leftover to best performers
    if remaining > 0.1:
        for k in priority_order:
            if remaining > 0.1:
                add = min(remaining, 5.0)  # Add in small increments
                shares[k] += add
                remaining -= add
    
    # Phase 3: Stability and churn management
    # Calculate current attainment and potential churn
    current_attainments = []
    for k in range(n):
        delivered = min(current_mbps[k], rates[k] * shares[k]) if shares[k] > 0 else 0.0
        attain = 1.0 if delivered >= targets[k] - 0.5 else 0.0
        current_attainments.append(attain)
    
    attainment_sum = sum(current_attainments)
    
    # Calculate churn cost
    churn_count = sum(1 for k in range(n) if abs(shares[k] - prev_shares[k]) > 0.5)
    
    # Adaptive damping: if we have high attainment and low churn risk, be more conservative
    # If attainment is low, be more aggressive in finding better allocations
    if attainment_sum >= 3.0 and churn_count <= 1:
        # High attainment + low churn: dampen changes
        damping = 0.4
        shares = [(1.0 - damping) * shares[k] + damping * prev_shares[k] for k in range(n)]
    elif attainment_sum < 2.0:
        # Low attainment: be more aggressive (less damping)
        damping = 0.1
        shares = [(1.0 - damping) * shares[k] + damping * prev_shares[k] for k in range(n)]
    else:
        # Medium attainment: moderate damping
        damping = 0.2
        shares = [(1.0 - damping) * shares[k] + damping * prev_shares[k] for k in range(n)]
    
    # Phase 4: Final normalization
    shares = [max(0.0, s) for s in shares]
    total = sum(shares)
    if total > guaranteed_cap:
        shares = [s * guaranteed_cap / total for s in shares]
    
    # Update state
    state.set('prev_shares', shares)
    
    return shares