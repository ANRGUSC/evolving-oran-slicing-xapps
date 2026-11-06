def controller(history, state, sla, predictors):
    """
    Allocate PRB shares to four guaranteed slices to maximize weighted satisfaction score.
    
    Strategy:
    1. Track per-slice efficiency (Mbps per PRB share) via CQI and delivered throughput
    2. Allocate shares proportionally to (weight × efficiency) to meet targets
    3. Minimize churn by hysteresis and only adjusting when efficiency changes significantly
    4. Protect best-effort slice by not over-allocating to guaranteed slices
    """
    
    n = sla.n  # 4
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    pool_max = sla.pool_max  # 100
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Initialize state
    if state.get('prev_shares') is None:
        state.set('prev_shares', [0.0] * n)
    if state.get('efficiency') is None:
        state.set('efficiency', [1.0] * n)  # Mbps per share (baseline)
    
    prev_shares = state.get('prev_shares', [0.0] * n)
    prev_efficiency = state.get('efficiency', [1.0] * n)
    
    # Gather current data from history
    current_mbps = []
    cqis = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is not None:
            current_mbps.append(history.slices[k].latest_mbps())
            cqis.append(sample.cqi)
        else:
            current_mbps.append(0.0)
            cqis.append(7.5)  # neutral default
    
    # Estimate efficiency (Mbps per PRB share) from CQI
    # CQI range: 5..15 (deep null to excellent)
    # Map to efficiency multiplier: 5->0.1, 15->2.0
    efficiency = []
    for k in range(n):
        cqi = cqis[k]
        # Linear map: CQI 5->0.1, CQI 15->2.0
        eff = 0.1 + (cqi - 5.0) / 10.0 * 1.9
        eff = max(0.05, min(3.0, eff))  # Clamp to reasonable range
        efficiency.append(eff)
    
    # Smooth efficiency to reduce noise
    smoothing = 0.7
    smoothed_efficiency = []
    for k in range(n):
        se = smoothing * prev_efficiency[k] + (1.0 - smoothing) * efficiency[k]
        smoothed_efficiency.append(se)
    
    # Allocate shares to meet targets
    # Start with proportional allocation based on (weight * efficiency)
    # Then adjust to meet targets if possible
    
    # Calculate shares needed to meet each target
    required_shares = []
    for k in range(n):
        if smoothed_efficiency[k] > 0:
            req = targets[k] / smoothed_efficiency[k]
            required_shares.append(req)
        else:
            required_shares.append(guaranteed_cap)
    
    # Total required
    total_required = sum(required_shares)
    
    # If total required <= guaranteed_cap, allocate exactly what's needed
    if total_required <= guaranteed_cap:
        shares = required_shares[:]
    else:
        # Allocate proportionally to weight * efficiency, capped at guaranteed_cap
        weighted_eff = [weights[k] * smoothed_efficiency[k] for k in range(n)]
        total_we = sum(weighted_eff)
        
        if total_we > 0:
            shares = [w * guaranteed_cap / total_we for w in weighted_eff]
        else:
            shares = [guaranteed_cap / n] * n
    
    # Apply hysteresis to reduce churn
    # Only change shares if efficiency changed significantly or if benefit is clear
    hysteresis_threshold = 0.05  # 5% efficiency change threshold
    eff_changed = any(
        abs(smoothed_efficiency[k] - prev_efficiency[k]) / max(prev_efficiency[k], 0.1) > hysteresis_threshold
        for k in range(n)
    )
    
    if not eff_changed:
        # No major efficiency change; use previous shares if they were reasonable
        shares = prev_shares[:]
    
    # Ensure shares are non-negative and sum to at most guaranteed_cap
    shares = [max(0.0, s) for s in shares]
    total_shares = sum(shares)
    
    if total_shares > guaranteed_cap:
        scale = guaranteed_cap / total_shares
        shares = [s * scale for s in shares]
    
    # Round to reasonable precision
    shares = [round(s, 2) for s in shares]
    
    # Update state
    state.set('prev_shares', shares)
    state.set('efficiency', smoothed_efficiency)
    
    return shares