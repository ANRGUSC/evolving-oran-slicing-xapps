def controller(history, state, sla, predictors):
    """
    Enhanced PRB allocation controller maximizing weighted attainment + best-effort.
    
    Key improvements:
    1. More aggressive rate estimation from CQI with better low-CQI handling
    2. Dual-phase allocation: first guarantee targets, then boost high-weight slices
    3. Dynamic churn penalty awareness: reduce smoothing when targets are at risk
    4. Better demand-aware allocation: prioritize slices closest to target achievement
    5. Explicit best-effort maximization in tie-breaking
    """
    
    n = sla.n
    guaranteed_cap = sla.guaranteed_cap  # 70
    pool_max = sla.pool_max  # 100
    
    targets = [sla.slices[k].target_mbps for k in range(n)]
    weights = [sla.slices[k].weight for k in range(n)]
    
    # Initialize state
    if not state.get("init", False):
        state.set("init", True)
        state.set("prev_shares", [0.0] * n)
        state.set("stable_cqi", [10.0] * n)
        state.set("miss_count", [0] * n)
    
    # Extract telemetry
    rates = []
    cqis = []
    actual_mbps_list = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is None:
            rates.append(0.6)
            cqis.append(10.0)
            actual_mbps_list.append(0.0)
        else:
            cqis.append(sample.cqi)
            actual_mbps = sample.dl_bytes * 8 / 1e6
            actual_mbps_list.append(actual_mbps)
            
            # Aggressive CQI-based rate estimation
            # Map CQI 5-15 to rate factor 0.05-1.5 (nonlinear)
            cqi_raw = max(5.0, min(15.0, sample.cqi))
            cqi_norm = (cqi_raw - 5.0) / 10.0  # 0 to 1
            
            # Use cubic function to emphasize mid-range performance
            cqi_factor = 0.05 + 1.45 * (cqi_norm ** 1.8)
            
            # If we have a recent observation at reasonable share, use it
            if sample.prb_share > 0.5:
                observed_rate = actual_mbps / sample.prb_share
                # Heavy weight to observation; light blend with CQI estimate
                rate = 0.75 * observed_rate + 0.25 * cqi_factor
                rates.append(max(0.05, min(2.5, rate)))
            else:
                rates.append(max(0.05, cqi_factor))
    
    # Track attainment failures to adjust strategy
    prev_miss = state.get("miss_count", [0] * n)
    miss_count = list(prev_miss)
    
    # Compute target reachability
    can_meet = []
    min_shares_for_target = []
    for k in range(n):
        if rates[k] > 0.05:
            min_share = targets[k] / rates[k]
            can_meet.append(min_share <= guaranteed_cap)
            min_shares_for_target.append(min_share)
        else:
            can_meet.append(False)
            min_shares_for_target.append(guaranteed_cap + 1.0)
    
    # PHASE 1: Allocate to meet achievable targets
    shares = [0.0] * n
    
    # Find the subset of slices we can afford to satisfy
    affordable = [(k, min_shares_for_target[k]) for k in range(n) if can_meet[k]]
    unachievable = [k for k in range(n) if not can_meet[k]]
    
    if not affordable:
        # No targets achievable: proportional weighted allocation
        total_w = sum(weights)
        for k in range(n):
            shares[k] = (weights[k] / total_w) * guaranteed_cap
    else:
        # Sort affordable by target share (greedy: satisfy cheapest targets first)
        affordable.sort(key=lambda x: x[1])
        
        cumsum = 0.0
        satisfied = []
        for k, min_sh in affordable:
            if cumsum + min_sh <= guaranteed_cap:
                shares[k] = min_sh
                cumsum += min_sh
                satisfied.append(k)
            else:
                unachievable.append(k)
        
        remaining = guaranteed_cap - cumsum
        
        # PHASE 2: Distribute remaining pool among high-weight slices and unachievable
        # Prioritize: (1) high-weight satisfied slices (boost them), (2) high-weight unsatisfied
        if remaining > 0.1:
            phase2_candidates = []
            
            # Add satisfied high-weight slices
            for k in satisfied:
                phase2_candidates.append((k, weights[k], "satisfy"))
            
            # Add unachievable slices (sorted by weight)
            for k in unachievable:
                phase2_candidates.append((k, weights[k], "unsatisfy"))
            
            phase2_candidates.sort(key=lambda x: -x[1])
            
            # Distribute remaining to boost score
            boost_per_slice = remaining / max(1, len(phase2_candidates))
            for k, w, typ in phase2_candidates:
                shares[k] += boost_per_slice
        
        # For unachievable slices, ensure minimum allocation
        for k in unachievable:
            if shares[k] < 0.5:
                shares[k] = min(3.0, remaining * 0.3 / max(1, len(unachievable)))
    
    # Normalize to guaranteed_cap
    shares = [max(0.0, s) for s in shares]
    total = sum(shares)
    if total > guaranteed_cap + 0.01:
        scale = guaranteed_cap / total
        shares = [s * scale for s in shares]
    
    # PHASE 3: Intelligent smoothing to reduce churn while protecting targets
    prev_shares = state.get("prev_shares", [0.0] * n)
    
    if any(prev_shares):
        # Detect which slices are at risk of missing targets
        at_risk = []
        for k in range(n):
            delivered = rates[k] * prev_shares[k]
            if delivered < targets[k] - 0.5:
                at_risk.append(k)
                miss_count[k] = min(10, miss_count[k] + 1)
            else:
                miss_count[k] = max(0, miss_count[k] - 1)
        
        # Reduce smoothing for at-risk slices, increase for stable ones
        smoothed = []
        for k in range(n):
            if k in at_risk:
                alpha = 0.5  # Less smoothing to fix underperformance
            elif cqis[k] < 7.0:
                alpha = 0.4  # Volatile channel: track faster
            else:
                alpha = 0.15  # Stable channel: smooth more
            
            new_share = alpha * shares[k] + (1.0 - alpha) * prev_shares[k]
            smoothed.append(new_share)
        
        shares = smoothed
    
    # Final normalization
    shares = [max(0.0, s) for s in shares]
    total = sum(shares)
    if total > guaranteed_cap + 0.01:
        scale = guaranteed_cap / total
        shares = [s * scale for s in shares]
    
    state.set("prev_shares", shares)
    state.set("miss_count", miss_count)
    
    return shares