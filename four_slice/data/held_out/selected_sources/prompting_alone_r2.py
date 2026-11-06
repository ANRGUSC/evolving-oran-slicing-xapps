def controller(history, state, sla, predictors):
    """
    Controller for PRB allocation to four guaranteed network slices.
    
    Strategy:
    1. Estimate each slice's current rate (Mbps per PRB share) from recent CQI and throughput
    2. Allocate shares proportionally to weights, adjusted for achievability of targets
    3. Use a water-filling approach: allocate more to slices that can achieve targets with less share
    4. Smooth allocation changes to minimize churn penalty
    5. Reserve minimum for best-effort slice
    """
    
    # Extract SLA parameters
    n = sla.n  # 4
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    pool_max = sla.pool_max  # 100
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Initialize state
    if state.get("prev_shares") is None:
        state.set("prev_shares", [0.0] * n)
    
    prev_shares = state.get("prev_shares", [0.0] * n)
    
    # Estimate rate per share for each slice
    rates = []
    for k in range(n):
        hist = history.slices[k]
        last = hist.last()
        
        if last is None:
            # No history; assume moderate rate
            rates.append(2.0)
        else:
            cqi = last.cqi
            last_mbps = hist.latest_mbps()
            last_share = last.prb_share
            
            # Estimate rate from recent throughput and share
            if last_share > 0.1:
                estimated_rate = last_mbps / last_share
            else:
                # Fallback: use CQI to estimate (CQI 5..15 maps to rate 0.5..4.0 Mbps/share)
                estimated_rate = 0.5 + (cqi - 5.0) / 10.0 * 3.5
            
            # Smooth estimate with exponential moving average
            prev_rate = state.get(f"rate_{k}", estimated_rate)
            smoothed_rate = 0.7 * prev_rate + 0.3 * estimated_rate
            state.set(f"rate_{k}", smoothed_rate)
            rates.append(max(0.1, smoothed_rate))
    
    # Calculate minimum shares needed to meet targets
    min_shares_for_target = []
    for k in range(n):
        if rates[k] > 0.01:
            min_share = targets[k] / rates[k]
        else:
            min_share = guaranteed_cap + 1  # Impossible to meet
        min_shares_for_target.append(min_share)
    
    # Water-filling allocation:
    # Allocate inversely proportional to the share needed per unit weight
    # Prioritize slices where we can achieve targets efficiently
    
    allocation = [0.0] * n
    remaining_cap = guaranteed_cap
    
    # Calculate efficiency: weight / min_share_needed
    # Higher efficiency = more important to allocate to
    efficiencies = []
    for k in range(n):
        min_sh = min_shares_for_target[k]
        if min_sh < guaranteed_cap:
            eff = weights[k] / min_sh
        else:
            eff = 0.0
        efficiencies.append(eff)
    
    # Phase 1: Allocate to meet all targets that are feasible
    sorted_indices = sorted(range(n), key=lambda k: -efficiencies[k])
    
    for k in sorted_indices:
        min_sh = min_shares_for_target[k]
        if min_sh <= remaining_cap:
            allocation[k] = min(min_sh, remaining_cap)
            remaining_cap -= allocation[k]
    
    # Phase 2: Distribute remaining capacity proportionally by weight
    total_weight = sum(weights)
    if remaining_cap > 0.01 and total_weight > 0:
        for k in range(n):
            extra = (weights[k] / total_weight) * remaining_cap
            allocation[k] += extra
    
    # Phase 3: Smooth with previous allocation to reduce churn
    smoothed = []
    for k in range(n):
        prev_share = prev_shares[k]
        new_share = allocation[k]
        # Apply smoothing: keep some of previous allocation
        smoothed_share = 0.6 * new_share + 0.4 * prev_share
        smoothed.append(smoothed_share)
    
    allocation = smoothed
    
    # Ensure non-negative and sum <= guaranteed_cap
    allocation = [max(0.0, a) for a in allocation]
    total_allocation = sum(allocation)
    
    if total_allocation > guaranteed_cap:
        scale_factor = guaranteed_cap / total_allocation
        allocation = [a * scale_factor for a in allocation]
    
    # Round to reasonable precision and store
    allocation = [round(a, 2) for a in allocation]
    state.set("prev_shares", allocation)
    
    return allocation