def controller(history, state, sla, predictors):
    """
    PRB allocation controller for 4 guaranteed slices + 1 best-effort slice.
    Maximizes weighted score based on target attainment and best-effort throughput.
    """
    
    # Initialize state tracking
    if not state.get("initialized"):
        state.set("initialized", True)
        state.set("last_shares", [0.0, 0.0, 0.0, 0.0])
        state.set("cqi_history", [[] for _ in range(4)])
        state.set("rate_history", [[] for _ in range(4)])
    
    last_shares = state.get("last_shares", [0.0, 0.0, 0.0, 0.0])
    cqi_history = state.get("cqi_history", [[] for _ in range(4)])
    rate_history = state.get("rate_history", [[] for _ in range(4)])
    
    # Extract current state from history
    current_cqi = []
    current_rates = []
    
    for k in range(4):
        sample = history.slices[k].last()
        if sample is not None:
            cqi = sample.cqi
            current_cqi.append(cqi)
            
            # Estimate rate (Mbps per PRB share) from recent delivered throughput
            if sample.prb_share > 0.01:
                delivered_mbps = history.slices[k].latest_mbps()
                rate = delivered_mbps / sample.prb_share if delivered_mbps >= 0 else 0
            else:
                rate = 0
            current_rates.append(rate)
            
            # Update histories
            if len(cqi_history[k]) > 20:
                cqi_history[k].pop(0)
            if len(rate_history[k]) > 20:
                rate_history[k].pop(0)
            cqi_history[k].append(cqi)
            rate_history[k].append(rate)
        else:
            current_cqi.append(7.5)  # Default middle CQI if no data
            current_rates.append(0.5)  # Conservative default rate
    
    state.set("cqi_history", cqi_history)
    state.set("rate_history", rate_history)
    
    # Calculate average rates per slice over recent history
    avg_rates = []
    for k in range(4):
        if len(rate_history[k]) > 0:
            valid_rates = [r for r in rate_history[k] if r > 0]
            if valid_rates:
                avg_rates.append(sum(valid_rates) / len(valid_rates))
            else:
                avg_rates.append(0.5)
        else:
            avg_rates.append(0.5)
    
    # Get targets and weights
    weights = [sla.slices[k].weight for k in range(4)]
    targets = [sla.slices[k].target_mbps for k in range(4)]
    
    # Allocate PRBs to meet targets based on current rates and CQI
    shares = [0.0] * 4
    used = 0.0
    capacity = sla.guaranteed_cap  # 70
    
    # Priority: allocate to meet targets, weighted by slice weight and CQI quality
    # Slices with better CQI can reach targets with fewer shares
    allocations = []
    for k in range(4):
        rate = max(avg_rates[k], 0.1)  # Avoid division by zero
        share_needed = targets[k] / rate
        priority = weights[k] * (current_cqi[k] / 10.0)  # Higher weight & CQI = higher priority
        allocations.append({
            'k': k,
            'share_needed': share_needed,
            'priority': priority,
            'rate': rate,
            'target': targets[k]
        })
    
    # Sort by priority (descending) to allocate to most important slices first
    allocations.sort(key=lambda x: x['priority'], reverse=True)
    
    for alloc in allocations:
        k = alloc['k']
        share_needed = min(alloc['share_needed'], capacity - used)
        shares[k] = max(0.0, share_needed)
        used += shares[k]
    
    # If total exceeds capacity, scale down proportionally
    if used > capacity:
        scale = capacity / used
        shares = [s * scale for s in shares]
    
    # Smooth transitions: blend with previous shares to reduce churn
    smoothing = 0.3
    smoothed_shares = [
        last_shares[k] * (1 - smoothing) + shares[k] * smoothing
        for k in range(4)
    ]
    
    # Clamp to valid range
    smoothed_shares = [max(0.0, s) for s in smoothed_shares]
    
    # Final scale if needed
    total = sum(smoothed_shares)
    if total > capacity:
        scale = capacity / total if total > 0 else 1.0
        smoothed_shares = [s * scale for s in smoothed_shares]
    
    smoothed_shares = [round(s, 2) for s in smoothed_shares]
    
    # Update state
    state.set("last_shares", smoothed_shares)
    
    return smoothed_shares