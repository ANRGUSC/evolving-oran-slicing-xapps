def controller(history, state, sla, predictors):
    """
    PRB allocation controller for four guaranteed slices + one best-effort slice.
    Maximizes weighted attainment of SLA targets plus best-effort throughput,
    penalizing churn.
    """
    
    # Initialize state if needed
    if state.get("initialized") is None:
        state.set("initialized", True)
        state.set("prev_shares", [0.0, 0.0, 0.0, 0.0])
        state.set("tick_count", 0)
    
    tick_count = state.get("tick_count", 0)
    state.set("tick_count", tick_count + 1)
    
    n = sla.n  # 4
    targets = [sla.slices[k].target_mbps for k in range(n)]
    weights = [sla.slices[k].weight for k in range(n)]
    pool_max = sla.pool_max  # 100
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Estimate rate_k (Mbps per PRB share) for each slice from history
    rates = []
    cqis = []
    prev_shares = state.get("prev_shares", [0.0, 0.0, 0.0, 0.0])
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is None:
            # No history: assume moderate rate (e.g., CQI 10 equivalent)
            rates.append(3.0)
            cqis.append(10.0)
        else:
            cqis.append(sample.cqi)
            # Estimate rate from last tick: if share > 0, rate ≈ delivered_Mbps / share
            if sample.prb_share > 0:
                delivered = history.slices[k].latest_mbps()
                if delivered is not None and delivered >= 0:
                    rate = delivered / sample.prb_share
                    rates.append(max(0.1, rate))  # Floor at 0.1 Mbps/share
                else:
                    rates.append(3.0)
            else:
                rates.append(3.0)
    
    # Allocate shares to meet targets if possible, weighted by importance and channel quality
    # Strategy:
    # 1. Try to meet each target with a baseline allocation proportional to target/rate and weight
    # 2. Adjust based on CQI: boost allocation for good channels, reduce for poor ones
    # 3. Stay within guaranteed_cap constraint
    
    shares = [0.0] * n
    
    # Compute minimum share needed to meet target for each slice (if rate allows)
    min_shares = []
    for k in range(n):
        if rates[k] > 0:
            min_share = targets[k] / rates[k]
        else:
            min_share = guaranteed_cap + 1  # Impossible to meet
        min_shares.append(min_share)
    
    # Check which targets can be met at all
    can_meet = [min_shares[k] <= guaranteed_cap for k in range(n)]
    
    # If all can be met within cap:
    total_min = sum(min_shares[k] if can_meet[k] else 0 for k in range(n))
    
    if total_min <= guaranteed_cap:
        # All targets can be met; allocate baseline + buffer
        for k in range(n):
            if can_meet[k]:
                shares[k] = min_shares[k]
        
        # Distribute remaining capacity proportional to weights
        remaining = guaranteed_cap - total_min
        weight_sum = sum(weights[k] for k in range(n))
        if weight_sum > 0:
            for k in range(n):
                shares[k] += remaining * (weights[k] / weight_sum)
    else:
        # Not all targets can be met; prioritize by weight and feasibility
        # Allocate to slices in order of weight, respecting rate constraints
        candidates = [(weights[k], k) for k in range(n)]
        candidates.sort(reverse=True)
        
        remaining_cap = guaranteed_cap
        for _, k in candidates:
            if can_meet[k]:
                alloc = min(min_shares[k], remaining_cap)
            else:
                # Slice k cannot meet target; allocate conservatively
                alloc = min(remaining_cap * 0.3, remaining_cap)
            shares[k] = alloc
            remaining_cap -= alloc
            if remaining_cap <= 0:
                break
    
    # Fine-tune based on CQI: slightly boost slices with good channels
    avg_cqi = sum(cqis) / len(cqis) if cqis else 10.0
    cqi_adjustment = [max(0.8, min(1.2, cqis[k] / avg_cqi)) for k in range(n)]
    
    for k in range(n):
        shares[k] *= cqi_adjustment[k]
    
    # Renormalize to stay within cap
    total_shares = sum(shares)
    if total_shares > guaranteed_cap:
        scale = guaranteed_cap / total_shares
        shares = [s * scale for s in shares]
    
    # Apply churn penalty: prefer to keep previous shares if score doesn't improve much
    prev_shares_list = state.get("prev_shares", [0.0, 0.0, 0.0, 0.0])
    
    # Smooth shares slightly to reduce churn, unless large improvement is possible
    smoothing = 0.3
    for k in range(n):
        shares[k] = smoothing * prev_shares_list[k] + (1 - smoothing) * shares[k]
    
    # Clamp to [0, guaranteed_cap] and ensure sum <= guaranteed_cap
    shares = [max(0.0, s) for s in shares]
    total_shares = sum(shares)
    if total_shares > guaranteed_cap:
        scale = guaranteed_cap / total_shares
        shares = [s * scale for s in shares]
    
    # Update state
    state.set("prev_shares", shares)
    
    return shares