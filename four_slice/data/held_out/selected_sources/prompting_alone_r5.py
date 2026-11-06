def controller(history, state, sla, predictors):
    """
    PRB allocation controller for 4 guaranteed slices + 1 best-effort slice.
    
    Strategy:
    1. Estimate each slice's throughput per PRB share based on CQI
    2. Allocate shares to maximize weighted attainment of targets
    3. Prefer stability (minimize churn) when allocations are similar in value
    4. Ensure best-effort gets its fair share
    """
    
    # Initialize state tracking
    if not state.get('prev_shares'):
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
    
    n = sla.n  # 4
    targets = [sla.slices[k].target_mbps for k in range(n)]
    weights = [sla.slices[k].weight for k in range(n)]
    pool_max = sla.guaranteed_cap  # 70
    
    # Extract latest metrics for each slice
    mbps_per_share = []
    cqis = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is None:
            # No data yet; assume neutral CQI and moderate throughput
            mbps_per_share.append(1.0)
            cqis.append(10.0)
        else:
            cqi = sample.cqi
            cqis.append(cqi)
            
            # Estimate Mbps per share from CQI
            # CQI range: 5 (bad) to 15 (good)
            # At CQI 5: ~0.1 Mbps/share
            # At CQI 15: ~4.0 Mbps/share
            # Use exponential-like curve: mbps = 0.1 * 2^((cqi - 5) / 2.5)
            if cqi < 5.0:
                cqi = 5.0
            if cqi > 15.0:
                cqi = 15.0
            mbps = 0.1 * (2.0 ** ((cqi - 5.0) / 2.5))
            mbps_per_share.append(mbps)
    
    # Compute minimum shares needed to meet each target
    min_shares = []
    for k in range(n):
        rate = mbps_per_share[k]
        if rate > 0.01:
            min_share = targets[k] / rate
        else:
            min_share = pool_max + 1  # Impossible to meet
        min_shares.append(min_share)
    
    # Check if all targets can be met simultaneously
    total_min = sum(min_shares)
    
    if total_min <= pool_max:
        # All targets can be met; allocate minimum + extra proportionally by weight
        shares = list(min_shares)
        surplus = pool_max - total_min
        
        # Distribute surplus proportionally to weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            for k in range(n):
                extra = (weights[k] / weight_sum) * surplus
                shares[k] += extra
    else:
        # Cannot meet all targets; allocate greedily by weighted efficiency
        # Use water-filling: allocate to slices with highest weighted marginal value
        
        shares = [0.0] * n
        remaining = pool_max
        
        # Sort slices by (weight / min_share_needed), descending
        # Prioritize high-weight, achievable targets
        priorities = []
        for k in range(n):
            if min_shares[k] > 0 and min_shares[k] <= pool_max:
                priority = weights[k] / min_shares[k]
            else:
                priority = 0.0 if min_shares[k] > pool_max else weights[k]
            priorities.append((priority, k))
        
        priorities.sort(reverse=True, key=lambda x: x[0])
        
        # Allocate in priority order up to target, then distribute remainder
        for _, k in priorities:
            if remaining <= 0:
                break
            alloc = min(min_shares[k], remaining)
            shares[k] = alloc
            remaining -= alloc
        
        # Distribute any leftover to high-weight slices
        if remaining > 0:
            weight_sum = sum(weights)
            if weight_sum > 0:
                for k in range(n):
                    extra = (weights[k] / weight_sum) * remaining
                    shares[k] += extra
    
    # Apply stability bias: if previous allocation is close, prefer it to reduce churn
    prev_shares = state.get('prev_shares', [0.0] * n)
    
    # Smooth small changes (< 2 PRBs) to reduce churn penalty
    for k in range(n):
        if abs(shares[k] - prev_shares[k]) < 2.0:
            shares[k] = prev_shares[k]
    
    # Clamp to valid range and ensure sum <= pool_max
    shares = [max(0.0, s) for s in shares]
    total = sum(shares)
    
    if total > pool_max:
        scale = pool_max / total
        shares = [s * scale for s in shares]
    
    # Store for next iteration
    state.set('prev_shares', shares)
    
    return shares