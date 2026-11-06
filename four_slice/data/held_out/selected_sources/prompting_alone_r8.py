def controller(history, state, sla, predictors):
    """
    PRB allocation controller for 4 guaranteed slices + 1 best-effort slice.
    Maximizes: sum_k [ w_k * attain_k - 0.5 * w_k * (1 - attain_k) ] + 0.6 * (be_mbps / 239) - 0.02 * churn
    
    Strategy:
    1. Estimate each slice's throughput per share from recent history (using CQI and delivery)
    2. Allocate shares to maximize attainment of targets, prioritizing high-weight slices
    3. Use water-filling / proportional-fair heuristic adjusted for weights
    4. Smooth allocation to minimize churn
    """
    
    # Initialize state tracking
    if not state.get('prev_shares'):
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
    
    prev_shares = state.get('prev_shares', [0.0, 0.0, 0.0, 0.0])
    
    n = sla.n  # 4 guaranteed slices
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Estimate throughput per share for each slice from history
    mbps_per_share = []
    for k in range(n):
        hist = history.slices[k]
        last = hist.last()
        
        if last is None:
            # No history: use CQI to estimate
            cqi = 10.0  # neutral assumption
            mbps_per_share.append(cqi_to_mbps_per_share(cqi))
        else:
            # Estimate from last sample: mbps_per_share = delivered_mbps / prev_share
            delivered_mbps = hist.latest_mbps()
            prev_share = last.prb_share if last.prb_share > 0 else 1.0
            estimated = delivered_mbps / prev_share
            
            # Also consider CQI trend
            cqi = last.cqi
            cqi_estimate = cqi_to_mbps_per_share(cqi)
            
            # Blend: trust history but clip to CQI bounds
            mbps_per_share.append(max(0.1, min(estimated, cqi_estimate * 1.2)))
    
    # Compute shares using weighted water-filling heuristic
    shares = allocate_shares(targets, weights, mbps_per_share, guaranteed_cap)
    
    # Apply smoothing to reduce churn
    alpha = 0.3  # smoothing factor
    smoothed_shares = [
        alpha * shares[k] + (1 - alpha) * prev_shares[k]
        for k in range(n)
    ]
    
    # Normalize to stay within cap
    total = sum(smoothed_shares)
    if total > guaranteed_cap:
        smoothed_shares = [s * guaranteed_cap / total for s in smoothed_shares]
    
    # Clamp negative values to 0
    smoothed_shares = [max(0.0, s) for s in smoothed_shares]
    
    # Store for next tick
    state.set('prev_shares', smoothed_shares)
    
    return smoothed_shares


def cqi_to_mbps_per_share(cqi):
    """
    Map CQI (5..15) to estimated Mbps per share.
    CQI 5 = poor channel (near 0.3 Mbps/share)
    CQI 15 = excellent channel (near 2.5 Mbps/share)
    """
    if cqi < 5:
        return 0.1
    if cqi > 15:
        return 2.5
    # Linear interpolation
    return 0.3 + (cqi - 5.0) * (2.5 - 0.3) / (15.0 - 5.0)


def allocate_shares(targets, weights, mbps_per_share, cap):
    """
    Allocate shares to meet targets, prioritizing by weight and efficiency.
    Uses a greedy water-filling approach adjusted for weighted priorities.
    """
    n = len(targets)
    shares = [0.0] * n
    remaining_cap = cap
    
    # Compute minimum share needed to meet each target (if possible)
    min_shares_for_target = []
    for k in range(n):
        rate_k = mbps_per_share[k]
        if rate_k > 0.01:
            min_share = targets[k] / rate_k
        else:
            min_share = cap + 1  # impossible
        min_shares_for_target.append(min_share)
    
    # Greedy allocation: prioritize high-weight slices that can meet target
    # with small share allocation
    priorities = []
    for k in range(n):
        if min_shares_for_target[k] <= cap:
            # Efficiency metric: weight / required_share
            efficiency = weights[k] / max(0.1, min_shares_for_target[k])
            priorities.append((efficiency, -weights[k], k))  # Sort by efficiency, then weight
        else:
            priorities.append((0.0, -weights[k], k))  # Cannot meet target
    
    priorities.sort(reverse=True)
    
    # Phase 1: allocate to meet targets (greedy by efficiency and weight)
    for _, _, k in priorities:
        min_needed = min_shares_for_target[k]
        if min_needed <= remaining_cap:
            shares[k] = min(min_needed, remaining_cap)
            remaining_cap -= shares[k]
    
    # Phase 2: distribute remaining capacity proportionally by weight
    if remaining_cap > 0.01:
        total_weight = sum(weights)
        for k in range(n):
            if total_weight > 0:
                extra = remaining_cap * weights[k] / total_weight
                shares[k] += extra
    
    # Normalize if exceeded cap (safety)
    total = sum(shares)
    if total > cap:
        shares = [s * cap / total for s in shares]
    
    return shares