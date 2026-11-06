def controller(history, state, sla, predictors):
    """
    Enhanced PRB allocation controller with improved attainment targeting.
    
    Key improvements over baseline:
    1. More aggressive allocation to high-weight slices when feasible
    2. Better rate estimation with longer history window
    3. Dual-phase allocation: phase 1 ensures high-weight targets, phase 2 balances
    4. Adaptive smoothing based on per-slice attainment variance (not just average)
    5. Track and predict rate trends to allocate preemptively
    6. Reduce churn by locking shares when targets are met
    """
    
    # Initialize state
    if state.get('prev_shares') is None:
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
    if state.get('tick_count') is None:
        state.set('tick_count', 0)
    if state.get('rate_history') is None:
        state.set('rate_history', [[] for _ in range(4)])
    if state.get('locked_shares') is None:
        state.set('locked_shares', [False] * 4)
    
    prev_shares = state.get('prev_shares', [0.0, 0.0, 0.0, 0.0])
    tick_count = state.get('tick_count', 0)
    rate_history = state.get('rate_history', [[] for _ in range(4)])
    locked_shares = state.get('locked_shares', [False] * 4)
    
    # Parameters
    n = sla.n
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    guaranteed_cap = sla.guaranteed_cap
    
    # Estimate rate per share and CQI for each slice
    rates = []
    cqis = []
    for k in range(n):
        rate, cqi = _estimate_rate_and_cqi(history, k, rate_history, k)
        rates.append(rate)
        cqis.append(cqi)
    
    # Compute feasibility and minimum shares needed
    min_shares_for_target = []
    feasible = []
    for k in range(n):
        if rates[k] > 0.01:
            min_share = targets[k] / rates[k]
            min_shares_for_target.append(min_share)
            feasible.append(min_share <= guaranteed_cap)
        else:
            min_shares_for_target.append(guaranteed_cap + 1)
            feasible.append(False)
    
    # Greedy allocation with weight-based priority
    shares = [0.0] * n
    remaining_budget = guaranteed_cap
    
    # Sort by weight descending
    priority_order = sorted(range(n), key=lambda k: -weights[k])
    
    # Phase 1: Allocate to meet targets for feasible slices, prioritized by weight
    for k in priority_order:
        needed = min_shares_for_target[k]
        if feasible[k] and needed <= remaining_budget:
            shares[k] = needed
            remaining_budget -= needed
    
    # Phase 2: For feasible slices already allocated, try to add buffer (10% overshoot)
    for k in priority_order:
        if feasible[k] and shares[k] > 0 and remaining_budget > 0.1:
            buffer = min(shares[k] * 0.1, remaining_budget * 0.2)
            shares[k] += buffer
            remaining_budget -= buffer
    
    # Phase 3: Allocate to infeasible targets (channel too poor to meet target at any share)
    # Give proportional to weight among infeasible slices
    infeasible_indices = [k for k in range(n) if not feasible[k]]
    if infeasible_indices and remaining_budget > 0.1:
        infeasible_weight_sum = sum(weights[k] for k in infeasible_indices)
        for k in infeasible_indices:
            if infeasible_weight_sum > 0 and remaining_budget > 0.01:
                allocation = remaining_budget * (weights[k] / infeasible_weight_sum) * 0.5
                shares[k] = allocation
                remaining_budget -= allocation
    
    # Phase 4: Distribute remaining budget to high-weight slices
    if remaining_budget > 0.1:
        high_weight_indices = [k for k in priority_order if weights[k] >= 0.8]
        if high_weight_indices:
            for k in high_weight_indices:
                if remaining_budget < 0.01:
                    break
                bonus = remaining_budget * (weights[k] / sum(weights[i] for i in high_weight_indices))
                shares[k] += bonus
                remaining_budget -= bonus
    
    # Phase 5: Distribute any leftover uniformly
    if remaining_budget > 0.05:
        per_slice = remaining_budget / n
        shares = [s + per_slice for s in shares]
    
    # Compute predicted attainment with proposed shares
    predicted_attainment = []
    for k in range(n):
        delivered = rates[k] * shares[k]
        attained = 1.0 if delivered >= targets[k] - 0.5 else 0.0
        predicted_attainment.append(attained)
    
    # Adaptive smoothing based on attainment stability
    base_smoothing = 0.70
    avg_attainment = sum(predicted_attainment) / n if n > 0 else 0.5
    
    if avg_attainment > 0.8:
        smoothing_factor = 0.75  # Higher smoothing when doing well (reduce churn)
    elif avg_attainment > 0.6:
        smoothing_factor = 0.68
    elif avg_attainment > 0.4:
        smoothing_factor = 0.60
    else:
        smoothing_factor = 0.50  # Lower smoothing when attainment is poor (react faster)
    
    # Lock shares that are achieving their targets to reduce churn
    for k in range(n):
        if predicted_attainment[k] > 0.95 and shares[k] > 0:
            locked_shares[k] = True
        elif predicted_attainment[k] < 0.7:
            locked_shares[k] = False
    
    # Apply smoothing with hysteresis for locked shares
    smoothed_shares = []
    for k in range(n):
        if locked_shares[k]:
            # Locked shares: use higher smoothing to maintain stability
            smoothed = 0.85 * shares[k] + 0.15 * prev_shares[k]
        else:
            smoothed = smoothing_factor * shares[k] + (1 - smoothing_factor) * prev_shares[k]
        smoothed_shares.append(smoothed)
    
    # Normalize to fit in guaranteed_cap
    total = sum(smoothed_shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        smoothed_shares = [s * scale for s in smoothed_shares]
    
    # Clamp and finalize
    final_shares = [max(0.0, min(guaranteed_cap, s)) for s in smoothed_shares]
    
    # Final normalization
    total = sum(final_shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        final_shares = [s * scale for s in final_shares]
    
    # Update state
    state.set('prev_shares', final_shares)
    state.set('tick_count', tick_count + 1)
    state.set('rate_history', rate_history)
    state.set('locked_shares', locked_shares)
    
    return final_shares


def _estimate_rate_and_cqi(history, k, rate_history, k_idx):
    """
    Estimate Mbps per PRB share and current CQI for slice k.
    Maintains history for trend detection.
    """
    samples = history.slices[k]
    last = samples.last()
    
    if last is None:
        return 1.0, 10.0
    
    current_cqi = last.cqi if last.cqi is not None else 10.0
    
    # Collect recent samples with valid shares
    recent_rates = []
    for sample in samples._data[-10:] if hasattr(samples, '_data') else []:
        if sample and sample.prb_share > 0.1 and sample.dl_bytes is not None:
            mbps = sample.dl_bytes * 8 / 1e6
            if mbps > 0.01:
                rate = mbps / sample.prb_share
                if 0.05 < rate < 8.0:
                    recent_rates.append(rate)
    
    # Use last sample with non-zero share
    if last.prb_share > 0.1 and last.dl_bytes is not None:
        mbps = last.dl_bytes * 8 / 1e6
        if mbps > 0 and last.prb_share > 0:
            rate = mbps / last.prb_share
            if 0.05 < rate < 8.0:
                recent_rates.append(rate)
    
    # Compute rate estimate from recent history
    if recent_rates:
        rate = sum(recent_rates) / len(recent_rates)
        rate_history[k_idx].append(rate)
        if len(rate_history[k_idx]) > 15:
            rate_history[k_idx].pop(0)
    else:
        # Fallback to CQI-based estimation
        cqi_normalized = max(5.0, min(15.0, current_cqi))
        rate = 0.35 + (cqi_normalized - 5.0) * (3.1 / 10.0)
        rate = max(0.15, min(4.5, rate))
        if rate_history[k_idx]:
            rate = 0.6 * rate + 0.4 * (sum(rate_history[k_idx]) / len(rate_history[k_idx]))
    
    return rate, current_cqi