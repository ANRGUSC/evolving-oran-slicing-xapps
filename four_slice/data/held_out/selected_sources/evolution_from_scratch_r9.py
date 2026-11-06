def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller with focus on:
    - Better rate estimation with multiplicative CQI effects
    - Dynamic weight reallocation based on attainment prospects
    - Aggressive best-effort protection (pool must stay productive)
    - Minimal churn through stability regions
    """
    
    # Initialize state
    if not state.get('initialized'):
        state.set('initialized', True)
        state.set('prev_shares', [0, 0, 0, 0])
        state.set('rate_estimates', [1.0, 1.0, 1.0, 1.0])
        state.set('rate_history', [[] for _ in range(4)])
        state.set('cqi_history', [[] for _ in range(4)])
        state.set('attainment_history', [[0]*10 for _ in range(4)])
        state.set('tick_count', 0)
    
    n_slices = sla.n
    targets = [sla.slices[k].target_mbps for k in range(n_slices)]
    weights = [sla.slices[k].weight for k in range(n_slices)]
    guaranteed_cap = sla.guaranteed_cap
    
    # Collect telemetry
    cqis = []
    last_shares = []
    delivered_mbps_list = []
    
    for k in range(n_slices):
        sample = history.slices[k].last()
        if sample is not None:
            cqis.append(sample.cqi)
            last_shares.append(sample.prb_share)
            delivered_mbps = sample.dl_bytes * 8 / 1e6
            delivered_mbps_list.append(delivered_mbps)
        else:
            cqis.append(10.0)
            last_shares.append(0)
            delivered_mbps_list.append(0)
    
    # Update CQI history
    cqi_history = state.get('cqi_history', [[] for _ in range(4)])
    for k in range(n_slices):
        cqi_history[k].append(cqis[k])
        if len(cqi_history[k]) > 15:
            cqi_history[k].pop(0)
    state.set('cqi_history', cqi_history)
    
    # Update attainment history
    attainment_history = state.get('attainment_history', [[0]*10 for _ in range(4)])
    for k in range(n_slices):
        attain = 1 if delivered_mbps_list[k] >= targets[k] - 0.5 else 0
        attainment_history[k].append(attain)
        if len(attainment_history[k]) > 10:
            attainment_history[k].pop(0)
    state.set('attainment_history', attainment_history)
    
    # Estimate rates with robust filtering
    prev_rates = state.get('rate_estimates', [1.0]*4)
    rate_history = state.get('rate_history', [[] for _ in range(4)])
    estimated_rates = []
    
    for k in range(n_slices):
        if last_shares[k] > 0.05:
            observed_rate = delivered_mbps_list[k] / last_shares[k]
            rate_history[k].append(observed_rate)
            if len(rate_history[k]) > 30:
                rate_history[k].pop(0)
            
            # Use median with outlier rejection
            if len(rate_history[k]) >= 5:
                sorted_rates = sorted(rate_history[k])
                median_rate = sorted_rates[len(sorted_rates) // 2]
                
                # IQR-based outlier detection
                q1 = sorted_rates[len(sorted_rates) // 4] if len(sorted_rates) > 2 else median_rate
                q3 = sorted_rates[3 * len(sorted_rates) // 4] if len(sorted_rates) > 2 else median_rate
                iqr = max(0.01, q3 - q1)
                
                if abs(observed_rate - median_rate) <= 2.5 * iqr:
                    valid_rate = observed_rate
                else:
                    valid_rate = median_rate
                
                # Adaptive alpha based on CQI trend
                cqi_trend = sum(cqi_history[k][-5:]) / len(cqi_history[k][-5:]) if len(cqi_history[k]) >= 5 else cqis[k]
                cqi_prev = cqi_history[k][-5] if len(cqi_history[k]) >= 5 else cqis[k]
                
                if cqi_trend > cqi_prev + 0.5:
                    alpha = 0.55  # Improving: respond faster
                else:
                    alpha = 0.35  # Stable/degrading: be conservative
                
                smoothed_rate = alpha * valid_rate + (1 - alpha) * prev_rates[k]
            else:
                smoothed_rate = 0.7 * observed_rate + 0.3 * prev_rates[k]
        else:
            smoothed_rate = prev_rates[k] * 0.88  # Decay unused slices
        
        estimated_rates.append(max(0.01, smoothed_rate))
    
    state.set('rate_estimates', estimated_rates)
    state.set('rate_history', rate_history)
    
    # Improved CQI-to-multiplier mapping (multiplicative effect)
    cqi_factors = []
    for cqi in cqis:
        if cqi <= 5.0:
            factor = 0.0
        elif cqi <= 5.5:
            factor = 0.002 + (cqi - 5.0) * 0.004
        elif cqi <= 7.0:
            factor = 0.005 + (cqi - 5.5) * 0.02
        elif cqi <= 9.0:
            factor = 0.035 + (cqi - 7.0) * 0.08
        elif cqi <= 11.0:
            factor = 0.195 + (cqi - 9.0) * 0.12
        elif cqi <= 13.0:
            factor = 0.435 + (cqi - 11.0) * 0.15
        else:
            factor = 0.735 + min((cqi - 13.0) * 0.1, 0.165)
        
        cqi_factors.append(max(0.0, min(1.0, factor)))
    
    # Compute achievable throughput per share
    achievable_per_share = [max(0.01, estimated_rates[k] * cqi_factors[k]) for k in range(n_slices)]
    
    # Compute required shares and attainability
    required_shares = []
    can_meet_target = []
    gap_scores = []  # How much above/below target we can go
    
    for k in range(n_slices):
        if achievable_per_share[k] > 0.01:
            req = targets[k] / achievable_per_share[k]
            required_shares.append(req)
            can_meet = req <= guaranteed_cap
            can_meet_target.append(can_meet)
            
            # Gap score: how easily can we meet target vs how achievable is next tier
            if can_meet:
                gap_scores.append(1.0 - (req / guaranteed_cap))
            else:
                gap_scores.append(-(req - guaranteed_cap) / guaranteed_cap)
        else:
            required_shares.append(1000.0)
            can_meet_target.append(False)
            gap_scores.append(-10.0)
    
    # Compute allocation priorities: maximize score
    # score += w_k * attain_k - 0.5 * w_k * (1 - attain_k) = w_k * (1.5*attain_k - 0.5)
    # When attain_k prob = p: expected = w_k * (1.5*p - 0.5)
    # Derivative w.r.t. share = w_k * 1.5 * dp/dshare
    
    recent_attain = [sum(attainment_history[k][-5:]) / 5.0 for k in range(n_slices)]
    
    priorities = []
    for k in range(n_slices):
        base_weight = weights[k]
        
        # Boost high-weight slices with good attainment
        if recent_attain[k] >= 0.7:
            weight_factor = 1.4
        elif recent_attain[k] >= 0.4:
            weight_factor = 1.1
        else:
            weight_factor = 0.9
        
        # Penalty for unreachable targets
        if can_meet_target[k]:
            priority = base_weight * weight_factor * (1.0 + gap_scores[k])
        else:
            # Still allocate to struggling slices but lower priority
            priority = base_weight * 0.5
        
        priorities.append(priority)
    
    # Allocation strategy: maximize score by weight and attainability
    allocation_order = sorted(range(n_slices), key=lambda k: -priorities[k])
    
    shares = [0.0] * n_slices
    remaining_capacity = guaranteed_cap
    
    # Phase 1: Allocate to meet targets for achievable slices, prioritized by weight
    for k in allocation_order:
        if can_meet_target[k] and remaining_capacity > 0.2:
            # Give 110% of required to add headroom, then clamp to remaining
            needed = min(required_shares[k] * 1.1, remaining_capacity)
            shares[k] = needed
            remaining_capacity -= needed
    
    # Phase 2: Allocate to struggling slices proportional to weight
    if remaining_capacity > 1.0:
        struggling = [k for k in range(n_slices) if not can_meet_target[k]]
        if struggling:
            struggle_weights = [weights[k] for k in struggling]
            total_struggle_weight = sum(struggle_weights)
            for k in struggling:
                if total_struggle_weight > 0:
                    alloc = remaining_capacity * (weights[k] / total_struggle_weight) * 0.8
                    shares[k] += alloc
                    remaining_capacity -= alloc
    
    # Phase 3: Reinvest any remaining in high-weight winners
    if remaining_capacity > 0.3:
        high_weight_winners = sorted(
            [k for k in range(n_slices) if can_meet_target[k]],
            key=lambda k: -weights[k]
        )
        for k in high_weight_winners:
            if remaining_capacity > 0.1:
                boost = min(remaining_capacity * 0.5, 2.0)
                shares[k] += boost
                remaining_capacity -= boost
    
    # Final safety
    shares = [max(0, s) for s in shares]
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / total
        shares = [s * scale for s in shares]
    
    # Churn reduction: smooth allocations for stable slices
    prev_shares = state.get('prev_shares', [0, 0, 0, 0])
    final_shares = []
    
    for k in range(n_slices):
        # Higher stability for slices with consistent attainment
        attain_streak = sum(attainment_history[k][-3:])
        
        if attain_streak >= 2.5:  # Recent consistent attainment
            smoothing = 0.8  # Strong stability
        elif attain_streak >= 1.0:
            smoothing = 0.6
        else:
            smoothing = 0.25  # Struggling: rapid adjustment
        
        smoothed = smoothing * prev_shares[k] + (1 - smoothing) * shares[k]
        final_shares.append(max(0, smoothed))
    
    state.set('prev_shares', final_shares)
    state.set('tick_count', state.get('tick_count', 0) + 1)
    
    # Final normalization
    total = sum(final_shares)
    if total > guaranteed_cap * 1.001:
        scale = guaranteed_cap / total
        final_shares = [s * scale for s in final_shares]
    
    return final_shares