def controller(history, state, sla, predictors):
    """
    Enhanced PRB allocation controller focusing on maximizing weighted attainment.
    
    Key improvements:
    - Dynamic rate estimation with better CQI mapping
    - Aggressive pursuit of high-weight slice attainment
    - Minimal churn through hysteresis and stability zones
    - Better handling of infeasible targets
    - Explicit best-effort optimization (room for it when high-weight slices attain)
    """
    
    n_slices = sla.n  # 4
    weights = [sla.slices[k].weight for k in range(n_slices)]
    targets = [sla.slices[k].target_mbps for k in range(n_slices)]
    guaranteed_cap = sla.guaranteed_cap  # 70
    
    # Current measurements
    current_mbps = []
    current_cqi = []
    current_shares = []
    
    for k in range(n_slices):
        sample = history.slices[k].last()
        mbps = history.slices[k].latest_mbps()
        current_mbps.append(mbps if mbps is not None else 0.0)
        
        if sample is not None:
            current_cqi.append(sample.cqi)
            current_shares.append(sample.prb_share)
        else:
            current_cqi.append(7.5)
            current_shares.append(0.0)
    
    # Improved rate-per-share estimation
    rate_per_share = []
    prev_rates = state.get('prev_rates', [1.2, 1.2, 1.2, 1.2])
    
    for k in range(n_slices):
        if current_shares[k] > 0.5 and current_mbps[k] > 0.5:
            # Measured rate with bias toward recent history
            measured_rate = current_mbps[k] / current_shares[k]
            smoothed_rate = 0.6 * prev_rates[k] + 0.4 * measured_rate
        else:
            # CQI-based estimation with refined mapping
            cqi = current_cqi[k]
            # Map CQI 5->0.5, CQI 10->1.2, CQI 15->2.2 (non-linear, captures saturation)
            if cqi <= 10:
                rate_estimate = 0.5 + (cqi - 5.0) * 0.14  # 0.5 to 1.2 over 5 to 10
            else:
                rate_estimate = 1.2 + (cqi - 10.0) * 0.2  # 1.2 to 2.2 over 10 to 15
            
            smoothed_rate = 0.7 * prev_rates[k] + 0.3 * rate_estimate
        
        smoothed_rate = max(0.3, min(smoothed_rate, 2.5))  # Clamp to reasonable range
        rate_per_share.append(smoothed_rate)
    
    state.set('prev_rates', rate_per_share)
    
    # Get previous allocation for churn minimization
    prev_shares = state.get('prev_shares', [17.5, 17.5, 17.5, 17.5])
    
    # Attainment threshold: target - 0.5 Mbps
    attainment_targets = [targets[k] - 0.5 for k in range(n_slices)]
    
    # Minimum shares needed for attainment
    min_shares_for_attain = []
    for k in range(n_slices):
        if rate_per_share[k] > 0.1:
            min_share = attainment_targets[k] / rate_per_share[k]
            min_shares_for_attain.append(min_share)
        else:
            min_shares_for_attain.append(guaranteed_cap + 1)  # Infeasible
    
    # Check which slices can be attained
    can_attain = [min_shares_for_attain[k] <= guaranteed_cap for k in range(n_slices)]
    
    # Score heuristic: prioritize by weight * attainability
    # Sort by (weight, can_attain descending, min_share ascending)
    priority_order = sorted(
        range(n_slices),
        key=lambda k: (-weights[k], -int(can_attain[k]), min_shares_for_attain[k])
    )
    
    # Allocate greedily to maximize weighted attainment
    shares = [0.0] * n_slices
    remaining = guaranteed_cap
    attained = [False] * n_slices
    
    # Phase 1: Secure attainment for as many high-weight slices as possible
    for k in priority_order:
        if can_attain[k]:
            needed = min_shares_for_attain[k]
            if remaining >= needed - 0.01:
                shares[k] = needed
                remaining -= needed
                attained[k] = True
    
    # Phase 2: For unattained slices that CAN be attained, top up toward attainment
    for k in priority_order:
        if not attained[k] and can_attain[k] and remaining > 0.1:
            gap = min_shares_for_attain[k] - shares[k]
            boost = min(gap, remaining * 0.5)  # Use half of remaining
            shares[k] += boost
            remaining -= boost
    
    # Phase 3: Preserve previous good allocations via hysteresis
    # If a slice was attaining before, try to keep its share stable
    for k in range(n_slices):
        prev_delivered = rate_per_share[k] * prev_shares[k]
        was_attaining = prev_delivered >= attainment_targets[k] - 0.2
        
        if was_attaining and shares[k] < prev_shares[k] and remaining > 0.5:
            # Previous allocation was good; try to restore it
            restore_amount = min(prev_shares[k] - shares[k], remaining * 0.2)
            shares[k] += restore_amount
            remaining -= restore_amount
    
    # Phase 4: Distribute remaining capacity by weight, avoiding unnecessary churn
    if remaining > 0.1:
        # Weight distribution with churn penalty consideration
        allocatable = []
        for k in range(n_slices):
            if shares[k] < guaranteed_cap:
                # Prefer slices where increasing share doesn't cause churn
                churn_cost = 1.0 if abs(shares[k] - prev_shares[k]) > 0.5 else 0.0
                value = weights[k] * (1.0 - 0.1 * churn_cost)
                allocatable.append((k, value))
        
        if allocatable:
            total_value = sum(v for _, v in allocatable)
            if total_value > 0.01:
                for k, value in allocatable:
                    share_of_remaining = (value / total_value) * remaining
                    shares[k] += share_of_remaining
                remaining = 0.0
    
    # Normalize to guaranteed cap
    total_shares = sum(shares)
    if total_shares > guaranteed_cap + 0.01:
        scale = guaranteed_cap / total_shares
        shares = [s * scale for s in shares]
    
    # Churn minimization: apply hysteresis to small changes
    final_shares = []
    hysteresis_threshold = 0.3
    
    for k in range(n_slices):
        new_share = shares[k]
        old_share = prev_shares[k]
        
        # If change is tiny and old was reasonable, stick with old
        if abs(new_share - old_share) < hysteresis_threshold:
            old_delivered = rate_per_share[k] * old_share
            if old_delivered >= attainment_targets[k] - 1.0:
                final_shares.append(old_share)
                continue
        
        final_shares.append(new_share)
    
    # Final clamp and normalize
    final_shares = [max(0.0, s) for s in final_shares]
    total = sum(final_shares)
    if total > guaranteed_cap + 0.01:
        scale = guaranteed_cap / total
        final_shares = [s * scale for s in final_shares]
    
    state.set('prev_shares', final_shares)
    
    return final_shares