def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller with:
    1. Better rate estimation using CQI-stratified history
    2. Probabilistic attainment targeting to guide allocation
    3. Reduced churn penalty acceptance when attainment improves
    4. Adaptive slack allocation prioritizing guaranteed targets
    5. Better handling of infeasible targets through demand modeling
    """
    n = sla.n  # 4
    guaranteed_cap = sla.guaranteed_cap  # 70
    pool_max = sla.pool_max  # 100
    
    # Persistent state
    prev_shares = state.get('prev_shares', [0.0] * n)
    rate_history = state.get('rate_history', [[] for _ in range(n)])
    cqi_history = state.get('cqi_history', [[] for _ in range(n)])
    attainment_history = state.get('attainment_history', [[0.5] * 10 for _ in range(n)])
    tick_count = state.get('tick_count', 0)
    
    # Slice parameters
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    
    # Current slice observations
    cqis = []
    latest_mbps = []
    latest_shares = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is not None:
            cqis.append(float(sample.cqi))
            latest_mbps.append(history.slices[k].latest_mbps())
            latest_shares.append(float(sample.prb_share))
        else:
            cqis.append(10.0)
            latest_mbps.append(0.0)
            latest_shares.append(0.0)
    
    tick_count += 1
    
    # Update CQI and attainment history
    for k in range(n):
        if len(cqi_history[k]) > 200:
            cqi_history[k].pop(0)
        cqi_history[k].append(cqis[k])
        
        # Track whether target was attained
        attained = 1.0 if latest_mbps[k] >= targets[k] - 0.5 else 0.0
        if len(attainment_history[k]) > 20:
            attainment_history[k].pop(0)
        attainment_history[k].append(attained)
    
    # Improved rate estimation with CQI stratification
    rates = []
    for k in range(n):
        # Collect rate observations with CQI bucketing
        if latest_shares[k] > 0.2 and latest_mbps[k] > 0.01:
            obs_rate = latest_mbps[k] / latest_shares[k]
            obs_rate = max(0.01, min(obs_rate, 15.0))
            if len(rate_history[k]) > 500:
                rate_history[k].pop(0)
            rate_history[k].append((cqis[k], obs_rate))
        
        # Estimate rate for current CQI with narrow matching
        cqi_matches = [(r, c) for c, r in rate_history[k] if abs(c - cqis[k]) < 0.8]
        
        if len(cqi_matches) >= 5:
            # Use recent CQI-matched observations with heavy recency
            num_recent = min(20, len(cqi_matches))
            recent_rates = [r for r, _ in cqi_matches[-num_recent:]]
            rate_k = sum(recent_rates) / len(recent_rates)
            rate_k = max(0.02, min(rate_k, 14.0))
        else:
            # Fallback: smooth CQI-to-rate with calibration
            normalized_cqi = max(0.0, min(1.0, (cqis[k] - 5.0) / 10.0))
            # More aggressive scaling at low CQI
            if normalized_cqi < 0.3:
                rate_k = 0.15 + 0.8 * (normalized_cqi ** 1.6)
            else:
                rate_k = 0.3 + 1.4 * (normalized_cqi ** 1.3)
            rate_k = max(0.02, rate_k)
        
        rates.append(rate_k)
    
    # Compute per-slice attainment probability (exponential smoothing)
    p_attain = []
    for k in range(n):
        recent_attainment = sum(attainment_history[k][-10:]) / max(1, len(attainment_history[k][-10:]))
        p_attain.append(recent_attainment)
    
    # Compute minimum shares needed for each target with margin
    min_shares_needed = []
    for k in range(n):
        min_sh = (targets[k] + 1.0) / max(rates[k], 0.05)  # +1 Mbps margin
        min_shares_needed.append(min_sh)
    
    # Compute attainability for each slice
    attainable = [min_shares_needed[k] <= guaranteed_cap * 0.95 for k in range(n)]
    
    # Phase 1: Greedy allocation prioritizing attainment
    shares = [0.0] * n
    remaining = guaranteed_cap
    
    # Sort by (weight * current attainment probability, descending)
    # Then by weight (higher weight gets priority)
    priority_order = sorted(
        range(n), 
        key=lambda k: (-weights[k] * (0.5 + 0.5 * p_attain[k]), -weights[k])
    )
    
    # First pass: Allocate to attainable slices
    for k in priority_order:
        if attainable[k] and remaining > 1.0:
            needed = min_shares_needed[k]
            alloc = min(needed * 1.08, remaining * 0.95)
            shares[k] = alloc
            remaining -= alloc
    
    # Second pass: Allocate to non-attainable slices (weighted by attainment priority)
    non_attainable = [k for k in range(n) if not attainable[k]]
    if non_attainable and remaining > 0.5:
        # Weight by (weight * historical attainment rate)
        total_priority = sum(
            weights[k] * (0.4 + 0.6 * p_attain[k]) for k in non_attainable
        )
        
        if total_priority > 0.001:
            # Allocate 60% of remaining to non-attainable
            for k in non_attainable:
                priority = weights[k] * (0.4 + 0.6 * p_attain[k])
                alloc = (priority / total_priority) * remaining * 0.6
                shares[k] += alloc
            remaining *= 0.4
    
    # Third pass: Top-up to maximize weighted attainment over remaining pool
    if remaining > 0.3:
        # For each slice, compute marginal benefit of more share
        marginal_priority = []
        for k in range(n):
            current_delivered = rates[k] * shares[k]
            shortfall = max(0.0, targets[k] - current_delivered)
            # Benefit is high if we're close to meeting target
            benefit = weights[k] * (1.0 if shortfall <= 2.0 else 0.5)
            marginal_priority.append(benefit)
        
        total_marginal = sum(marginal_priority)
        if total_marginal > 0.001:
            for k in range(n):
                alloc = (marginal_priority[k] / total_marginal) * remaining
                shares[k] += alloc
    
    # Phase 2: Hard constraint enforcement
    shares = [max(0.0, s) for s in shares]
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.01)
        shares = [s * scale for s in shares]
    
    # Phase 3: Intelligent churn control
    def compute_score(alloc):
        """Compute score for a given allocation."""
        score = 0.0
        for k in range(n):
            delivered = min(targets[k], rates[k] * alloc[k])
            attain_k = 1.0 if delivered >= targets[k] - 0.5 else 0.0
            score += weights[k] * attain_k - 0.5 * weights[k] * (1.0 - attain_k)
        be_share = max(0.0, min(guaranteed_cap, guaranteed_cap - sum(alloc)))
        be_mbps = (be_share / guaranteed_cap) * 239.0 * 0.72
        score += 0.6 * (be_mbps / 239.0)
        return score
    
    churn = sum(1 for k in range(n) if abs(shares[k] - prev_shares[k]) > 0.6)
    
    # Apply adaptive smoothing
    if churn > 0:
        new_score = compute_score(shares)
        old_score = compute_score(prev_shares)
        score_delta = new_score - old_score
        
        # More responsive: accept changes if score improves or churn is low
        if score_delta > 0.015 or churn <= 1:
            alpha = 0.75 + 0.15 * min(1.0, max(0.0, score_delta) / 0.05)
        elif churn == 2 and score_delta >= 0.0:
            alpha = 0.65
        else:
            alpha = 0.5 + 0.2 * min(1.0, max(0.0, score_delta) / 0.02)
        
        shares = [alpha * shares[k] + (1.0 - alpha) * prev_shares[k] for k in range(n)]
    
    # Phase 4: Final constraint enforcement
    shares = [max(0.0, s) for s in shares]
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.01)
        shares = [s * scale for s in shares]
    
    # Store state
    state.set('prev_shares', shares)
    state.set('rate_history', rate_history)
    state.set('cqi_history', cqi_history)
    state.set('attainment_history', attainment_history)
    state.set('tick_count', tick_count)
    
    return shares