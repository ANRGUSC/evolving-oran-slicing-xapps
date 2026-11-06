def controller(history, state, sla, predictors):
    """
    Improved PRB allocation controller with:
    1. More aggressive rate estimation with explicit CQI-to-Mbps calibration
    2. Dual-mode allocation: first saturate feasible targets, then redistribute
    3. Better infeasible slice detection and minimal allocation strategy
    4. Reduced smoothing during poor attainment to enable faster convergence
    5. Explicit churn minimization with hysteresis bands
    """
    
    if state.get('init') is None:
        state.set('init', True)
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
        state.set('rate_estimates', [1.0, 1.0, 1.0, 1.0])
        state.set('cqi_history', [[] for _ in range(4)])
        state.set('tick_count', 0)
        state.set('recent_attains', [[] for _ in range(4)])
        state.set('last_alloc_attempt', [None] * 4)
    
    n = sla.n
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    guaranteed_cap = sla.guaranteed_cap
    
    tick_count = state.get('tick_count', 0) + 1
    state.set('tick_count', tick_count)
    
    rate_estimates = state.get('rate_estimates', [1.0] * n)
    cqi_history = state.get('cqi_history', [[] for _ in range(4)])
    recent_attains = state.get('recent_attains', [[] for _ in range(4)])
    prev_shares = state.get('prev_shares', [0.0] * n)
    
    # --- Improved rate estimation with CQI calibration ---
    for k in range(n):
        sample = history.slices[k].last()
        if sample is not None:
            cqi = sample.cqi
            prb_share = sample.prb_share
            delivered = sample.dl_bytes * 8 / 1e6
            
            # Track CQI history (keep last 30 samples for better averaging)
            if cqi is not None:
                if len(cqi_history[k]) >= 30:
                    cqi_history[k].pop(0)
                cqi_history[k].append(cqi)
            
            # Track attainment for convergence detection
            attain = 1.0 if delivered >= targets[k] - 0.5 else 0.0
            if len(recent_attains[k]) >= 15:
                recent_attains[k].pop(0)
            recent_attains[k].append(attain)
            
            # Rate estimation with three tiers of confidence
            if prb_share > 0.5 and delivered > 0:
                # High confidence: direct measurement from substantial share
                observed_rate = delivered / prb_share
                alpha = 0.80  # Aggressive update
                rate_estimates[k] = alpha * observed_rate + (1 - alpha) * rate_estimates[k]
            elif prb_share > 0.1 and delivered > 0:
                # Medium confidence: measurement from moderate share
                observed_rate = delivered / prb_share
                alpha = 0.60
                rate_estimates[k] = alpha * observed_rate + (1 - alpha) * rate_estimates[k]
            elif len(cqi_history[k]) >= 3:
                # Low confidence: CQI-based inference
                recent_cqi = cqi_history[k][-3:]
                avg_cqi = sum(recent_cqi) / len(recent_cqi)
                avg_cqi = max(5.0, min(15.0, avg_cqi))
                
                # Improved CQI-to-rate mapping
                # At CQI=5: ~0.15 Mbps/share, at CQI=15: ~2.0 Mbps/share
                cqi_norm = (avg_cqi - 5.0) / 10.0
                inferred_rate = 0.15 + cqi_norm * 1.85
                
                alpha = 0.50
                rate_estimates[k] = alpha * inferred_rate + (1 - alpha) * rate_estimates[k]
            
            rate_estimates[k] = max(0.03, min(3.0, rate_estimates[k]))
    
    state.set('rate_estimates', rate_estimates)
    state.set('cqi_history', cqi_history)
    state.set('recent_attains', recent_attains)
    
    # --- Improved feasibility detection ---
    min_shares_needed = [0.0] * n
    is_feasible = [False] * n
    
    for k in range(n):
        rate_k = rate_estimates[k]
        if rate_k > 0.03:
            # Minimum share needed to meet target
            min_share = targets[k] / rate_k
            min_shares_needed[k] = min_share
            
            # Feasible if target can be met with reasonable share
            # Use 0.80 threshold instead of 0.85 for stricter feasibility
            if min_share <= guaranteed_cap * 0.80:
                is_feasible[k] = True
    
    # --- Compute convergence metrics ---
    avg_attainment = 0.5
    if tick_count > 15:
        all_recent = []
        for k in range(n):
            if len(recent_attains[k]) > 0:
                all_recent.extend(recent_attains[k][-10:])
        if all_recent:
            avg_attainment = sum(all_recent) / len(all_recent)
    
    # --- Primary allocation strategy ---
    shares = [0.0] * n
    remaining = guaranteed_cap
    
    # Step 1: Allocate to feasible slices to meet targets
    # Sort by weight (descending) to prioritize high-weight slices
    feasible_slices = [k for k in range(n) if is_feasible[k]]
    feasible_slices.sort(key=lambda k: -weights[k])
    
    for k in feasible_slices:
        needed = min_shares_needed[k]
        # Allocate with small margin (2% buffer for stability)
        alloc = needed * 1.02
        if alloc <= remaining:
            shares[k] = alloc
            remaining -= alloc
        else:
            # Not enough for target + margin; still allocate what we can
            if remaining > needed * 0.9:
                shares[k] = remaining
                remaining = 0
            else:
                # Save capacity for better allocations
                shares[k] = 0.0
    
    # Step 2: Redistribute excess to feasible slices by weight
    if remaining > 1.0 and len(feasible_slices) > 0:
        total_feasible_weight = sum(weights[k] for k in feasible_slices)
        if total_feasible_weight > 0:
            for k in feasible_slices:
                extra = remaining * (weights[k] / total_feasible_weight)
                shares[k] += extra
            remaining = 0.0
    
    # Step 3: Minimal allocation to infeasible slices to reduce penalty
    # Only give to highest-weight infeasible slices
    infeasible_slices = [k for k in range(n) if not is_feasible[k]]
    if len(infeasible_slices) > 0 and remaining > 0.2:
        infeasible_slices.sort(key=lambda k: -weights[k])
        # Give small allocations to top 2 infeasible slices
        for i, k in enumerate(infeasible_slices[:2]):
            if remaining > 0.1:
                alloc = min(1.5, remaining / (2 - i))
                shares[k] = alloc
                remaining -= alloc
    
    # Ensure we don't exceed cap
    total = sum(shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.001)
        shares = [s * scale for s in shares]
    
    # --- Adaptive smoothing with hysteresis ---
    # Goal: reduce churn (which costs 0.02 per change) while staying responsive
    
    # Determine smoothing level based on convergence
    if avg_attainment > 0.75:
        base_smoothing = 0.35  # High stability when doing well
    elif avg_attainment > 0.55:
        base_smoothing = 0.15  # Moderate smoothing during medium performance
    else:
        base_smoothing = 0.03  # Very responsive during poor performance
    
    # Check for large CQI changes that warrant more responsiveness
    cqi_variance = 0.0
    for k in range(n):
        if len(cqi_history[k]) >= 5:
            recent_cqi = cqi_history[k][-5:]
            mean_cqi = sum(recent_cqi) / len(recent_cqi)
            var = sum((c - mean_cqi) ** 2 for c in recent_cqi) / len(recent_cqi)
            cqi_variance += var
    
    if cqi_variance > 2.0:
        base_smoothing = max(0.02, base_smoothing * 0.4)
    
    # Apply hysteresis to reduce thrashing
    # Only update if change exceeds threshold or smoothing is very low
    final_shares = []
    for k in range(n):
        delta = abs(shares[k] - prev_shares[k])
        # Hysteresis: only update if change is significant (>0.5) or smoothing is minimal
        if delta > 0.5 or base_smoothing < 0.05:
            smoothed = base_smoothing * prev_shares[k] + (1 - base_smoothing) * shares[k]
        else:
            smoothed = prev_shares[k]
        final_shares.append(max(0.0, smoothed))
    
    # Final cap enforcement
    total = sum(final_shares)
    if total > guaranteed_cap:
        scale = guaranteed_cap / max(total, 0.001)
        final_shares = [s * scale for s in final_shares]
    
    state.set('prev_shares', final_shares)
    
    return final_shares