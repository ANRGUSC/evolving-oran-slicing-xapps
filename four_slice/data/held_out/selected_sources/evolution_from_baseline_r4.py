def controller(history, state, sla, predictors):
    """
    Enhanced controller with:
    1. Improved rate estimation with robust CQI-to-rate mapping
    2. Target-aware allocation with demand buffering
    3. Dynamic priority based on attainment gap and channel quality
    4. Aggressive reallocation when targets are chronically unmet
    5. Churn minimization through conservative smoothing
    6. Pool efficiency optimization balancing guaranteed and best-effort
    """
    
    if not state.get("init", False):
        state.set("init", True)
        state.set("rates", [1.0, 1.0, 1.0, 1.0])
        state.set("prev_shares", [17, 17, 18, 18])
        state.set("rate_ema", [1.0, 1.0, 1.0, 1.0])
        state.set("attain_count", [0, 0, 0, 0])  # Count of last 10 ticks attaining
        state.set("tick_count", 0)
    
    rates = state.get("rates", [1.0, 1.0, 1.0, 1.0])
    prev_shares = state.get("prev_shares", [17, 17, 18, 18])
    rate_ema = state.get("rate_ema", [1.0, 1.0, 1.0, 1.0])
    attain_count = state.get("attain_count", [0, 0, 0, 0])
    tick_count = state.get("tick_count", 0)
    
    tick_count += 1
    state.set("tick_count", tick_count)
    
    # Gather telemetry
    cqis = []
    mbps_delivered = []
    targets = []
    weights = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        cqis.append(cqi)
        
        # Improved CQI-to-rate mapping using power law
        # Empirically calibrated: at CQI=5 rate≈0.3, at CQI=15 rate≈2.4
        def cqi_to_rate(cqi_val):
            c = max(5.0, min(15.0, cqi_val))
            # Power-law fit: rate ≈ 0.25 + 2.0 * ((c-5)/10)^1.6
            normalized = (c - 5.0) / 10.0
            return 0.25 + 2.0 * (normalized ** 1.6)
        
        rate = cqi_to_rate(cqi)
        
        # EMA smoothing with adaptive alpha (faster response to degradation)
        if rate < rate_ema[k] * 0.85:
            # Fast decay when channel degrades
            alpha = 0.4
        else:
            # Slow recovery when channel improves
            alpha = 0.15
        
        rate_ema[k] = alpha * rate + (1.0 - alpha) * rate_ema[k]
        rates[k] = rate_ema[k]
        
        mbps = history.slices[k].latest_mbps() or 0.0
        mbps_delivered.append(mbps)
        
        target = sla.slices[k].target_mbps
        targets.append(target)
        weights.append(sla.slices[k].weight)
        
        # Track attainment in sliding window
        is_attaining = mbps >= target - 0.5
        if is_attaining:
            attain_count[k] = min(10, attain_count[k] + 1)
        else:
            attain_count[k] = max(0, attain_count[k] - 1)
    
    state.set("rates", rates)
    state.set("rate_ema", rate_ema)
    state.set("attain_count", attain_count)
    
    # Compute minimum shares needed for each target
    min_shares_for_target = []
    is_viable = []
    
    for k in range(sla.n):
        if rates[k] > 0.05:
            # Add small buffer for variance: +0.5 shares
            min_share = targets[k] / rates[k] + 0.5
            min_shares_for_target.append(min_share)
            is_viable.append(True)
        else:
            # Channel too degraded; target unachievable
            min_shares_for_target.append(sla.guaranteed_cap + 10)
            is_viable.append(False)
    
    # Assess pool pressure
    total_min_needed = sum(min_shares_for_target)
    pool_pressure = max(0.0, (total_min_needed - sla.guaranteed_cap) / max(1.0, total_min_needed))
    
    # Dynamic priority scoring
    priority_scores = []
    for k in range(sla.n):
        # Base priority: weight
        base_priority = weights[k]
        
        # Attainment gap: how far from target
        attainment_ratio = attain_count[k] / 10.0
        gap_factor = (1.0 - attainment_ratio) * 0.5  # Failing slices get boost
        
        # Channel viability factor (CQI-based)
        cqi_viable = max(0.1, (cqis[k] - 4.5) / 10.5)
        
        # Combined score: prioritize failing, high-weight, viable slices
        score = base_priority * (1.0 + gap_factor) * cqi_viable
        priority_scores.append(score)
    
    priority_order = sorted(range(sla.n), key=lambda k: priority_scores[k], reverse=True)
    
    # Allocation strategy
    out = [0] * sla.n
    remaining = sla.guaranteed_cap
    
    # Phase 1: Allocate to meet targets for viable slices, in priority order
    for k in priority_order:
        if remaining <= 0.1:
            break
        
        if is_viable[k]:
            min_needed = min_shares_for_target[k]
            # Add extra cushion for high-weight slices to reduce churn
            extra_buffer = 1.5 if weights[k] >= 0.85 else 0.5
            target_alloc = min(min_needed + extra_buffer, remaining)
            target_alloc = max(1, target_alloc)  # At least 1 share
            out[k] = target_alloc
            remaining -= target_alloc
    
    # Phase 2: Ensure minimum viability for low-priority viable slices
    for k in reversed(priority_order):
        if out[k] == 0 and remaining > 1.0 and is_viable[k]:
            if cqis[k] >= 10.0:  # Only if channel is reasonably good
                out[k] = min(1.5, remaining)
                remaining -= out[k]
    
    # Anti-churn stabilization: keep allocations close to previous if conditions allow
    churn_potential = sum(1 for k in range(sla.n) if abs(out[k] - prev_shares[k]) > 0.5)
    
    if churn_potential >= 2 and pool_pressure < 0.3 and tick_count > 5:
        # Apply conservative hysteresis to reduce churn
        smoothed = []
        for k in range(sla.n):
            prev = prev_shares[k]
            curr = out[k]
            
            # If current is close to previous and previous was feasible, use previous
            if abs(curr - prev) <= 1.5 and attain_count[k] >= 5:
                smoothed.append(prev)
            else:
                smoothed.append(curr)
        
        total_smoothed = sum(smoothed)
        if total_smoothed <= sla.guaranteed_cap:
            out = smoothed
        else:
            # Normalize down proportionally
            scale = sla.guaranteed_cap / total_smoothed
            out = [x * scale for x in smoothed]
    
    # Final safety checks and rounding
    out = [max(0, float(x)) for x in out]
    total = sum(out)
    
    if total > sla.guaranteed_cap:
        # Trim from lowest priority
        scale = sla.guaranteed_cap / total
        out = [x * scale for x in out]
    
    # Round intelligently: preserve order
    out_int = [int(x) for x in out]
    remainder = sla.guaranteed_cap - sum(out_int)
    
    if remainder > 0:
        # Distribute remainder to highest-priority slices
        fracs = [out[k] - out_int[k] for k in range(sla.n)]
        indices = sorted(range(sla.n), key=lambda k: fracs[k], reverse=True)
        for i in range(min(remainder, len(indices))):
            out_int[indices[i]] += 1
    
    out = [max(0, min(sla.guaranteed_cap, x)) for x in out_int]
    
    state.set("prev_shares", out)
    
    return out