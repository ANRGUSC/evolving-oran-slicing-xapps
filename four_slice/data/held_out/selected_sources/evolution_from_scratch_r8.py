def controller(history, state, sla, predictors):
    """
    Enhanced PRB allocation controller with:
    1. Adaptive efficiency learning with faster convergence
    2. Weighted-capacity matching: allocate proportionally to w_k * T_k
    3. Dynamic target feasibility detection
    4. Aggressive reallocation when efficiency changes detected
    5. Best-effort maximization by avoiding over-allocation
    """
    
    # Initialize state
    if state.get('initialized') is None:
        state.set('initialized', True)
        state.set('prev_shares', [0.0, 0.0, 0.0, 0.0])
        state.set('efficiency_estimates', [1.5, 1.5, 1.5, 1.5])
        state.set('recent_attain', [[0]*5 for _ in range(4)])
        state.set('tick_count', 0)
        state.set('last_total_demand', 0.0)
    
    state.set('tick_count', state.get('tick_count', 0) + 1)
    tick = state.get('tick_count')
    
    n = sla.n
    guaranteed_cap = sla.guaranteed_cap
    
    # Extract SLA parameters
    weights = [sla.slices[k].weight for k in range(n)]
    targets = [sla.slices[k].target_mbps for k in range(n)]
    
    # Collect current observations
    cqis = []
    last_shares = []
    last_mbps_list = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample is not None:
            cqis.append(sample.cqi)
            last_shares.append(sample.prb_share)
            last_mbps_list.append(history.slices[k].latest_mbps())
        else:
            cqis.append(10.0)
            last_shares.append(0.0)
            last_mbps_list.append(0.0)
    
    prev_shares = state.get('prev_shares', [0.0] * n)
    efficiency_estimates = state.get('efficiency_estimates', [1.5] * n)
    recent_attain = state.get('recent_attain', [[0]*5 for _ in range(4)])
    
    # Update efficiency estimates with aggressive learning
    new_efficiencies = []
    for k in range(n):
        if last_shares[k] > 0.2:
            actual_eff = last_mbps_list[k] / last_shares[k]
            # Faster convergence: 50% old, 50% new
            updated_eff = efficiency_estimates[k] * 0.5 + actual_eff * 0.5
        else:
            # CQI-based estimate with dynamic scaling
            cqi_norm = max(0.0, min(1.0, (cqis[k] - 5.0) / 10.0))
            # More aggressive efficiency curve
            cqi_eff = 0.2 + cqi_norm * cqi_norm * 3.8
            updated_eff = efficiency_estimates[k] * 0.6 + cqi_eff * 0.4
        
        updated_eff = max(0.15, min(4.5, updated_eff))
        new_efficiencies.append(updated_eff)
    
    efficiency_estimates = new_efficiencies
    state.set('efficiency_estimates', efficiency_estimates)
    
    # Track attainment history (rolling 5-tick window)
    current_attain = []
    for k in range(n):
        attain = 1.0 if last_mbps_list[k] >= targets[k] - 0.5 else 0.0
        current_attain.append(attain)
        recent_attain[k] = recent_attain[k][1:] + [attain]
    
    state.set('recent_attain', recent_attain)
    
    # Compute attainment rates (fraction of recent ticks where target met)
    attain_rates = [sum(recent_attain[k]) / 5.0 for k in range(n)]
    
    # Compute required shares to meet targets with confidence
    required_shares = []
    feasible = []
    
    for k in range(n):
        if efficiency_estimates[k] > 0.15:
            # Confidence margin: higher for unstable slices
            stability = attain_rates[k]
            margin = 1.10 + (1.0 - stability) * 0.15
            req = (targets[k] / efficiency_estimates[k]) * margin
            required_shares.append(req)
            feasible.append(req < guaranteed_cap * 0.95)
        else:
            required_shares.append(guaranteed_cap + 100)
            feasible.append(False)
    
    # Weighted capacity demand: w_k * T_k
    weighted_demands = [weights[k] * targets[k] for k in range(n)]
    total_weighted_demand = sum(weighted_demands)
    
    # Allocate based on weighted demand proportions, respecting efficiency
    base_allocation = []
    for k in range(n):
        if total_weighted_demand > 0:
            # Base: proportional to weighted demand
            base = (weighted_demands[k] / total_weighted_demand) * guaranteed_cap * 0.85
        else:
            base = guaranteed_cap / n
        base_allocation.append(base)
    
    # Adjust allocations to achieve targets
    shares = [0.0] * n
    remaining = guaranteed_cap
    
    # Phase 1: Assign minimum required shares for feasible targets (sorted by weight)
    order_by_weight = sorted(range(n), key=lambda k: weights[k], reverse=True)
    
    for k in order_by_weight:
        if feasible[k]:
            alloc = min(required_shares[k], remaining * 0.95)
            if alloc > 0.3:
                shares[k] = alloc
                remaining -= alloc
    
    # Phase 2: If pool is underutilized, distribute to high-weight non-attaining slices
    for k in order_by_weight:
        if attain_rates[k] < 0.7 and shares[k] > 0 and remaining > 1.0:
            boost = min(2.5, remaining * 0.25, required_shares[k] * 0.4)
            shares[k] += boost
            remaining -= boost
    
    # Phase 3: Reserve small portion for stability of consistently attaining high-weight slices
    for k in order_by_weight:
        if weights[k] >= 0.8 and attain_rates[k] >= 0.8 and remaining > 0.5:
            reserve = min(0.5, remaining * 0.1)
            shares[k] += reserve
            remaining -= reserve
    
    # Renormalize
    total_share = sum(shares)
    if total_share > guaranteed_cap + 0.01:
        scale = guaranteed_cap / total_share
        shares = [s * scale for s in shares]
    
    # Phase 4: Churn minimization with smart hysteresis
    final_shares = []
    churn_count = 0
    
    for k in range(n):
        prev = prev_shares[k]
        proposed = shares[k]
        
        # Strong stability when target is consistently met
        if attain_rates[k] >= 0.8 and prev > 0.5:
            if abs(proposed - prev) < 1.5:
                final_shares.append(prev)
            else:
                final_shares.append(proposed)
                if abs(proposed - prev) > 0.1:
                    churn_count += 1
        # Faster adaptation when target not met
        elif attain_rates[k] < 0.5:
            final_shares.append(proposed)
            if abs(proposed - prev) > 0.05:
                churn_count += 1
        # Medium stability for partially attaining
        else:
            blend = 0.6 * proposed + 0.4 * prev if prev > 0.2 else proposed
            final_shares.append(blend)
            if abs(blend - prev) > 0.1:
                churn_count += 1
    
    # Final cleanup and normalization
    final_shares = [max(0.0, s) for s in final_shares]
    
    # Remove negligible allocations
    final_shares = [s if s >= 0.15 else 0.0 for s in final_shares]
    
    total_final = sum(final_shares)
    if total_final > guaranteed_cap + 0.01:
        scale = guaranteed_cap / total_final
        final_shares = [s * scale for s in final_shares]
    
    state.set('prev_shares', final_shares[:])
    
    return final_shares