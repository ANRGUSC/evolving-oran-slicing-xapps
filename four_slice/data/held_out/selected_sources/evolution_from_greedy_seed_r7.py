import math


def controller(history, state, sla, predictors):
    """
    Improved controller with refined allocation strategy, better CQI prediction,
    and smarter prioritization based on marginal improvement analysis.
    """
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap
    pool_max = sla.pool_max

    def eff(cqi):
        c = max(5.0, min(15.0, cqi))
        return 0.30 + 2.09 * ((c - 5.0) / 10.0) ** 1.578

    # Get current observations
    obs = [history.slices[k].last().cqi if history.slices[k].last() else 15.0 for k in range(n)]
    current_mbps = [history.slices[k].latest_mbps() or 0.0 for k in range(n)]
    
    # Track CQI, delivery, and share history
    cqi_history = state.get("cqi_history", [[] for _ in range(n)])
    mbps_history = state.get("mbps_history", [[] for _ in range(n)])
    
    for k in range(n):
        cqi_history[k].append(obs[k])
        if len(cqi_history[k]) > 20:
            cqi_history[k].pop(0)
        
        mbps_history[k].append(current_mbps[k])
        if len(mbps_history[k]) > 20:
            mbps_history[k].pop(0)
    
    state.set("cqi_history", cqi_history)
    state.set("mbps_history", mbps_history)
    
    # Compute CQI statistics
    cqi_means = []
    cqi_trends = []
    cqi_stds = []
    
    for k in range(n):
        if len(cqi_history[k]) >= 2:
            mean = sum(cqi_history[k]) / len(cqi_history[k])
            cqi_means.append(mean)
            
            # Recent vs older trend
            recent = sum(cqi_history[k][-5:]) / min(5, len(cqi_history[k]))
            older = sum(cqi_history[k][:-5]) / max(1, len(cqi_history[k]) - 5) if len(cqi_history[k]) > 5 else mean
            trend = recent - older
            cqi_trends.append(trend)
            
            # Volatility
            var = sum((x - mean) ** 2 for x in cqi_history[k]) / len(cqi_history[k])
            std = math.sqrt(var)
            cqi_stds.append(std)
        else:
            cqi_means.append(obs[k])
            cqi_trends.append(0.0)
            cqi_stds.append(0.5)
    
    # Adaptive margin based on channel behavior
    margin_adjustments = []
    for k in range(n):
        # Start with base margin
        base = 0.88
        
        # Reduce margin for degrading channels to protect them
        trend_factor = max(0.0, -cqi_trends[k] * 0.03)
        
        # Increase margin for improving/stable channels
        if cqi_trends[k] > 0.1:
            trend_factor = -0.02
        
        # Reduce margin for volatile channels
        volatility_factor = min(0.08, cqi_stds[k] * 0.015)
        
        margin_k = base - trend_factor - volatility_factor
        margin_k = max(0.70, min(0.94, margin_k))
        margin_adjustments.append(margin_k)
    
    # Compute per-share rates
    rate = [max(0.05, eff(obs[k]) * margin_adjustments[k]) for k in range(n)]
    
    # Determine attainment status
    attaining = [current_mbps[k] >= T[k] - 0.5 for k in range(n)]
    
    # Compute minimum shares to meet target
    need = [max(1, math.ceil((T[k] - 0.5) / rate[k])) if rate[k] > 0 else cap for k in range(n)]
    
    # Start with zero allocation
    shares = [0] * n
    used = 0
    
    # Phase 1: Mandatory allocation - ensure all slices get at least minimal service
    # Allocate to meet targets by weight priority, but also by current attainment status
    priority_order = sorted(
        range(n),
        key=lambda k: (
            not attaining[k],  # Non-attaining first (True > False, so not attaining = True)
            -W[k],              # Then by weight descending
            -cqi_means[k]       # Then by CQI mean descending
        )
    )
    
    for k in priority_order:
        required = need[k]
        if used + required <= cap:
            shares[k] = required
            used += required
    
    # Phase 2: Secondary allocation for slices that couldn't get their full need
    # Try to give them at least 1 share to avoid zero allocation
    for k in range(n):
        if shares[k] == 0 and used + 1 <= cap:
            shares[k] = 1
            used += 1
    
    # Phase 3: Distribute remaining capacity strategically
    remaining_cap = cap - used
    if remaining_cap > 0:
        # For each unfinished slice, compute marginal benefit of additional share
        marginal_benefits = []
        for k in range(n):
            if shares[k] < need[k] and shares[k] < cap:
                # Marginal gain: additional share moves toward target
                current_delivery = rate[k] * shares[k]
                next_delivery = rate[k] * (shares[k] + 1)
                deficit = max(0, T[k] - current_delivery)
                next_deficit = max(0, T[k] - next_delivery)
                
                # Benefit is deficit reduction weighted by slice weight
                marginal_benefit = W[k] * (deficit - next_deficit)
                marginal_benefits.append((marginal_benefit, k))
        
        # Allocate greedily by marginal benefit
        marginal_benefits.sort(reverse=True)
        for benefit, k in marginal_benefits:
            if remaining_cap <= 0:
                break
            alloc = min(1, remaining_cap, need[k] - shares[k])
            shares[k] += alloc
            remaining_cap -= alloc
    
    # Phase 4: Churn control with smarter decision logic
    prev = state.get("prev_shares")
    
    if prev is not None:
        churn = sum(1 for k in range(n) if shares[k] != prev[k])
        
        if churn > 0:
            # Compute expected score change
            def estimate_score(s_list):
                score = 0.0
                for k in range(n):
                    delivered = min(T[k], rate[k] * s_list[k])
                    attain = 1 if delivered >= T[k] - 0.5 else 0
                    score += W[k] * attain - 0.5 * W[k] * (1 - attain)
                return score
            
            score_new = estimate_score(shares)
            score_prev = estimate_score(prev)
            score_delta = score_new - score_prev
            
            # Churn cost
            churn_cost = 0.02 * churn
            
            # Only accept if net gain is positive or very small churn
            if score_delta < churn_cost and churn > 1:
                shares = list(prev)
    
    state.set("prev_shares", list(shares))
    
    return shares