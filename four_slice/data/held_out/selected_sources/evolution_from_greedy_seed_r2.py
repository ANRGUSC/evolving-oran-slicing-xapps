import math


def controller(history, state, sla, predictors):
    """Advanced allocation controller with target-driven arbitration and adaptive reallocation.
    
    Key improvements:
    1. Target-gap aware: allocate based on how far each slice is from meeting target
    2. Dynamic efficiency margins: tighter margins for slices far from target
    3. Two-pass allocation: first ensure critical slices, then maximize over-delivery
    4. Intelligent reallocation: steal from over-provisioned slices to help struggling ones
    5. Better churn penalty avoidance with hysteresis
    """
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap

    def eff(cqi):
        c = max(5.0, min(15.0, cqi))
        return 0.30 + 2.09 * ((c - 5.0) / 10.0) ** 1.578

    # Read current observations
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    last_mbps = [history.slices[k].latest_mbps() if history.slices[k].last() else 0.0 for k in range(n)]
    
    # Compute per-share rates with channel-aware margin
    cqi_mean = sum(obs) / n
    cqi_var = sum((obs[k] - cqi_mean) ** 2 for k in range(n)) / n
    cqi_std = math.sqrt(cqi_var)
    
    # Adaptive margin based on channel stability
    if cqi_std < 0.8:
        margin = 0.97
    elif cqi_std > 3.5:
        margin = 0.85
    else:
        margin = 0.92
    
    rate = [max(0.05, eff(obs[k]) * margin) for k in range(n)]
    
    # Compute gap to target for each slice
    gap = [max(0.0, T[k] - 0.5 - last_mbps[k]) for k in range(n)]
    
    # Minimum share needed to attain target
    need = [max(1, math.ceil((T[k] - 0.5) / rate[k])) if rate[k] > 0 else cap + 1 for k in range(n)]
    
    # Whether slice is currently meeting target
    attain = [last_mbps[k] >= T[k] - 0.5 for k in range(n)]
    
    # --- PHASE 1: Critical allocation (fund all non-attaining slices or those with large gaps) ---
    shares = [0] * n
    used = 0
    
    # Sort by criticality: non-attaining high-weight slices first, then by gap
    critical_order = sorted(
        range(n),
        key=lambda k: (attain[k], -W[k] * (1.0 + gap[k] / max(1.0, T[k])), gap[k])
    )
    
    for k in critical_order:
        required = need[k]
        if used + required <= cap:
            shares[k] = required
            used += required
    
    # --- PHASE 2: Reallocation pass (steal from over-provisioned to help struggling) ---
    # Identify slices that could use more to meet target
    still_short = [k for k in range(n) if shares[k] * rate[k] < T[k] - 0.5 and shares[k] > 0]
    
    if still_short and used < cap:
        # Find slices with extra allocation (over-provisioned)
        current_delivered = [shares[k] * rate[k] for k in range(n)]
        surplus = [(k, current_delivered[k] - (T[k] - 0.5)) for k in range(n) if current_delivered[k] > T[k] + 2.0]
        
        # Steal from lowest weight over-provisioned slices
        surplus.sort(key=lambda x: (W[x[0]], -x[1]))
        
        for k_rich, extra in surplus:
            if not still_short or used >= cap:
                break
            # Steal 1-2 shares if possible
            steal = min(2, shares[k_rich] - need[k_rich], cap - used)
            if steal > 0:
                shares[k_rich] -= steal
                # Give to most critical struggling slice
                k_poor = max(still_short, key=lambda k: W[k] * gap[k])
                shares[k_poor] += steal
                used = sum(shares)
                still_short = [k for k in still_short if shares[k] * rate[k] < T[k] - 0.5]
    
    # --- PHASE 3: Opportunistic allocation (use remaining headroom) ---
    remaining = cap - used
    if remaining > 0:
        # Allocate bonus shares to high-weight slices, prioritizing those not fully attaining
        bonus_order = sorted(
            range(n),
            key=lambda k: (-W[k], attain[k], -gap[k])
        )
        
        for k in bonus_order:
            if remaining <= 0:
                break
            # Allocate 1-4 extra shares based on weight and gap
            max_bonus = min(4, remaining)
            shares[k] += max_bonus
            remaining -= max_bonus
    
    # --- PHASE 4: Churn control with hysteresis ---
    prev = state.get("prev")
    if prev is not None:
        churn = sum(1 for k in range(n) if shares[k] != prev[k])
        
        # Strong hysteresis: only change if churn is modest AND we gain significant benefit
        if churn >= 2:
            # Evaluate benefit of change
            prev_delivered = [prev[k] * rate[k] for k in range(n)]
            new_delivered = [shares[k] * rate[k] for k in range(n)]
            
            prev_score = sum(
                W[k] * (1.0 if prev_delivered[k] >= T[k] - 0.5 else 0.0)
                for k in range(n)
            )
            new_score = sum(
                W[k] * (1.0 if new_delivered[k] >= T[k] - 0.5 else 0.0)
                for k in range(n)
            )
            
            # Only switch if new score is meaningfully better
            benefit = new_score - prev_score
            churn_cost = 0.02 * churn
            
            if benefit < churn_cost + 0.1:
                shares = list(prev)
    
    state.set("prev", list(shares))
    
    return shares