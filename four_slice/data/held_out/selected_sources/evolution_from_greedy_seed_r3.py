import math


def controller(history, state, sla, predictors):
    """Enhanced allocation controller with adaptive targeting and demand prediction.
    
    Key improvements:
    1. Adaptive target thresholds based on channel conditions (not fixed 0.5 Mbps tolerance)
    2. Demand prediction from recent history to allocate more aggressively when sustainable
    3. Dynamic weight adjustment favoring slices with high attainment potential
    4. Relaxed churn thresholds to enable faster recovery from poor allocations
    5. Better handling of slices near saturation (high CQI) to maximize best-effort
    """
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap
    
    def eff(cqi):
        c = max(5.0, min(15.0, cqi))
        return 0.30 + 2.09 * ((c - 5.0) / 10.0) ** 1.578
    
    # Gather observations
    obs = []
    recent_mbps = []
    for k in range(n):
        last = history.slices[k].last()
        cqi = last.cqi if last else 15.0
        mbps = history.slices[k].latest_mbps()
        obs.append(cqi)
        recent_mbps.append(mbps if mbps is not None else 0.0)
    
    # Estimate per-share rates with channel-aware confidence
    rate = []
    confidence = []
    for k in range(n):
        base_rate = max(0.05, eff(obs[k]))
        
        # Channel quality affects our confidence in meeting targets
        cqi_norm = (obs[k] - 5.0) / 10.0
        conf = max(0.5, min(1.0, cqi_norm))
        confidence.append(conf)
        
        # When confidence is high and we're underperforming, be aggressive
        if recent_mbps[k] < T[k] and conf > 0.7:
            margin = 0.99
        elif recent_mbps[k] < T[k]:
            margin = 0.96
        else:
            margin = 0.93
        
        rate.append(base_rate * margin)
    
    # Adaptive tolerance: tighter when channel is good, looser when degraded
    tolerance = [2.0 if confidence[k] < 0.6 else 1.0 if confidence[k] < 0.8 else 0.3 for k in range(n)]
    
    # Calculate minimum shares needed with adaptive tolerance
    need = [max(0, math.ceil((T[k] - tolerance[k]) / rate[k])) if rate[k] > 0 else cap + 1 for k in range(n)]
    
    # Phase 1: Calculate attainment potential for each slice
    attain_potential = []
    for k in range(n):
        # How close is this slice to meeting SLA with minimal share?
        if need[k] > 0 and need[k] <= cap:
            potential = 1.0 if recent_mbps[k] >= T[k] - 0.5 else 0.5
        else:
            potential = 0.0
        attain_potential.append(potential)
    
    # Phase 2: Weighted criticality with attainment potential and gap
    criticality = []
    for k in range(n):
        gap = max(0.0, T[k] - recent_mbps[k])
        # Weight by: slice weight, attainment potential, and performance gap
        score = W[k] * (1.5 * attain_potential[k] + gap / max(1.0, T[k]))
        criticality.append((score, need[k], k))
    
    criticality.sort(reverse=True, key=lambda x: (x[0], -x[1]))
    
    # Phase 3: Greedy allocation prioritizing high-criticality slices
    shares = [0] * n
    used = 0
    
    for score, min_need, k in criticality:
        if min_need > 0 and min_need <= cap - used:
            # Allocate minimum needed for this slice
            shares[k] = min_need
            used += min_need
    
    # Phase 4: Distribute remaining budget to slices that can benefit
    if used < cap:
        remaining = cap - used
        # Prioritize high-weight, high-confidence slices
        bonus_order = sorted(
            range(n),
            key=lambda k: (-W[k], -confidence[k], -shares[k])
        )
        
        for k in bonus_order:
            if remaining <= 0:
                break
            if shares[k] > 0:  # Only boost slices already allocated
                boost = min(8, remaining)
                shares[k] += boost
                remaining -= boost
    
    # Phase 5: Enforce cap
    total = sum(shares)
    if total > cap:
        overage = total - cap
        for k in sorted(range(n), key=lambda k: -shares[k]):
            if overage <= 0:
                break
            cut = min(shares[k], overage)
            shares[k] -= cut
            overage -= cut
    
    # Phase 6: Smart churn control with adaptive thresholds
    prev = state.get("prev")
    prev_churn = state.get("churn_count", 0)
    tick_count = state.get("tick_count", 0)
    
    if prev is not None:
        churn = sum(1 for k in range(n) if shares[k] != prev[k])
        magnitude = sum(abs(shares[k] - prev[k]) for k in range(n))
        
        # Early ticks: allow more changes for recovery
        churn_threshold = 3 if tick_count < 10 else 2
        magnitude_threshold = 5 if tick_count < 10 else 3
        
        if churn <= 1:
            # Single change is always good
            pass
        elif churn <= churn_threshold and magnitude <= magnitude_threshold:
            # Small churn and magnitude: likely noise, revert
            shares = list(prev)
        elif prev_churn >= 2 and churn >= 2:
            # Recent high churn; be conservative
            # Only accept if improvement is clear
            improvement = sum(
                W[k] * (1.0 if shares[k] >= need[k] else 0.5)
                for k in range(n)
            ) - sum(
                W[k] * (1.0 if prev[k] >= need[k] else 0.5)
                for k in range(n)
            )
            if improvement < 0.2:
                shares = list(prev)
        
        state.set("churn_count", churn)
    else:
        state.set("churn_count", 0)
    
    state.set("prev", list(shares))
    state.set("tick_count", tick_count + 1)
    
    return shares