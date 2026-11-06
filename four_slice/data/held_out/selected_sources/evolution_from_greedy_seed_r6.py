import math


def controller(history, state, sla, predictors):
    """Enhanced controller with improved attainment targeting and dynamic reallocation.
    
    Key improvements:
    1. Explicitly track per-slice attainment over sliding window (last 10 ticks)
    2. Reallocate shares from consistently underperforming slices to high-weight overachievers
    3. Use CQI stability (variance) to adjust aggressiveness of reallocation
    4. Implement smart margin scaling: tighter for high-CQI slices, looser for degraded channels
    5. Prioritize weight-normalized performance: maximize w_k * attain_k contributions
    6. Reduce churn through selective matching to previous allocation
    """
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap

    def eff(cqi):
        c = max(5.0, min(15.0, cqi))
        return 0.30 + 2.09 * ((c - 5.0) / 10.0) ** 1.578

    # Current observations
    obs = [history.slices[k].last().cqi if history.slices[k].last() else 15.0 for k in range(n)]
    
    # Track CQI history for stability and trend analysis
    cqi_history = state.get("cqi_history", [])
    cqi_history.append(obs)
    if len(cqi_history) > 10:
        cqi_history = cqi_history[-10:]
    state.set("cqi_history", cqi_history)
    
    # Compute per-slice CQI stability (variance) and trend
    cqi_stability = []
    cqi_trend = []
    for k in range(n):
        cqis = [h[k] for h in cqi_history]
        if len(cqis) >= 2:
            avg_cqi = sum(cqis) / len(cqis)
            variance = sum((c - avg_cqi) ** 2 for c in cqis) / len(cqis)
            stability = 1.0 / (1.0 + variance)  # higher = more stable
            cqi_stability.append(stability)
            trend = cqis[-1] - cqis[0]  # overall trend
            cqi_trend.append(trend)
        else:
            cqi_stability.append(0.8)
            cqi_trend.append(0.0)
    
    # Adaptive margin: stable high-CQI slices get tighter margins; unstable/degraded get looser
    margins = []
    for k in range(n):
        base_margin = 0.90
        # Adjust based on CQI level and stability
        if obs[k] >= 13.0:
            base_margin *= 0.97  # tight margin for strong channels
        elif obs[k] <= 7.0:
            base_margin *= 1.08  # loose margin for weak channels
        
        # Further adjust by stability
        base_margin *= (0.95 + 0.10 * cqi_stability[k])
        
        margins.append(min(1.0, max(0.80, base_margin)))
    
    rate = [max(0.05, eff(obs[k]) * margins[k]) for k in range(n)]
    
    # Minimum share needed to meet target
    need = [max(0, math.ceil((T[k] - 0.5) / rate[k])) for k in range(n)]
    
    # Track attainment history (sliding window of last 10 ticks)
    attain_history = state.get("attain_history", {k: [] for k in range(n)})
    
    # Update attainment from current observations
    current_attain = []
    for k in range(n):
        if history.slices[k].last():
            delivered = history.slices[k].latest_mbps()
            attain = 1.0 if delivered >= T[k] - 0.5 else 0.0
            current_attain.append(attain)
            if k not in attain_history:
                attain_history[k] = []
            attain_history[k].append(attain)
            if len(attain_history[k]) > 10:
                attain_history[k] = attain_history[k][-10:]
        else:
            current_attain.append(0.0)
    state.set("attain_history", attain_history)
    
    # Compute recent attainment rate per slice
    recent_attain_rate = []
    for k in range(n):
        if attain_history[k]:
            rate_k = sum(attain_history[k]) / len(attain_history[k])
        else:
            rate_k = 0.0
        recent_attain_rate.append(rate_k)
    
    # Stage 1: allocate minimum needs in descending weight order
    shares = [0] * n
    used = 0
    ordered = sorted(range(n), key=lambda k: -W[k])
    
    for k in ordered:
        if used + need[k] <= cap:
            shares[k] = need[k]
            used += need[k]
    
    # Stage 2: reallocation strategy
    # Identify high-weight slices with strong recent attainment (overachievers)
    # and low-weight slices with weak attainment (underachievers)
    # Reallocate from underachievers to overachievers if it improves weighted score
    
    remaining = cap - used
    
    # Build list of slices: (weight, attain_rate, current_share, index)
    slice_profile = []
    for k in range(n):
        slice_profile.append((W[k], recent_attain_rate[k], shares[k], k))
    
    # High-weight, high-attainment slices that could absorb more
    overachievers = sorted(
        [(W[k], recent_attain_rate[k], k) for k in range(n) 
         if shares[k] > 0 and recent_attain_rate[k] >= 0.6],
        key=lambda x: -x[0]  # prioritize by weight
    )
    
    # Low-weight, low-attainment slices we might reduce
    underachievers = sorted(
        [(W[k], recent_attain_rate[k], k) for k in range(n) 
         if shares[k] > need[k] and recent_attain_rate[k] < 0.4],
        key=lambda x: x[0]  # reduce lowest-weight first
    )
    
    # Reallocate from underachievers to overachievers
    realloc_available = 0
    for w, att, k in underachievers:
        # Try to reduce underachiever by 1-2 shares
        reduction = min(2, max(1, shares[k] - need[k]))
        shares[k] -= reduction
        realloc_available += reduction
    
    remaining += realloc_available
    
    # Distribute remaining to overachievers
    for w, att, k in overachievers:
        if remaining <= 0:
            break
        boost = min(3, remaining)
        shares[k] += boost
        remaining -= boost
    
    # Stage 3: boost slices close to target but not quite meeting it
    for k in range(n):
        if shares[k] > 0 and remaining > 0:
            current_delivery = shares[k] * rate[k]
            if T[k] - 1.0 < current_delivery < T[k] - 0.5:
                # Very close; add 1 share to push over
                boost = min(1, remaining)
                shares[k] += boost
                remaining -= boost
    
    # Stage 4: soft churn control with attainment awareness
    prev_shares = state.get("prev_shares")
    prev_attain_score = state.get("prev_attain_score", 0.0)
    
    if prev_shares is not None:
        churn = sum(1 for k in range(n) if shares[k] != prev_shares[k])
        
        # Compute current weighted attainment score
        current_score = sum(W[k] * current_attain[k] for k in range(n))
        
        # If churn is low and we were doing well, consider keeping previous
        if churn <= 1 and prev_attain_score >= current_score * 0.95:
            shares = list(prev_shares)
        else:
            state.set("prev_attain_score", current_score)
    else:
        current_score = sum(W[k] * current_attain[k] for k in range(n))
        state.set("prev_attain_score", current_score)
    
    state.set("prev_shares", list(shares))
    
    return shares