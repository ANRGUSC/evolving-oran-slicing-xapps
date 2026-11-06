"""Enhanced controller with demand-aware allocation and improved pool utilization.

Key improvements:
1. Demand-aware threshold: use CQI to predict achievable throughput vs target
2. Priority weighting: allocate more intelligently when pool is constrained
3. Dynamic reallocation: redistribute from underutilized slices to high-priority ones
4. Improved churn handling: smooth transitions with incremental adjustments
5. Best-effort awareness: ensure pool isn't wasted on slices that can't use shares
"""
import math

TAU_BASE = (8.5, 8.5, 10.0, 10.0)  # per-slice CQI thresholds
POOL_MAX = 70
POOL_TOTAL = 100


def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578


def controller(history, state, sla, predictors):
    """
    Demand-aware PRB controller with improved pool efficiency.
    Strategy:
    - Assess each slice's CQI and predicted throughput capacity
    - Identify which slices can realistically meet targets
    - Prioritize by weight * (1 - recent_attainment) to focus on struggling slices
    - Use incremental share adjustments to minimize churn
    - Reallocate unused capacity to high-priority slices
    """
    
    # Initialize state tracking
    if not state.get('prev_shares'):
        state.set('prev_shares', [0, 0, 0, 0])
    if not state.get('attainment_history'):
        state.set('attainment_history', {k: [] for k in range(sla.n)})
    if not state.get('cqi_history'):
        state.set('cqi_history', {k: [] for k in range(sla.n)})
    
    prev_shares = state.get('prev_shares', [0, 0, 0, 0])
    attain_hist = state.get('attainment_history', {k: [] for k in range(sla.n)})
    cqi_hist = state.get('cqi_history', {k: [] for k in range(sla.n)})
    
    # Collect current slice data
    slices_data = []
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        delivered_mbps = history.slices[k].latest_mbps()
        target = sla.slices[k].target_mbps
        weight = sla.slices[k].weight
        rate = _rate_of_cqi(cqi)
        
        # Track CQI history
        if k not in cqi_hist:
            cqi_hist[k] = []
        cqi_hist[k].append(cqi)
        if len(cqi_hist[k]) > 15:
            cqi_hist[k].pop(0)
        
        # Compute CQI trend (recent vs older)
        if len(cqi_hist[k]) >= 5:
            recent_cqi = sum(cqi_hist[k][-3:]) / 3.0
            older_cqi = sum(cqi_hist[k][:3]) / 3.0
            cqi_trend = recent_cqi - older_cqi
        else:
            cqi_trend = 0.0
        
        # Track attainment (with hysteresis: target - 0.5)
        attained = 1.0 if delivered_mbps >= target - 0.5 else 0.0
        if k not in attain_hist:
            attain_hist[k] = []
        attain_hist[k].append(attained)
        if len(attain_hist[k]) > 12:
            attain_hist[k].pop(0)
        
        recent_attain = sum(attain_hist[k]) / len(attain_hist[k]) if attain_hist[k] else 0.0
        
        # Predict achievable throughput with current and increased share
        predicted_mbps_current = rate * max(prev_shares[k], 1)
        predicted_mbps_2x = rate * (2 * max(prev_shares[k], 1))
        
        slices_data.append({
            'k': k,
            'cqi': cqi,
            'cqi_trend': cqi_trend,
            'rate': rate,
            'target': target,
            'weight': weight,
            'delivered': delivered_mbps,
            'attained': attained,
            'recent_attain': recent_attain,
            'prev_share': prev_shares[k],
            'predicted_current': predicted_mbps_current,
            'predicted_2x': predicted_mbps_2x,
        })
    
    # Phase 1: Classify slices by viability
    viable = []      # Can likely meet target
    struggling = []  # Below target, but potentially fixable
    hopeless = []    # Channel too poor to meet target at any share
    
    for d in slices_data:
        k = d['k']
        min_share_needed = math.ceil(d['target'] / max(d['rate'], 0.01))
        max_achievable = d['rate'] * POOL_MAX
        
        if d['cqi'] < 5.5:  # Deep fade
            hopeless.append(k)
        elif max_achievable >= d['target'] - 0.5:  # Can meet target
            if d['attained'] > 0.5:
                viable.append(k)
            else:
                struggling.append(k)
        else:
            hopeless.append(k)
    
    # Phase 2: Allocate baseline shares
    shares = [0] * sla.n
    total_used = 0
    
    # First: guarantee minimum for viable slices
    baseline_alloc = []
    for k in viable:
        d = slices_data[k]
        min_share = max(1, math.ceil(d['target'] / max(d['rate'], 0.01)))
        baseline_alloc.append((k, min_share))
    
    baseline_alloc.sort(key=lambda x: x[1])  # Allocate smallest first
    
    for k, min_share in baseline_alloc:
        if total_used + min_share <= POOL_MAX:
            shares[k] = min_share
            total_used += min_share
    
    # Second: allocate to struggling slices by priority (weight * deficit)
    struggling_priority = []
    for k in struggling:
        d = slices_data[k]
        deficit = 1.0 - d['recent_attain']
        priority_score = d['weight'] * deficit
        min_share = max(1, math.ceil(d['target'] / max(d['rate'], 0.01)))
        struggling_priority.append((priority_score, k, min_share))
    
    struggling_priority.sort(reverse=True)
    
    for priority, k, min_share in struggling_priority:
        if total_used + min_share <= POOL_MAX:
            shares[k] = min_share
            total_used += min_share
    
    # Phase 3: Redistribute surplus to highest-priority slices
    remaining_pool = POOL_MAX - total_used
    if remaining_pool > 0:
        # Sort all slices by urgency (weight * deficit)
        urgency = []
        for k in range(sla.n):
            if shares[k] > 0:  # Already got baseline
                d = slices_data[k]
                deficit = max(0, 1.0 - d['recent_attain'])
                score = d['weight'] * deficit
                urgency.append((score, k))
        
        urgency.sort(reverse=True)
        
        # Distribute remaining pool
        for score, k in urgency:
            if remaining_pool <= 0:
                break
            d = slices_data[k]
            additional = min(remaining_pool, max(1, POOL_MAX // sla.n))
            shares[k] += additional
            remaining_pool -= additional
    
    # Phase 4: Churn minimization — smooth transitions
    new_shares = shares[:]
    for k in range(sla.n):
        old_share = prev_shares[k]
        new_share = new_shares[k]
        
        # If share would drop to 0 from positive, try to keep at least 1
        if new_share == 0 and old_share > 0 and sum(new_shares) < POOL_MAX:
            new_shares[k] = 1
    
    # Clamp to pool limits
    total_new = sum(new_shares)
    if total_new > POOL_MAX:
        scale = POOL_MAX / total_new
        new_shares = [max(0, int(s * scale)) for s in new_shares]
    
    shares = [max(0, s) for s in new_shares]
    
    # Final safety check: ensure sum <= POOL_MAX
    while sum(shares) > POOL_MAX:
        max_idx = shares.index(max(shares))
        shares[max_idx] -= 1
    
    # Update state
    state.set('prev_shares', shares)
    state.set('attainment_history', attain_hist)
    state.set('cqi_history', cqi_hist)
    
    return shares