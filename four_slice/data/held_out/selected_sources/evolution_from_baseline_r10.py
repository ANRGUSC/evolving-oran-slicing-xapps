import math

TAU = (9.0, 9.0, 11.0, 11.0)


def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578


def controller(history, state, sla, predictors):
    # Initialize state
    if not state.get('initialized', False):
        state.set('initialized', True)
        state.set('last_shares', [0, 0, 0, 0])
        state.set('tick_count', 0)
        state.set('cqi_history', [[] for _ in range(sla.n)])
        state.set('attainment_history', [[] for _ in range(sla.n)])
        state.set('rate_history', [[] for _ in range(sla.n)])
    
    tick = state.get('tick_count', 0)
    state.set('tick_count', tick + 1)
    
    # Gather current slice info
    slices_info = []
    cqi_history = state.get('cqi_history', [[] for _ in range(sla.n)])
    attainment_history = state.get('attainment_history', [[] for _ in range(sla.n)])
    rate_history = state.get('rate_history', [[] for _ in range(sla.n)])
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        rate = _rate_of_cqi(cqi)
        target = sla.slices[k].target_mbps
        weight = sla.slices[k].weight
        
        # Track history
        cqi_history[k].append(cqi)
        if len(cqi_history[k]) > 30:
            cqi_history[k].pop(0)
        
        rate_history[k].append(rate)
        if len(rate_history[k]) > 30:
            rate_history[k].pop(0)
        
        # Trend analysis: average recent rate vs older rate
        recent_rate = sum(rate_history[k][-5:]) / len(rate_history[k][-5:]) if len(rate_history[k]) >= 5 else rate
        older_rate = sum(rate_history[k][-15:-5]) / 10 if len(rate_history[k]) >= 15 else recent_rate
        rate_trend = recent_rate - older_rate
        
        # Minimum share needed to meet target
        min_share = int(math.ceil(target / rate)) if rate > 0 else sla.guaranteed_cap
        min_share = min(max(min_share, 0), sla.guaranteed_cap)
        
        # Comfortable share (1.3x target for buffer against variations)
        comfortable_share = int(math.ceil(1.3 * target / rate)) if rate > 0 else sla.guaranteed_cap
        comfortable_share = min(max(comfortable_share, 0), sla.guaranteed_cap)
        
        slices_info.append({
            'k': k,
            'cqi': cqi,
            'rate': rate,
            'rate_trend': rate_trend,
            'target': target,
            'weight': weight,
            'min_share': min_share,
            'comfortable_share': comfortable_share,
            'attainable': cqi >= TAU[k],
            'recent_attainment': sum(attainment_history[k][-10:]) / len(attainment_history[k][-10:]) if attainment_history[k] else 0.0
        })
    
    state.set('cqi_history', cqi_history)
    state.set('rate_history', rate_history)
    
    # Separate slices
    attainable = [s for s in slices_info if s['attainable']]
    non_attainable = [s for s in slices_info if not s['attainable']]
    
    # Sort attainable by priority: weight and recent attainment
    # Prioritize high-weight slices that are struggling to meet targets
    attainable.sort(key=lambda s: (s['weight'], 1.0 - s['recent_attainment']), reverse=True)
    
    shares = [0, 0, 0, 0]
    remaining_cap = sla.guaranteed_cap
    
    # Phase 1: Guarantee minimum shares for all attainable slices
    for s in attainable:
        if remaining_cap > 0:
            alloc = min(s['min_share'], remaining_cap)
            shares[s['k']] = alloc
            remaining_cap -= alloc
    
    # Phase 2: Boost toward comfortable share, prioritizing:
    #   - Higher weight
    #   - Lower recent attainment (struggling slices)
    #   - Positive rate trend (improving channels)
    for s in attainable:
        k = s['k']
        if remaining_cap > 0 and shares[k] < s['comfortable_share']:
            priority_score = s['weight'] * (1.0 - s['recent_attainment']) * (1.0 + max(s['rate_trend'], 0.0))
            boost_amount = min(s['comfortable_share'] - shares[k], remaining_cap)
            shares[k] += boost_amount
            remaining_cap -= boost_amount
    
    # Phase 3: Probe non-attainable slices showing strong improvement trends
    # Only allocate if trend is significantly positive AND we have excess capacity
    non_attainable.sort(key=lambda s: s['rate_trend'], reverse=True)
    for s in non_attainable:
        k = s['k']
        if remaining_cap >= 5 and s['rate_trend'] > 0.05:  # Stricter improvement threshold
            probe = min(5, remaining_cap)
            shares[k] = probe
            remaining_cap -= probe
    
    # Phase 4: Distribute remaining capacity, heavily weighted toward:
    #   - High-weight slices
    #   - Slices with poor recent attainment
    if remaining_cap > 0 and attainable:
        # Create weighted distribution favoring high-weight, low-attainment slices
        weighted_slices = []
        for s in attainable:
            # Weight by: slice weight * (1 - recent_attainment)
            weight_factor = s['weight'] * (1.0 - s['recent_attainment'] + 0.2)
            weighted_slices.append((s['k'], weight_factor))
        
        total_weight = sum(w for _, w in weighted_slices)
        if total_weight > 0:
            for k, weight_factor in weighted_slices:
                if remaining_cap > 0:
                    extra = int(math.ceil(remaining_cap * weight_factor / total_weight))
                    extra = min(extra, remaining_cap)
                    if extra > 0:
                        shares[k] += extra
                        remaining_cap -= extra
    
    # Ensure sum doesn't exceed guaranteed cap
    total = sum(shares)
    if total > sla.guaranteed_cap:
        # Scale down uniformly, then adjust by importance
        overage = total - sla.guaranteed_cap
        
        # Remove from non-attainable first
        for s in non_attainable:
            k = s['k']
            if overage > 0 and shares[k] > 0:
                reduction = min(shares[k], overage)
                shares[k] -= reduction
                overage -= reduction
        
        # Then remove from attainable with lowest priority
        for s in reversed(attainable):
            k = s['k']
            if overage > 0 and shares[k] > s['min_share']:
                reduction = min(shares[k] - s['min_share'], overage)
                shares[k] -= reduction
                overage -= reduction
    
    # Clamp negatives
    shares = [max(0, s) for s in shares]
    
    # Track attainment for next iteration
    for k in range(sla.n):
        delivered = history.slices[k].latest_mbps()
        target = sla.slices[k].target_mbps
        attained = 1 if delivered >= target - 0.5 else 0
        attainment_history[k].append(attained)
        if len(attainment_history[k]) > 30:
            attainment_history[k].pop(0)
    
    state.set('attainment_history', attainment_history)
    state.set('last_shares', shares)
    
    return shares