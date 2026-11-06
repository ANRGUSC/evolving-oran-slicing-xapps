import math

TAU = (9.0, 9.0, 11.0, 11.0)

def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578


def controller(history, state, sla, predictors):
    """Improved controller with better attainment focus and reduced churn."""
    
    # Initialize state tracking
    if state.get('prev_shares') is None:
        state.set('prev_shares', [0, 0, 0, 0])
    if state.get('channel_history') is None:
        state.set('channel_history', [[] for _ in range(sla.n)])
    
    prev_shares = state.get('prev_shares', [0, 0, 0, 0])
    channel_history = state.get('channel_history', [[] for _ in range(sla.n)])
    
    # Collect current state
    cqis = []
    rates = []
    targets = []
    attainments = []
    recent_mbps = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        cqis.append(cqi)
        
        # Track channel history for trend detection (last 5 samples)
        channel_history[k].append(cqi)
        if len(channel_history[k]) > 5:
            channel_history[k].pop(0)
        
        rate = _rate_of_cqi(cqi)
        rates.append(rate)
        tgt = sla.slices[k].target_mbps
        targets.append(tgt)
        
        # Get recent delivery
        mbps = history.slices[k].latest_mbps()
        recent_mbps.append(mbps if mbps is not None else 0)
        attain = 1 if (mbps is not None and mbps >= tgt - 0.5) else 0
        attainments.append(attain)
    
    # Detect channel trends: improving vs degrading
    channel_trends = []
    for k in range(sla.n):
        if len(channel_history[k]) >= 2:
            trend = channel_history[k][-1] - channel_history[k][-2]
        else:
            trend = 0
        channel_trends.append(trend)
    
    # Compute minimum viable shares needed for target attainment
    min_shares = []
    for k in range(sla.n):
        tgt = targets[k]
        rate = rates[k]
        # Lower threshold for slices trending upward (channel improving)
        effective_tau = TAU[k] - (0.5 if channel_trends[k] > 0.5 else 0)
        
        if cqis[k] >= effective_tau and rate > 0.01:
            min_share = math.ceil(tgt / rate)
            min_shares.append(min(min_share, sla.guaranteed_cap))
        else:
            min_shares.append(0)
    
    total_min = sum(min_shares)
    
    # Primary allocation: guarantee all feasible targets
    if total_min <= sla.guaranteed_cap:
        # Start with minimum viable
        out = list(min_shares)
        remaining = sla.guaranteed_cap - total_min
        
        # Distribute extra capacity based on: weight, attainment status, and channel quality
        # Prioritize helping slices with good channels that aren't yet attaining
        extra_priority = []
        for k in range(sla.n):
            w = sla.slices[k].weight
            
            # Strong priority: unmet targets in good channels
            if attainments[k] == 0 and cqis[k] >= TAU[k]:
                # How much marginal gain could this slice get?
                deficit = max(0, targets[k] - recent_mbps[k])
                marginal_share_value = deficit / max(0.1, rates[k])
                # Weighted priority with deficit emphasis
                priority = w * (1.5 + 0.5 * min(1.0, marginal_share_value / 5.0))
                extra_priority.append((priority, k, 'unmet_good'))
            
            # Medium priority: already attaining, weight only
            elif attainments[k] == 1 and cqis[k] >= TAU[k] - 1.0:
                priority = w * 0.3
                extra_priority.append((priority, k, 'attaining'))
            
            # Low priority: channel degrading, reduce allocation
            elif channel_trends[k] < -1.0:
                priority = w * 0.1
                extra_priority.append((priority, k, 'degrading'))
        
        extra_priority.sort(reverse=True, key=lambda x: x[0])
        
        # Allocate extra in larger chunks to avoid excessive churn
        for priority, k, category in extra_priority:
            if remaining <= 0:
                break
            
            if category == 'unmet_good':
                # Allocate more aggressively for unmet targets
                extra = min(remaining, max(2, min(8, (sla.guaranteed_cap - out[k]) // 2)))
            elif category == 'attaining':
                # Very conservative for already attaining
                extra = min(remaining, 1)
            else:
                # Minimal for degrading
                extra = 0
            
            out[k] += extra
            remaining -= extra
        
        # Distribute final remainder by weight only
        if remaining > 0:
            weight_order = sorted(range(sla.n), key=lambda k: sla.slices[k].weight, reverse=True)
            for idx in range(remaining):
                out[weight_order[idx % sla.n]] += 1
    
    else:
        # Oversubscribed: prioritize attainment and weight
        # Give strong boost to attaining slices to maintain their attainment
        priorities = []
        for k in range(sla.n):
            w = sla.slices[k].weight
            # Boost attaining slices heavily, degrade non-attaining
            attain_boost = 3.0 if attainments[k] else 0.7
            # Also consider channel quality as secondary factor
            cqi_factor = 0.8 + 0.2 * min(1.0, (cqis[k] - 5.0) / 10.0)
            priority = w * attain_boost * cqi_factor
            priorities.append(priority)
        
        total_priority = sum(priorities)
        if total_priority > 0:
            out = [max(0, int(sla.guaranteed_cap * priorities[k] / total_priority)) for k in range(sla.n)]
        else:
            out = [sla.guaranteed_cap // sla.n] * sla.n
    
    # Clamp and normalize
    out = [max(0, x) for x in out]
    total = sum(out)
    if total > sla.guaranteed_cap:
        scale = sla.guaranteed_cap / total
        out = [int(x * scale) for x in out]
        deficit = sla.guaranteed_cap - sum(out)
        # Distribute remainder to highest weight * attainment
        if deficit > 0:
            remainder_priority = [(sla.slices[k].weight * (2.0 if attainments[k] else 1.0), k) for k in range(sla.n)]
            remainder_priority.sort(reverse=True)
            for idx in range(deficit):
                _, k = remainder_priority[idx % sla.n]
                out[k] += 1
    
    # Churn minimization: only change share if necessary
    smoothed_out = list(out)
    for k in range(sla.n):
        delta = abs(out[k] - prev_shares[k])
        
        # Stay put if: attaining target, channel stable, and change is small
        if attainments[k] == 1 and abs(channel_trends[k]) < 0.5 and delta <= 2:
            smoothed_out[k] = prev_shares[k]
        
        # Stay put if: channel improving (trending up), attainment was close, change is small
        elif channel_trends[k] > 1.0 and attainments[k] == 0 and recent_mbps[k] >= targets[k] - 2.0 and delta <= 2:
            smoothed_out[k] = prev_shares[k]
        
        # Force change only if: channel dropped significantly below threshold
        elif channel_trends[k] < -1.5 or cqis[k] < TAU[k] - 2.0:
            pass  # Allow the change
    
    out = smoothed_out
    
    # Final validation and clamp
    out = [max(0, min(sla.guaranteed_cap, x)) for x in out]
    total = sum(out)
    if total > sla.guaranteed_cap:
        overage = total - sla.guaranteed_cap
        for _ in range(overage):
            # Remove from lowest-weight slices that are attaining
            removal_priority = []
            for k in range(sla.n):
                if out[k] > 0:
                    # Prefer removing from attaining low-weight slices
                    penalty = -sla.slices[k].weight if attainments[k] else sla.slices[k].weight
                    removal_priority.append((penalty, k))
            removal_priority.sort()
            if removal_priority:
                _, k = removal_priority[0]
                out[k] = max(0, out[k] - 1)
    
    # Update state
    state.set('prev_shares', out)
    state.set('channel_history', channel_history)
    
    return out