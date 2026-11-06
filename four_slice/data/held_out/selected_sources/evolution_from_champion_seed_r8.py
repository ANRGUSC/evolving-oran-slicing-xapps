import math

TAU = (9.0, 9.0, 11.0, 11.0)

def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578

def controller(history, state, sla, predictors):
    # Track the previous allocation to compute churn
    prev_shares = state.get('prev_shares', None)
    
    # Collect current CQI, rate, and compute deficit/surplus for each slice
    cqis = []
    rates = []
    attainments = []
    deficits = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        cqis.append(cqi)
        
        rate = _rate_of_cqi(cqi)
        rates.append(rate)
        
        # Current delivered throughput
        current_mbps = h.dl_bytes * 8 / 1e6 if h is not None else 0.0
        tgt = sla.slices[k].target_mbps
        
        attain = 1.0 if current_mbps >= tgt - 0.5 else 0.0
        attainments.append(attain)
        
        # Deficit: how much extra throughput is needed
        deficit = max(0.0, tgt - current_mbps)
        deficits.append(deficit)
    
    # Strategy: prioritize slices with high weight and high deficit, 
    # but only if their CQI is above threshold (capable)
    
    weights = [sla.slices[k].weight for k in range(sla.n)]
    
    # Build priority list: (index, priority_score)
    # Priority = weight * deficit_indicator, but only for slices above threshold
    priorities = []
    for k in range(sla.n):
        if cqis[k] >= TAU[k] and rates[k] > 0:
            # High priority if weight is high and deficit is high
            priority = weights[k] * (1.0 + deficits[k] / max(1.0, sla.slices[k].target_mbps))
            priorities.append((k, priority))
        elif cqis[k] < TAU[k]:
            # Low priority: below threshold, hard to meet target
            priorities.append((k, 0.0))
    
    # Sort by priority (descending)
    priorities.sort(key=lambda x: -x[1])
    
    # Allocate greedily: give each slice just enough to meet target, 
    # then distribute remainder by weight
    out = [0] * sla.n
    remaining_cap = sla.guaranteed_cap
    
    # First pass: allocate minimum needed for each slice above threshold
    for k in range(sla.n):
        if cqis[k] >= TAU[k] and rates[k] > 0:
            tgt = sla.slices[k].target_mbps
            min_share = max(1, int(math.ceil(tgt / rates[k])))
            
            if min_share <= remaining_cap:
                out[k] = min_share
                remaining_cap -= min_share
    
    # Second pass: distribute remaining capacity by weight among those above threshold
    above_threshold = [
        k for k in range(sla.n) 
        if cqis[k] >= TAU[k] and rates[k] > 0
    ]
    
    if above_threshold and remaining_cap > 0:
        total_weight = sum(weights[k] for k in above_threshold)
        
        # Distribute remaining capacity proportionally by weight
        for k in above_threshold:
            extra = int((weights[k] / total_weight) * remaining_cap)
            out[k] += extra
        
        # Handle rounding: assign any leftover
        if sum(out) < sla.guaranteed_cap:
            leftover = sla.guaranteed_cap - sum(out)
            # Give to highest-weight slice above threshold
            if above_threshold:
                best_k = max(above_threshold, key=lambda k: weights[k])
                out[best_k] += leftover
    elif remaining_cap > 0 and not above_threshold:
        # All slices below threshold; distribute equally
        per_slice = remaining_cap // sla.n
        remainder = remaining_cap % sla.n
        for k in range(sla.n):
            out[k] += per_slice
            if k < remainder:
                out[k] += 1
    
    # Clamp and normalize
    out = [max(0, s) for s in out]
    total = sum(out)
    if total > sla.guaranteed_cap:
        excess = total - sla.guaranteed_cap
        for k in range(sla.n - 1, -1, -1):
            if out[k] > 0:
                reduction = min(out[k], excess)
                out[k] -= reduction
                excess -= reduction
                if excess == 0:
                    break
    
    state.set('prev_shares', out)
    
    return out