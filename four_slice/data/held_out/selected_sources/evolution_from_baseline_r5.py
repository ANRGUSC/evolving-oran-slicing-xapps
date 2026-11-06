import math

TAU = (9.0, 9.0, 11.0, 11.0)

def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578

def controller(history, state, sla, predictors):
    # Initialize state tracking
    if not state.get('prev_shares'):
        state.set('prev_shares', [0, 0, 0, 0])
    
    prev_shares = state.get('prev_shares', [0, 0, 0, 0])
    
    # Gather current CQI, rates, and attainment status for all slices
    slices_info = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        tgt = sla.slices[k].target_mbps
        rate = _rate_of_cqi(cqi)
        
        # Minimum share needed to meet target
        min_share = int(math.ceil(tgt / rate)) if rate > 0.01 else sla.guaranteed_cap
        min_share = max(1, min(min_share, sla.guaranteed_cap))
        
        # Current delivered throughput
        delivered = h.dl_bytes * 8 / 1e6 if h is not None else 0.0
        is_attaining = delivered >= (tgt - 0.5)
        
        slices_info.append({
            'k': k,
            'cqi': cqi,
            'rate': rate,
            'target': tgt,
            'min_share': min_share,
            'delivered': delivered,
            'is_attaining': is_attaining,
            'weight': sla.slices[k].weight,
            'prb_share': h.prb_share if h is not None else 0,
            'tau': TAU[k]
        })
    
    out = [0] * sla.n
    
    # Strategy: weighted allocation favoring high-weight slices and those not attaining
    # Key insight: maximize score by prioritizing weight and attainment status
    
    # Step 1: Calculate priority scores for each slice
    # Higher priority = more urgent (high weight, not attaining, good channel)
    for info in slices_info:
        attainment_urgency = 1.0 if not info['is_attaining'] else 0.3
        channel_quality = max(0.0, (info['cqi'] - 5.0) / 10.0)
        info['priority'] = info['weight'] * attainment_urgency * (1.0 + 0.5 * channel_quality)
    
    # Step 2: Sort by priority (descending)
    priority_list = sorted(slices_info, key=lambda x: -x['priority'])
    
    # Step 3: Allocate budget using weighted proportional allocation
    # Start with a base allocation for all slices that can use it
    remaining_budget = sla.guaranteed_cap
    allocated = [0] * sla.n
    
    # First: guarantee minimum share for slices above CQI threshold
    for info in priority_list:
        k = info['k']
        if info['cqi'] >= info['tau']:
            share_to_allocate = min(info['min_share'], remaining_budget)
            allocated[k] = share_to_allocate
            remaining_budget -= share_to_allocate
    
    # Second: distribute remaining budget by priority among all slices above threshold
    slices_above_tau = [info for info in slices_info if info['cqi'] >= info['tau']]
    
    if remaining_budget > 0 and slices_above_tau:
        # Weight by priority * weight for secondary allocation
        total_priority = sum(info['priority'] for info in slices_above_tau)
        
        if total_priority > 0:
            for info in slices_above_tau:
                k = info['k']
                share_fraction = info['priority'] / total_priority
                bonus_share = int(remaining_budget * share_fraction)
                allocated[k] += bonus_share
    
    # Step 4: Ensure sum doesn't exceed guaranteed_cap and apply fractional scaling
    total_allocated = sum(allocated)
    
    if total_allocated > sla.guaranteed_cap:
        scale_factor = sla.guaranteed_cap / total_allocated
        allocated = [max(0, int(s * scale_factor)) for s in allocated]
    
    # Step 5: Distribute any leftover PRBs to maximize score
    total_allocated = sum(allocated)
    if total_allocated < sla.guaranteed_cap:
        deficit = sla.guaranteed_cap - total_allocated
        
        # Give leftovers to highest-weight non-attaining slices above threshold
        candidates = [
            info for info in slices_info
            if not info['is_attaining'] and info['cqi'] >= info['tau']
        ]
        
        if not candidates:
            candidates = [
                info for info in slices_info
                if info['cqi'] >= info['tau']
            ]
        
        candidates.sort(key=lambda x: -x['weight'])
        
        for info in candidates:
            if deficit <= 0:
                break
            k = info['k']
            allocated[k] += 1
            deficit -= 1
    
    # Step 6: Final clamp and normalization
    out = [max(0, min(s, sla.guaranteed_cap)) for s in allocated]
    
    # Ensure sum constraint
    total_allocated = sum(out)
    if total_allocated > sla.guaranteed_cap:
        # Scale down proportionally, but prefer keeping allocations to high-weight slices
        sorted_indices = sorted(range(sla.n), key=lambda k: -sla.slices[k].weight)
        
        while total_allocated > sla.guaranteed_cap:
            for k in sorted_indices:
                if out[k] > 0:
                    out[k] -= 1
                    total_allocated -= 1
                    if total_allocated <= sla.guaranteed_cap:
                        break
    
    # Track shares for churn penalty
    state.set('prev_shares', out)
    
    return out