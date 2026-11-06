import math

TAU = (9.0, 9.0, 11.0, 11.0)

def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578

def controller(history, state, sla, predictors):
    # Track the previous allocation to compute churn
    prev_shares = state.get('prev_shares', None)
    
    # Collect current CQI and compute the minimum required share for each slice
    cqis = []
    min_shares = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        cqis.append(cqi)
        
        tgt = sla.slices[k].target_mbps
        rate = _rate_of_cqi(cqi)
        
        # Minimum share to meet target (at least 1 if cqi is decent)
        if rate > 0 and cqi >= TAU[k]:
            min_share = int(math.ceil(tgt / rate))
        else:
            min_share = 0
        
        min_shares.append(min_share)
    
    # Compute total minimum needed
    total_min = sum(min_shares)
    
    if total_min <= sla.guaranteed_cap:
        # We can afford all minimum allocations; allocate and distribute slack
        out = list(min_shares)
        slack = sla.guaranteed_cap - total_min
        
        # Distribute slack proportionally to weights for slices above threshold
        # to maximize score (higher weight slices get priority)
        above_threshold = [
            (k, sla.slices[k].weight)
            for k in range(sla.n)
            if cqis[k] >= TAU[k]
        ]
        
        if above_threshold and slack > 0:
            total_weight = sum(w for _, w in above_threshold)
            for k, w in above_threshold:
                additional = int(math.ceil((w / total_weight) * slack))
                out[k] = min(out[k] + additional, sla.guaranteed_cap)
            
            # Re-normalize if we overallocated
            if sum(out) > sla.guaranteed_cap:
                excess = sum(out) - sla.guaranteed_cap
                for k in range(sla.n - 1, -1, -1):
                    if out[k] > min_shares[k]:
                        reduction = min(out[k] - min_shares[k], excess)
                        out[k] -= reduction
                        excess -= reduction
                        if excess == 0:
                            break
    else:
        # Constrained: distribute the cap among slices proportionally by weight,
        # but prioritize those with CQI above threshold
        weights = [sla.slices[k].weight for k in range(sla.n)]
        above_threshold_weight = sum(
            weights[k] for k in range(sla.n) if cqis[k] >= TAU[k]
        )
        
        if above_threshold_weight > 0:
            # Allocate cap proportionally among slices above threshold
            out = []
            for k in range(sla.n):
                if cqis[k] >= TAU[k]:
                    share = int(math.ceil(
                        (weights[k] / above_threshold_weight) * sla.guaranteed_cap
                    ))
                    out.append(share)
                else:
                    out.append(0)
            
            # Normalize
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
        else:
            # No slice above threshold; equal distribution (fallback)
            out = [sla.guaranteed_cap // sla.n] * sla.n
    
    # Clamp negatives and prepare for next iteration
    out = [max(0, s) for s in out]
    state.set('prev_shares', out)
    
    return out