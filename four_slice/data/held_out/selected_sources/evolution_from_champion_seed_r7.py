import math

TAU = (9.0, 9.0, 11.0, 11.0)

def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578

def controller(history, state, sla, predictors):
    # Track the previous allocation to compute churn
    prev_shares = state.get('prev_shares', None)
    
    # Collect current CQI, rates, and delivered Mbps for each slice
    cqis = []
    rates = []
    delivered_mbps = []
    min_shares = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        cqis.append(cqi)
        
        rate = _rate_of_cqi(cqi)
        rates.append(rate)
        
        # Current delivered throughput
        curr_mbps = history.slices[k].latest_mbps()
        delivered_mbps.append(curr_mbps if curr_mbps is not None else 0.0)
        
        tgt = sla.slices[k].target_mbps
        
        # Minimum share to meet target
        if rate > 0 and cqi >= TAU[k]:
            min_share = max(1, int(math.ceil(tgt / rate)))
        else:
            min_share = 0
        
        min_shares.append(min_share)
    
    # Compute total minimum needed
    total_min = sum(min_shares)
    
    if total_min <= sla.guaranteed_cap:
        # We can afford all minimum allocations; allocate and distribute slack
        out = list(min_shares)
        slack = sla.guaranteed_cap - total_min
        
        # Prioritize slack distribution to slices that are underperforming
        # and have high weight, to maximize attainment and score
        candidates = []
        for k in range(sla.n):
            if cqis[k] >= TAU[k]:
                underperformance = max(0, sla.slices[k].target_mbps - delivered_mbps[k])
                # Score: higher weight, higher underperformance => higher priority
                priority = sla.slices[k].weight * (1.0 + underperformance / max(1.0, sla.slices[k].target_mbps))
                candidates.append((priority, k))
        
        candidates.sort(reverse=True)
        
        # Greedily allocate slack to highest-priority candidates
        for priority, k in candidates:
            if slack <= 0:
                break
            # Add 1 share at a time if it helps this slice reach target
            additional_needed = int(math.ceil(max(0, sla.slices[k].target_mbps - delivered_mbps[k]) / rates[k]))
            additional_alloc = min(slack, additional_needed)
            if additional_alloc > 0:
                out[k] += additional_alloc
                slack -= additional_alloc
        
        # Distribute remaining slack equally among high-weight slices above threshold
        if slack > 0:
            above_threshold = [
                k for k in range(sla.n) if cqis[k] >= TAU[k]
            ]
            if above_threshold:
                per_slice = slack // len(above_threshold)
                remainder = slack % len(above_threshold)
                for i, k in enumerate(above_threshold):
                    out[k] += per_slice
                    if i < remainder:
                        out[k] += 1
        
        # Ensure we don't exceed cap
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
        # Constrained: distribute cap with focus on high-weight slices and those
        # with good channel quality (CQI above threshold)
        weights = [sla.slices[k].weight for k in range(sla.n)]
        
        # Score each slice by weight and channel quality
        slice_scores = []
        for k in range(sla.n):
            if cqis[k] >= TAU[k]:
                # Bonus for being above threshold
                score = weights[k] * (1.0 + (cqis[k] - TAU[k]) / 6.0)
            else:
                # Below threshold: small allocation
                score = weights[k] * 0.1
            slice_scores.append((score, k))
        
        slice_scores.sort(reverse=True)
        
        # Allocate greedily by score
        out = [0] * sla.n
        remaining = sla.guaranteed_cap
        
        for score, k in slice_scores:
            if remaining <= 0:
                break
            # Allocate share for this slice
            alloc = min(remaining, max(1, min_shares[k]))
            out[k] = alloc
            remaining -= alloc
        
        # Distribute any remaining capacity to highest-weight slices
        if remaining > 0:
            weight_sorted = sorted(range(sla.n), key=lambda k: weights[k], reverse=True)
            for k in weight_sorted:
                if remaining <= 0:
                    break
                out[k] += 1
                remaining -= 1
    
    # Clamp negatives
    out = [max(0, s) for s in out]
    
    # Ensure sum doesn't exceed cap (safety)
    total_out = sum(out)
    if total_out > sla.guaranteed_cap:
        excess = total_out - sla.guaranteed_cap
        for k in range(sla.n - 1, -1, -1):
            if out[k] > 0:
                reduction = min(out[k], excess)
                out[k] -= reduction
                excess -= reduction
                if excess == 0:
                    break
    
    state.set('prev_shares', out)
    
    return out