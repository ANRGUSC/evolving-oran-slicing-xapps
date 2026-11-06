import math

TAU = (9.0, 9.0, 11.0, 11.0)

def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578

def controller(history, state, sla, predictors):
    # Track the previous allocation to compute churn
    prev_shares = state.get('prev_shares', None)
    
    # Collect current CQI, rates, and compute the minimum required share for each slice
    cqis = []
    rates = []
    min_shares = []
    attainable = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        cqis.append(cqi)
        
        rate = _rate_of_cqi(cqi)
        rates.append(rate)
        
        tgt = sla.slices[k].target_mbps
        
        # Check if target is attainable at all
        can_attain = (rate > 0.01 and cqi >= TAU[k])
        attainable.append(can_attain)
        
        # Minimum share to meet target
        if can_attain:
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
        
        if slack > 0:
            # Prioritize weight and attainability for slack distribution
            # Higher weight + attainable slices get priority
            candidates = [
                (k, sla.slices[k].weight)
                for k in range(sla.n)
                if attainable[k]
            ]
            
            if candidates:
                total_weight = sum(w for _, w in candidates)
                # Distribute slack proportionally to weights
                for k, w in candidates:
                    additional = int((w / total_weight) * slack)
                    out[k] += additional
                
                # Handle remainder greedily by highest weight
                remainder = slack - sum(additional for k, _ in candidates)
                candidates.sort(key=lambda x: -x[1])
                for k, _ in candidates:
                    if remainder <= 0:
                        break
                    out[k] += 1
                    remainder -= 1
            else:
                # No attainable slices; distribute equally
                per_slice = slack // sla.n
                remainder = slack % sla.n
                for k in range(sla.n):
                    out[k] += per_slice
                    if k < remainder:
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
        # Constrained: prioritize by (weight * attainability), then allocate proportionally
        # Slices that can't attain get minimal allocation
        weights = [sla.slices[k].weight for k in range(sla.n)]
        
        # Split into attainable and non-attainable
        attainable_weight = sum(weights[k] for k in range(sla.n) if attainable[k])
        
        if attainable_weight > 0:
            out = []
            attainable_indices = [k for k in range(sla.n) if attainable[k]]
            
            # Allocate to attainable slices proportionally by weight
            per_unit_weight = sla.guaranteed_cap / attainable_weight
            for k in range(sla.n):
                if attainable[k]:
                    share = int(weights[k] * per_unit_weight)
                    out.append(max(1, share))
                else:
                    out.append(0)
            
            # Normalize to cap
            total = sum(out)
            if total > sla.guaranteed_cap:
                excess = total - sla.guaranteed_cap
                # Remove from lowest weight attainable slices first
                attainable_indices.sort(key=lambda k: weights[k])
                for k in attainable_indices:
                    if excess <= 0:
                        break
                    reduction = min(out[k] - 1, excess)
                    out[k] -= reduction
                    excess -= reduction
            elif total < sla.guaranteed_cap:
                # Add remainder to highest weight attainable slices
                remainder = sla.guaranteed_cap - total
                attainable_indices.sort(key=lambda k: -weights[k])
                for k in attainable_indices:
                    if remainder <= 0:
                        break
                    out[k] += 1
                    remainder -= 1
        else:
            # No slice attainable; equal distribution
            out = [sla.guaranteed_cap // sla.n] * sla.n
            remainder = sla.guaranteed_cap % sla.n
            for k in range(remainder):
                out[k] += 1
    
    # Clamp negatives
    out = [max(0, s) for s in out]
    
    # Store for next iteration
    state.set('prev_shares', out)
    
    return out