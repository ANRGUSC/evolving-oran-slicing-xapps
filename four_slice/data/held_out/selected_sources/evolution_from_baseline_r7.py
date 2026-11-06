import math

# CQI thresholds for each slice
TAU = (7.5, 7.5, 8.5, 8.5)

# Hysteresis and sharing parameters
HYSTERESIS = 0.3
MIN_SHARE = 1


def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate (calibrated curve)."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578


def _compute_demand(target_mbps, rate):
    """Compute minimum share needed to meet target."""
    if rate <= 0:
        return float('inf')
    return target_mbps / rate


def controller(history, state, sla, predictors):
    """
    Enhanced controller optimizing for attainment + best-effort balance.
    
    Strategy:
    1. Track per-slice attainment history and adapt allocation strategy.
    2. Prioritize meeting targets for slices with sufficient channel quality.
    3. Aggressively allocate to high-weight slices to maximize score.
    4. Minimize churn through selective hysteresis while maintaining responsiveness.
    5. Use remaining capacity to boost best-effort (avoid over-allocation to guaranteed).
    """
    
    # Initialize state tracking
    if not state.get('prev_shares'):
        state.set('prev_shares', [0, 0, 0, 0])
    if not state.get('attain_history'):
        state.set('attain_history', [[0]*5 for _ in range(4)])
    
    prev_shares = state.get('prev_shares', [0, 0, 0, 0])
    attain_history = state.get('attain_history', [[0]*5 for _ in range(4)])
    
    # Gather current CQI, rates, delivered throughput, and compute metrics
    cqis = []
    rates = []
    demands = []
    attainable = []
    delivered = []
    recent_attain = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        cqis.append(cqi)
        
        rate = _rate_of_cqi(cqi)
        rates.append(rate)
        
        target = sla.slices[k].target_mbps
        demand = _compute_demand(target, rate)
        demands.append(demand)
        
        attainable.append(demand <= sla.guaranteed_cap)
        
        # Track delivered throughput for attainment calculation
        delivered_mbps = history.slices[k].latest_mbps()
        delivered.append(delivered_mbps)
        
        # Attainment: 1 if delivered >= target - 0.5, else 0
        is_attained = 1 if delivered_mbps >= target - 0.5 else 0
        recent_attain.append(is_attained)
        
        # Update rolling history (keep last 5 samples)
        attain_history[k] = (attain_history[k] + [is_attained])[-5:]
    
    state.set('attain_history', attain_history)
    
    # Compute mean attainment for each slice over recent history
    mean_attain = [sum(h) / len(h) if h else 0 for h in attain_history]
    
    # Phase 1: Allocate to meet targets for high-priority slices with good channels
    allocations = [0, 0, 0, 0]
    total_assigned = 0
    
    # Priority order: (weight, attainment_deficit, slice_index)
    # Favor high-weight slices that are struggling to meet targets
    priority = []
    for k in range(sla.n):
        weight = sla.slices[k].weight
        deficit = 1.0 - mean_attain[k]  # Higher deficit = more priority
        priority.append((weight, deficit, k))
    
    # Sort by weight first (descending), then by deficit (descending)
    priority.sort(key=lambda x: (-x[0], -x[1]))
    
    # Allocate minimum viable shares to slices with acceptable channels
    for weight, deficit, k in priority:
        if attainable[k] and cqis[k] >= TAU[k]:
            min_share = max(MIN_SHARE, int(math.ceil(demands[k])))
            if total_assigned + min_share <= sla.guaranteed_cap:
                allocations[k] = min_share
                total_assigned += min_share
    
    # Phase 2: Distribute remaining capacity to improve attainment
    remaining = sla.guaranteed_cap - total_assigned
    
    if remaining > 1:
        # Allocate to slices that are close to attainment but need more shares
        candidates = []
        for k in range(sla.n):
            if cqis[k] >= TAU[k] - 0.5 and allocations[k] < sla.guaranteed_cap:
                # Score by: (1) currently not attained, (2) weight
                score = (1.0 - recent_attain[k]) * sla.slices[k].weight
                candidates.append((score, sla.slices[k].weight, k))
        
        candidates.sort(reverse=True)
        
        # Allocate remaining in chunks, respecting cap
        for score, weight, k in candidates:
            if remaining <= 0:
                break
            boost = min(remaining, max(1, int(remaining * 0.3)))
            allocations[k] = min(allocations[k] + boost, sla.guaranteed_cap)
            remaining -= boost
    
    # Phase 3: Apply hysteresis selectively to reduce churn
    for k in range(sla.n):
        if prev_shares[k] > 0 and allocations[k] > 0:
            # Apply hysteresis only if change is small and attainment is stable
            diff = abs(allocations[k] - prev_shares[k])
            if diff <= 2 and mean_attain[k] >= 0.5:
                allocations[k] = prev_shares[k]
    
    # Phase 4: Final normalization
    total = sum(allocations)
    if total > sla.guaranteed_cap:
        # Scale down proportionally, but preserve relative priorities
        scale = sla.guaranteed_cap / total
        allocations = [max(0, int(a * scale)) for a in allocations]
        shortfall = sla.guaranteed_cap - sum(allocations)
        
        # Redistribute shortfall to highest-priority slices
        if shortfall > 0:
            high_priority = sorted(range(sla.n), key=lambda k: -sla.slices[k].weight)
            for k in high_priority:
                if shortfall <= 0:
                    break
                add = min(shortfall, 1)
                allocations[k] += add
                shortfall -= add
    
    # Store for next iteration
    state.set('prev_shares', allocations)
    
    return allocations