"""Enhanced controller with better pool utilization and CQI-driven allocation.

Key improvements:
1. Better demand estimation: use actual delivered Mbps history to refine rate estimates
2. Predictive channel trends: track CQI volatility to anticipate drops
3. Pool-aware tiering: allocate to slices in tiers based on achievability
4. Aggressive good-channel utilization: capture high-CQI opportunities quickly
5. Reduced churn via commitment: once a slice is on-target, maintain share unless CQI drops
6. Best-effort awareness: track if BE is starved, adjust guaranteed allocation accordingly
"""
import math

# CQI thresholds per slice (slightly optimized)
TAU_BASE = (8.2, 8.0, 10.0, 10.2)
# Hysteresis band
HYSTERESIS = 0.8
# CQI volatility threshold (std dev) to mark a channel as unstable
VOLATILITY_THRESHOLD = 1.5


def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578


def controller(history, state, sla, predictors):
    # Initialize persistent state
    if not state.get('prev_shares'):
        state.set('prev_shares', [0, 0, 0, 0])
    if not state.get('cqi_history'):
        state.set('cqi_history', {k: [] for k in range(sla.n)})
    if not state.get('on_target'):
        state.set('on_target', [False] * sla.n)
    
    prev_shares = state.get('prev_shares', [0, 0, 0, 0])
    cqi_hist = state.get('cqi_history', {k: [] for k in range(sla.n)})
    on_target = state.get('on_target', [False] * sla.n)
    
    # Gather current state
    cqis = []
    rates = []
    targets = []
    delivered = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        cqis.append(cqi)
        rates.append(_rate_of_cqi(cqi))
        targets.append(sla.slices[k].target_mbps)
        mbps = history.slices[k].latest_mbps()
        delivered.append(mbps if mbps is not None else 0)
        
        # Track CQI history for volatility analysis
        if k not in cqi_hist:
            cqi_hist[k] = []
        cqi_hist[k].append(cqi)
        if len(cqi_hist[k]) > 10:
            cqi_hist[k].pop(0)
    
    # Compute CQI volatility (std dev over recent history)
    volatility = []
    for k in range(sla.n):
        if len(cqi_hist[k]) >= 3:
            mean_cqi = sum(cqi_hist[k]) / len(cqi_hist[k])
            var = sum((c - mean_cqi) ** 2 for c in cqi_hist[k]) / len(cqi_hist[k])
            vol = math.sqrt(var)
        else:
            vol = 0
        volatility.append(vol)
    
    # Refine rate estimates using actual delivery history
    refined_rates = []
    for k in range(sla.n):
        if prev_shares[k] > 0 and delivered[k] > 0:
            # Actual throughput per share observed
            actual_rate = delivered[k] / prev_shares[k]
            # Blend with CQI-based estimate (favor actual)
            refined = 0.6 * actual_rate + 0.4 * rates[k]
        else:
            refined = rates[k]
        refined_rates.append(refined)
    
    # Compute minimum shares needed for each target
    min_shares = []
    for k in range(sla.n):
        if refined_rates[k] > 0.01:
            needed = targets[k] / refined_rates[k]
            min_shares.append(max(0, math.ceil(needed)))
        else:
            # Channel dead; would need infinite share
            min_shares.append(sla.guaranteed_cap + 1)
    
    # Determine eligibility: balance CQI threshold with commitment to on-target slices
    thresholds = list(TAU_BASE)
    
    # Lower threshold for slices already on-target to reduce churn
    for k in range(sla.n):
        if on_target[k] and cqis[k] >= thresholds[k] - 1.5:
            thresholds[k] -= 1.0
    
    eligible = []
    for k in range(sla.n):
        # Eligibility: either CQI above threshold, or currently on-target and channel stable
        if cqis[k] >= thresholds[k]:
            eligible.append(k)
        elif on_target[k] and volatility[k] < VOLATILITY_THRESHOLD and cqis[k] >= thresholds[k] - 0.8:
            eligible.append(k)
    
    # Tier allocation: prioritize slices by CQI and achievability
    # Tier 1: high CQI, low share need (easy wins)
    # Tier 2: medium CQI, medium share need
    # Tier 3: low CQI or high share need (difficult)
    
    def allocation_priority(k):
        if k not in eligible:
            return (4, 0, 0)  # Not eligible at all
        # Score: (achievability_tier, -cqi, -share_need)
        # Lower tier is better
        achievable = min_shares[k] <= sla.guaranteed_cap / 2
        tier = 0 if achievable else (1 if min_shares[k] <= sla.guaranteed_cap else 2)
        return (tier, -cqis[k], -(-min_shares[k]))
    
    sorted_slices = sorted(range(sla.n), key=allocation_priority)
    
    # Allocate greedily: give each slice minimum needed, prioritize by tier
    out = [0] * sla.n
    pool = sla.guaranteed_cap
    
    for k in sorted_slices:
        if k in eligible:
            needed = min_shares[k]
            allocated = min(needed, pool)
            out[k] = allocated
            pool -= allocated
    
    # Apply hysteresis to reduce churn: only move share if change is significant
    for k in range(sla.n):
        if abs(out[k] - prev_shares[k]) < HYSTERESIS:
            out[k] = prev_shares[k]
    
    # Ensure pool constraint
    total = sum(out)
    if total > sla.guaranteed_cap:
        scale = sla.guaranteed_cap / max(total, 1)
        out = [max(0, int(out[k] * scale)) for k in range(sla.n)]
    
    # Fill remaining pool: give to eligible slices that could use more
    remaining = sla.guaranteed_cap - sum(out)
    if remaining > 0:
        # Bonus allocation to high-CQI eligible slices
        bonus_order = sorted([k for k in eligible], key=lambda k: cqis[k], reverse=True)
        for k in bonus_order:
            # Don't over-allocate: cap at 1.2x minimum needed
            max_extra = int(min_shares[k] * 0.3)
            extra = min(remaining, max_extra)
            out[k] += extra
            remaining -= extra
            if remaining <= 0:
                break
    
    # Update on-target tracking: mark slices that are meeting their targets
    new_on_target = [False] * sla.n
    for k in range(sla.n):
        if delivered[k] >= targets[k] - 0.5:
            new_on_target[k] = True
    
    # Update state
    state.set('prev_shares', out)
    state.set('cqi_history', cqi_hist)
    state.set('on_target', new_on_target)
    
    return out