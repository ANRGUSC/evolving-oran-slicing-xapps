"""Enhanced controller with improved attainment targeting, aggressive feasibility detection, and better allocation dynamics."""
import math

def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578


def controller(history, state, sla, predictors):
    """Enhanced controller with focus on attainment maximization and pool efficiency."""
    
    # Initialize state
    if not state.get('init'):
        state.set('init', True)
        state.set('prev_shares', [0, 0, 0, 0])
        state.set('attain_history', [[0.0]*40 for _ in range(4)])
        state.set('cqi_history', [[10.0]*40 for _ in range(4)])
        state.set('share_history', [[0]*40 for _ in range(4)])
        state.set('episode_tick', 0)
    
    tick = state.get('episode_tick', 0)
    state.set('episode_tick', tick + 1)
    
    # Gather current observations
    cqis = []
    rates = []
    targets = []
    weights = []
    
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 10.0
        cqis.append(cqi)
        rates.append(_rate_of_cqi(cqi))
        targets.append(sla.slices[k].target_mbps)
        weights.append(sla.slices[k].weight)
    
    prev_shares = state.get('prev_shares', [0, 0, 0, 0])
    attain_hist = state.get('attain_history', [[0.0]*40 for _ in range(4)])
    cqi_hist = state.get('cqi_history', [[10.0]*40 for _ in range(4)])
    share_hist = state.get('share_history', [[0]*40 for _ in range(4)])
    
    # Update histories
    for k in range(4):
        cqi_hist[k] = cqi_hist[k][1:] + [cqis[k]]
        share_hist[k] = share_hist[k][1:] + [prev_shares[k]]
    
    # Compute long-term metrics for each slice
    long_attain = []
    recent_attain = []
    cqi_mean = []
    cqi_volatility = []
    attain_trend = []
    
    for k in range(4):
        # Long-term attainment (last 25 ticks)
        long_hist = attain_hist[k][-25:] if len(attain_hist[k]) >= 25 else attain_hist[k]
        long_att = sum(long_hist) / len(long_hist) if long_hist else 0.0
        long_attain.append(long_att)
        
        # Recent attainment (last 10 ticks)
        recent_hist = attain_hist[k][-10:] if len(attain_hist[k]) >= 10 else attain_hist[k]
        recent_att = sum(recent_hist) / len(recent_hist) if recent_hist else 0.0
        recent_attain.append(recent_att)
        
        # CQI statistics
        c_hist = cqi_hist[k][-25:] if len(cqi_hist[k]) >= 25 else cqi_hist[k]
        if c_hist:
            mean_cqi = sum(c_hist) / len(c_hist)
            variance = sum((x - mean_cqi)**2 for x in c_hist) / len(c_hist)
            cqi_mean.append(mean_cqi)
            cqi_volatility.append(math.sqrt(variance))
        else:
            cqi_mean.append(10.0)
            cqi_volatility.append(1.0)
        
        # Attainment trend (recent vs long-term)
        attain_trend.append(recent_att - long_att)
    
    # Compute minimum shares needed at current CQI
    min_shares = []
    for k in range(4):
        if rates[k] > 0.001:
            min_share = math.ceil(targets[k] / rates[k])
        else:
            min_share = sla.guaranteed_cap + 1
        min_shares.append(max(1, min(min_share, sla.guaranteed_cap)))
    
    total_min = sum(min_shares)
    pool_pressure = total_min / float(sla.guaranteed_cap)
    
    # Dynamic feasibility thresholds based on pool pressure and performance
    if pool_pressure > 1.2:
        # Extreme pressure: only serve most viable slices
        base_thresholds = [9.5, 9.5, 10.5, 10.5]
    elif pool_pressure > 1.0:
        # High pressure
        base_thresholds = [8.8, 8.8, 9.5, 9.5]
    elif pool_pressure > 0.9:
        # Medium-high pressure
        base_thresholds = [8.2, 8.2, 8.8, 8.8]
    elif pool_pressure > 0.7:
        # Low-medium pressure
        base_thresholds = [7.8, 7.8, 8.2, 8.2]
    else:
        # Low pressure: serve all
        base_thresholds = [6.5, 6.5, 7.0, 7.0]
    
    thresholds = list(base_thresholds)
    
    # Adjust thresholds based on performance history
    for k in range(4):
        # Chronically underperforming slices: lower threshold
        if long_attain[k] < 0.25:
            thresholds[k] = max(5.0, thresholds[k] - 2.0)
        elif long_attain[k] < 0.40:
            thresholds[k] = max(5.5, thresholds[k] - 1.2)
        elif long_attain[k] < 0.60:
            thresholds[k] = max(6.5, thresholds[k] - 0.5)
        
        # Improving slices: slight boost in threshold
        if attain_trend[k] > 0.15:
            thresholds[k] += 0.2
    
    feasible = [cqis[k] >= thresholds[k] for k in range(4)]
    
    # Compute allocation scores for each feasible slice
    # Prioritize: weight * underperformance, inverse of min_share cost
    scores = []
    for k in range(4):
        if feasible[k]:
            underperf = max(0.0, 1.0 - long_attain[k])
            weight_factor = weights[k]
            perf_factor = 0.5 + 0.5 * (1.0 - underperf)  # Boost underperformers
            cost_factor = 1.0 / (1.0 + min_shares[k] / 20.0)  # Prefer cheaper allocations
            score = weight_factor * perf_factor * cost_factor
        else:
            score = -1.0
        scores.append((score, k))
    
    scores.sort(reverse=True)
    
    # Greedy allocation with boost multipliers
    out = [0, 0, 0, 0]
    remaining_cap = sla.guaranteed_cap
    allocated_capacity = {}
    
    for score_val, k in scores:
        if score_val <= 0.0 or remaining_cap <= 0:
            out[k] = 0
            continue
        
        needed = min_shares[k]
        
        # Compute boost based on long-term attainment deficit
        if long_attain[k] < 0.2:
            boost = 1.6
        elif long_attain[k] < 0.35:
            boost = 1.45
        elif long_attain[k] < 0.50:
            boost = 1.25
        elif long_attain[k] < 0.65:
            boost = 1.10
        else:
            boost = 1.0
        
        # Also boost if recent trend is negative (degrading)
        if attain_trend[k] < -0.1:
            boost *= 1.08
        
        ideal_share = min(int(math.ceil(needed * boost)), remaining_cap)
        
        if ideal_share > 0:
            grant = min(ideal_share, remaining_cap)
            out[k] = grant
            allocated_capacity[k] = grant
            remaining_cap -= grant
    
    # Distribute any remaining capacity to highest-priority feasible slice
    if remaining_cap > 0:
        for score_val, k in scores:
            if score_val > 0.0:
                out[k] += remaining_cap
                break
    
    # Smooth transitions: avoid thrashing while preserving attainment gains
    for k in range(4):
        old_share = prev_shares[k]
        new_share = out[k]
        
        if old_share == 0 or new_share == 0 or old_share == new_share:
            continue
        
        # Compute delivered throughput at old and new allocations
        old_delivered = rates[k] * old_share if rates[k] > 0 else 0.0
        new_delivered = rates[k] * new_share if rates[k] > 0 else 0.0
        
        old_attain = 1.0 if old_delivered >= targets[k] - 0.5 else 0.0
        new_attain = 1.0 if new_delivered >= targets[k] - 0.5 else 0.0
        
        # Only revert if: (1) no attainment gain, (2) already performing well, (3) difference is small
        if (old_attain >= new_attain and recent_attain[k] >= 0.70 and 
            abs(new_share - old_share) <= 5):
            out[k] = old_share
    
    # Final clamp and normalize
    out = [max(0, int(x)) for x in out]
    total = sum(out)
    
    if total > sla.guaranteed_cap:
        # Proportional scale with protection for high-priority slices
        scale = sla.guaranteed_cap / float(total)
        out_scaled = [int(x * scale) if x > 0 else 0 for x in out]
        total_scaled = sum(out_scaled)
        remainder = sla.guaranteed_cap - total_scaled
        
        # Distribute remainder to highest-scoring feasible slice
        for score_val, k in scores:
            if remainder > 0 and feasible[k] and out[k] > 0:
                out_scaled[k] += remainder
                break
        
        out = out_scaled
    
    # Update state
    state.set('prev_shares', list(out))
    state.set('cqi_history', cqi_hist)
    state.set('share_history', share_hist)
    
    # Update attainment history
    for k in range(4):
        delivered = rates[k] * out[k] if rates[k] > 0 else 0.0
        attain = 1.0 if delivered >= targets[k] - 0.5 else 0.0
        attain_hist[k] = attain_hist[k][1:] + [attain]
    
    state.set('attain_history', attain_hist)
    
    return out