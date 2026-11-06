import math


def controller(history, state, sla, predictors):
    """Advanced allocation controller with CQI-weighted arbitration and slack optimization.
    
    Key improvements:
    - Use CQI as a direct signal for channel quality to prioritize allocation
    - Implement smarter slack distribution based on rate potential
    - Refine efficiency function calibration
    - Better distinction between guaranteed demand and achievable targets
    - Minimize churn while maximizing attainment
    """
    n = sla.n
    W = [sla.slices[k].weight for k in range(n)]
    T = [sla.slices[k].target_mbps for k in range(n)]
    cap = sla.guaranteed_cap
    
    def eff(cqi):
        c = max(5.0, min(15.0, cqi))
        # Calibrated efficiency: Mbps per PRB share
        return 0.25 + 2.15 * ((c - 5.0) / 10.0) ** 1.55
    
    # Gather observations
    obs_cqi = []
    obs_mbps = []
    obs_share = []
    
    for k in range(n):
        sample = history.slices[k].last()
        if sample:
            obs_cqi.append(sample.cqi)
            obs_mbps.append(history.slices[k].latest_mbps())
            obs_share.append(sample.prb_share)
        else:
            obs_cqi.append(10.0)
            obs_mbps.append(0.0)
            obs_share.append(0.0)
    
    # Estimate rates with empirical calibration
    rate = []
    cqi_valid = []
    
    for k in range(n):
        base_rate = eff(obs_cqi[k])
        
        # Use empirical data if available with sufficient share
        if obs_share[k] > 0.1 and obs_mbps[k] > 0:
            empirical = obs_mbps[k] / obs_share[k]
            # Trust empirical more when CQI is stable
            rate.append(0.5 * base_rate + 0.5 * empirical)
            cqi_valid.append(obs_cqi[k])
        else:
            rate.append(base_rate)
            cqi_valid.append(obs_cqi[k])
    
    # Compute minimum shares needed for targets with tight tolerance
    tolerance = 0.3
    min_share = []
    achievable = []
    
    for k in range(n):
        r = rate[k]
        if r < 0.01:
            # Channel too degraded; target unreachable at any share
            min_share.append(cap + 1)
            achievable.append(False)
        else:
            needed = (T[k] - tolerance) / r
            min_share.append(max(0, math.ceil(needed)))
            achievable.append(obs_cqi[k] >= 7.0)
    
    # Get previous state
    prev_shares = state.get("prev_shares", [0] * n)
    prev_attain = state.get("prev_attain", [0] * n)
    
    # Calculate current attainment
    attain = []
    for k in range(n):
        if obs_mbps[k] >= T[k] - 0.5:
            attain.append(1.0)
        else:
            attain.append(max(0.0, obs_mbps[k] / max(0.1, T[k])))
    
    # Phase 1: Mandatory minimum allocation
    shares = [0] * n
    used = 0
    
    # Allocate in order of weight * achievability, prioritizing high-weight slices
    order = sorted(range(n), 
                   key=lambda k: (-W[k] * (1.0 if achievable[k] else 0.3), 
                                 attain[k] < 0.95))
    
    for k in order:
        if min_share[k] <= cap - used:
            shares[k] = min_share[k]
            used += min_share[k]
    
    # Phase 2: Allocate remaining slack to maximize weighted attainment improvement
    slack = cap - used
    
    if slack > 0:
        # Calculate marginal benefit of additional share
        candidates = []
        for k in range(n):
            if shares[k] < cap - 5:  # Don't exceed reasonable single-slice limit
                current_delivery = rate[k] * shares[k]
                next_delivery = rate[k] * (shares[k] + 1)
                benefit = W[k] * max(0, min(1.0, (next_delivery - T[k]) / max(0.1, T[k])))
                if rate[k] > 0.05:
                    candidates.append((k, benefit, rate[k]))
        
        candidates.sort(key=lambda x: (-x[1], -x[2]))
        
        for k, _, _ in candidates:
            if slack <= 0:
                break
            # Allocate in chunks
            bump = min(slack, 4)
            shares[k] += bump
            slack -= bump
    
    # Phase 3: Churn mitigation with stability heuristic
    # Count differences from previous
    changes = sum(1 for k in range(n) if shares[k] != prev_shares[k])
    
    # If minimal change and situation hasn't deteriorated, stick with previous
    if changes <= 2:
        deteriorated = sum(1 for k in range(n) 
                          if attain[k] < prev_attain[k] - 0.05)
        if deteriorated == 0:
            # Apply hysteresis: only keep new allocation if clearly beneficial
            new_weighted = sum(W[k] * attain[k] for k in range(n))
            prev_weighted = sum(W[k] * prev_attain[k] for k in range(n))
            if new_weighted <= prev_weighted + 0.01:
                shares = list(prev_shares)
    
    # Safety: ensure no negative shares and total <= cap
    shares = [max(0, s) for s in shares]
    total = sum(shares)
    if total > cap:
        # Scale down proportionally
        shares = [int(s * cap / total) for s in shares]
    
    state.set("prev_shares", list(shares))
    state.set("prev_attain", list(attain))
    
    return shares