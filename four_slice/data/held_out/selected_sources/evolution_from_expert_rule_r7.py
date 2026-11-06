import math

POOL, GCAP, NG = 100, 70, 4
W = (1.00, 0.90, 0.80, 0.70)
T = (44.0, 46.0, 48.0, 50.0)
CQI_LO, CQI_HI = 5.0, 15.0
E_MAX, E_FLOOR, E_EXP = 2.39, 0.30, 1.578
_E_C = (E_MAX - E_FLOOR) / (CQI_HI - CQI_LO) ** E_EXP
E_BE, BE_W, CHURN, OK_EPS = 2.39, 0.6, 0.02, 0.5
CEIL = E_BE * POOL

def e_of_cqi(cqi):
    c = min(max(cqi, CQI_LO), CQI_HI)
    return E_FLOOR + _E_C * (c - CQI_LO) ** E_EXP

def _be_share(s):
    return POOL - sum(s)

def _composite_tick(e_t, s, prev):
    d = [e_t[k] * s[k] for k in range(NG)]
    g = 0.0
    for k in range(NG):
        g += W[k] if d[k] >= T[k] - OK_EPS else -0.5 * W[k]
    be = BE_W * (E_BE * _be_share(s)) / CEIL
    changed = 0
    if prev is not None:
        full = list(s) + [_be_share(s)]
        pfull = list(prev) + [_be_share(prev)]
        changed = sum(1 for a, b in zip(full, pfull) if a != b)
    return g + be - CHURN * changed

def controller(history, state, sla, predictors):
    """Enhanced controller with adaptive CQI prediction, target-aware allocation,
    and reduced churn through stability bias."""
    
    n = sla.n
    
    # Observe current CQI and delivered throughput
    obs_cqi = []
    obs_mbps = []
    for k in range(n):
        sample = history.slices[k].last()
        obs_cqi.append(sample.cqi if sample else 15.0)
        obs_mbps.append(history.slices[k].latest_mbps() if sample else 0.0)
    
    # Maintain CQI history for trend analysis
    cqi_history = state.get('cqi_history', [[] for _ in range(n)])
    prev_cqi = state.get('prior_observed_cqi', None)
    prev_action = state.get('previous_action', None)
    episode_tick = state.get('episode_tick', 0)
    
    for k in range(n):
        cqi_history[k].append(obs_cqi[k])
        if len(cqi_history[k]) > 8:
            cqi_history[k].pop(0)
    
    # Adaptive CQI prediction with momentum
    pred_cqi = []
    for k in range(n):
        curr = obs_cqi[k]
        
        if len(cqi_history[k]) >= 3:
            # Multi-step trend: weight recent changes higher
            recent = cqi_history[k][-4:]
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            
            # Weighted average of diffs (recent changes matter more)
            weights = [0.2, 0.3, 0.5]
            avg_delta = sum(w * d for w, d in zip(weights, diffs)) / sum(weights)
            
            # Damped momentum: smooth changes, avoid oscillation
            momentum_factor = 0.35
            predicted = curr + momentum_factor * avg_delta
        else:
            predicted = curr
        
        pred_cqi.append(min(CQI_HI, max(CQI_LO, predicted)))
    
    # Estimate efficiency with conservative adjustment
    est_eff = [e_of_cqi(c) * 0.88 for c in pred_cqi]
    
    # Calculate minimum shares to meet targets
    min_shares = [0] * n
    for k in range(n):
        if est_eff[k] > 0.01:
            min_shares[k] = max(1, math.ceil(T[k] / est_eff[k]))
        else:
            min_shares[k] = GCAP
    
    # Clamp minimum shares
    for k in range(n):
        min_shares[k] = min(min_shares[k], GCAP)
    
    # Total minimum demand
    total_min = sum(min_shares)
    
    candidates = []
    
    # Strategy 1: Proportional allocation by weight (baseline)
    if total_min <= GCAP:
        base = min_shares[:]
        slack = GCAP - total_min
        
        # Distribute slack proportionally to weights
        weight_sum = sum(W)
        for k in range(n):
            extra = int((W[k] / weight_sum) * slack)
            base[k] += extra
        
        # Recover rounding
        deficit = GCAP - sum(base)
        for _ in range(deficit):
            k = max(range(n), key=lambda i: W[i])
            base[k] += 1
        
        candidates.append(base)
    
    # Strategy 2: Weight-priority with efficiency awareness
    ordered = sorted(range(n), key=lambda k: (W[k], est_eff[k]), reverse=True)
    priority_shares = [0] * n
    remaining = GCAP
    
    # First pass: minimum viable
    for k in ordered:
        alloc = min(min_shares[k], remaining)
        priority_shares[k] = alloc
        remaining -= alloc
    
    # Second pass: extra shares to highest-weight slices
    for k in ordered:
        if remaining <= 0:
            break
        extra = min(remaining, GCAP - priority_shares[k])
        priority_shares[k] += extra
        remaining -= extra
    
    candidates.append(priority_shares)
    
    # Strategy 3: Efficiency-weighted allocation (favor good channels)
    eff_sorted = sorted(range(n), key=lambda k: est_eff[k], reverse=True)
    eff_shares = [0] * n
    remaining = GCAP
    
    for k in eff_sorted:
        alloc = min(min_shares[k], remaining)
        eff_shares[k] = alloc
        remaining -= alloc
    
    if remaining > 0:
        for k in eff_sorted:
            add = min(remaining, GCAP - eff_shares[k])
            eff_shares[k] += add
            remaining -= add
            if remaining == 0:
                break
    
    candidates.append(eff_shares)
    
    # Strategy 4: Stability-biased (preserve previous with minor tuning)
    if prev_action is not None and episode_tick > 0:
        # Check CQI stability
        cqi_volatility = 0.0
        if prev_cqi is not None:
            cqi_volatility = sum(abs(obs_cqi[k] - prev_cqi[k]) for k in range(n)) / n
        
        if cqi_volatility < 1.2:
            # Keep previous action (high confidence in stability)
            candidates.append(list(prev_action))
        else:
            # Gradual adjustment: move by at most 2 per slice
            adjusted = list(prev_action)
            for k in range(n):
                target = min_shares[k]
                if adjusted[k] < target:
                    adjusted[k] = min(adjusted[k] + 2, target)
                elif adjusted[k] > target + 5:
                    adjusted[k] = max(adjusted[k] - 1, target)
            
            total_adj = sum(adjusted)
            if total_adj > GCAP:
                scale = GCAP / total_adj
                adjusted = [max(0, int(x * scale)) for x in adjusted]
                deficit = GCAP - sum(adjusted)
                for _ in range(deficit):
                    k = max(range(n), key=lambda i: W[i])
                    adjusted[k] += 1
            
            candidates.append(adjusted)
    
    # Strategy 5: Aggressive on highest-weight, conservative on others
    aggressive = [0] * n
    remaining = GCAP
    
    # Allocate greedily by weight
    by_weight = sorted(range(n), key=lambda k: W[k], reverse=True)
    for k in by_weight:
        alloc = min(min_shares[k] + (3 if k == 0 else 1), remaining)
        aggressive[k] = alloc
        remaining -= alloc
    
    if sum(aggressive) <= GCAP:
        candidates.append(aggressive)
    
    # Fallback
    if not candidates:
        candidates = [[17, 17, 17, 17]]
    
    # Evaluate all candidates
    best_score = float('-inf')
    best_candidate = candidates[0]
    
    for cand in candidates:
        # Clamp
        cand = [max(0, int(x)) for x in cand]
        total = sum(cand)
        
        if total > GCAP:
            scale = GCAP / total
            cand = [max(0, int(x * scale)) for x in cand]
            deficit = GCAP - sum(cand)
            for _ in range(deficit):
                k = max(range(n), key=lambda i: W[i])
                cand[k] += 1
        
        score = _composite_tick(est_eff, cand, prev_action)
        if score > best_score:
            best_score = score
            best_candidate = cand
    
    # Update state
    state.set('prior_observed_cqi', obs_cqi)
    state.set('previous_action', best_candidate)
    state.set('cqi_history', cqi_history)
    state.set('episode_tick', episode_tick + 1)
    
    return best_candidate