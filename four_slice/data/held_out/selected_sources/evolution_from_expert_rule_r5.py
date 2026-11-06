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
    """Enhanced controller with better demand estimation and adaptive allocation."""
    n = sla.n
    
    # Observation and CQI history tracking
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 12.5) for k in range(n)]
    prev_cqi = state.get('prior_observed_cqi', None)
    cqi_history = state.get('cqi_history', [[] for _ in range(n)])
    
    # Update CQI history
    for k in range(n):
        cqi_history[k].append(obs[k])
        if len(cqi_history[k]) > 8:
            cqi_history[k].pop(0)
    
    # Better CQI prediction with smoothing
    pred = []
    for k, c in enumerate(obs):
        if prev_cqi is not None:
            delta = c - prev_cqi[k]
            # Adaptive momentum based on recent trend
            if len(cqi_history[k]) >= 3:
                trend = sum(cqi_history[k][-i] - cqi_history[k][-i-1] for i in range(1, min(3, len(cqi_history[k])))) / min(2, len(cqi_history[k]) - 1)
                momentum = 0.08 * trend
            else:
                momentum = 0.12 * delta
            p = min(CQI_HI, max(CQI_LO, c + momentum))
        else:
            p = c
        pred.append(p)
    
    # Efficiency estimates with conservative scaling
    est_eff = [e_of_cqi(c) * 0.80 for c in pred]
    
    # Dynamic demand calculation with weight consideration
    required = []
    for k in range(n):
        if est_eff[k] > 0.1:
            req = max(1, math.ceil((sla.slices[k].target_mbps - OK_EPS) / est_eff[k]))
        else:
            req = max(1, int(2.0 * sla.slices[k].target_mbps / E_MAX))
        required.append(req)
    
    prev = state.get('previous_action', None)
    candidates = []
    
    # Generate candidates: required demands
    if sum(required) <= sla.guaranteed_cap:
        candidates.append(list(required))
    
    # Weighted proportional allocation (prioritize high-weight slices)
    total_weight = sum(W)
    weighted_shares = [max(1, int(sla.guaranteed_cap * W[k] / total_weight)) for k in range(n)]
    if sum(weighted_shares) <= sla.guaranteed_cap:
        candidates.append(weighted_shares)
    
    # Greedy: allocate to slices by efficiency-adjusted demand
    shares_greedy = [0] * n
    remaining = sla.guaranteed_cap
    scored = [(est_eff[k] * W[k], k) for k in range(n)]
    scored.sort(reverse=True)
    for eff_weight, k in scored:
        alloc = min(required[k], remaining)
        shares_greedy[k] = alloc
        remaining -= alloc
    candidates.append(shares_greedy)
    
    # Previous action (persistence)
    if prev is not None:
        candidates.append(list(prev))
    
    # Conservative fallback for poor channels
    fallback = []
    for k in range(n):
        if est_eff[k] < 0.4:
            fallback.append(1)
        elif est_eff[k] < 0.7:
            fallback.append(min(required[k], 3))
        else:
            fallback.append(min(required[k], 5))
    if sum(fallback) <= sla.guaranteed_cap:
        candidates.append(fallback)
    
    # Best-effort maximization candidate
    be_max = [1] * n
    while sum(be_max) < sla.guaranteed_cap - 5 and sum(be_max) < sla.guaranteed_cap:
        for k in range(n):
            if sum(be_max) < sla.guaranteed_cap:
                be_max[k] += 1
    candidates.append(be_max)
    
    best_score = -float('inf')
    best_share = list(required) if sum(required) <= sla.guaranteed_cap else [1] * n
    
    persistence_bonus = 0.006
    
    for s in candidates:
        if sum(s) > sla.guaranteed_cap:
            continue
        score = _composite_tick(est_eff, s, prev)
        if prev is not None and s == prev:
            score += persistence_bonus
        if score > best_score:
            best_score = score
            best_share = list(s)
    
    # Final cleanup
    if sum(best_share) > sla.guaranteed_cap:
        total = sum(best_share)
        best_share = [max(0, int(x * sla.guaranteed_cap / (total + 1e-6))) for x in best_share]
        deficit = sla.guaranteed_cap - sum(best_share)
        if deficit > 0:
            best_share[0] += deficit
    
    state.set('prior_observed_cqi', list(obs))
    state.set('previous_action', list(best_share))
    state.set('cqi_history', cqi_history)
    
    return list(best_share)