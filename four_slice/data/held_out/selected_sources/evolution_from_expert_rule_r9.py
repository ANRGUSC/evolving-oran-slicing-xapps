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
    """Improved controller: adaptive scaling with damping, weighted priority allocation,
    and smoother transitions. Uses CQI history for better prediction."""
    alpha, scale = 0.15, 0.85
    n = sla.n
    
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    prev_cqi = state.get('prior_observed_cqi', None)
    prev_action = state.get('previous_action', None)
    
    pred = []
    for k, c in enumerate(obs):
        if prev_cqi is not None:
            delta = c - prev_cqi[k]
            smoothed = c + alpha * delta
        else:
            smoothed = c
        pred.append(min(CQI_HI, max(CQI_LO, smoothed)))
    
    est_eff = [e_of_cqi(c) * scale for c in pred]
    
    required = [max(0, math.ceil((sla.slices[k].target_mbps - OK_EPS) / est_eff[k])) for k in range(n)]
    
    candidates = []
    
    for mask in range(1 << n):
        shares = [required[k] if mask & (1 << k) else 0 for k in range(n)]
        if sum(shares) <= sla.guaranteed_cap:
            candidates.append(shares)
    
    if prev_action is not None:
        candidates.append(list(prev_action))
        for k in range(n):
            for delta in [-2, -1, 1, 2]:
                variant = list(prev_action)
                variant[k] = max(0, variant[k] + delta)
                if sum(variant) <= sla.guaranteed_cap:
                    candidates.append(variant)
    
    best_score = -float('inf')
    best_candidate = candidates[0] if candidates else [0, 0, 0, 0]
    
    for s in candidates:
        score = _composite_tick(est_eff, s, prev_action)
        if score > best_score:
            best_score = score
            best_candidate = s
    
    selected = list(best_candidate)
    state.set('prior_observed_cqi', list(obs))
    state.set('previous_action', list(selected))
    
    return selected