import math

# --- env_d constants + functions inlined verbatim ---
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
    """Improved controller with better attainment tracking and adaptive allocation.
    Uses exponential smoothing on CQI predictions, weighted deficit prioritization,
    and smoother share transitions to reduce churn."""
    
    alpha = 0.15  # Increased CQI trend tracking
    scale = 0.85  # More conservative efficiency estimate
    deficit_weight = 1.2  # Favor slices further from target
    
    n = sla.n
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    
    prev_cqi = state.get('prior_observed_cqi', None)
    pred = [min(CQI_HI, max(CQI_LO, c + (alpha * (c - prev_cqi[k]) if prev_cqi is not None else 0.0)))
            for k, c in enumerate(obs)]
    
    est_eff = [e_of_cqi(c) * scale for c in pred]
    
    # Calculate minimum shares needed, with smoothing toward previous allocation
    required = [max(0, math.ceil((sla.slices[k].target_mbps - OK_EPS) / est_eff[k])) 
                for k in range(n)]
    
    prev_action = state.get('previous_action', None)
    prev = state.get('previous_action', None)
    
    # Build candidate pool with more granular options
    candidates = set()
    
    # 1. All subsets of minimum required (original approach)
    for mask in range(1 << n):
        shares = [required[k] if mask & (1 << k) else 0 for k in range(n)]
        if sum(shares) <= sla.guaranteed_cap:
            candidates.add(tuple(shares))
    
    # 2. Greedy deficit-based allocation: prioritize slices furthest from target
    base_shares = [0] * n
    deficit = [(T[k] - est_eff[k] * 0.5) * deficit_weight for k in range(n)]
    remaining_cap = sla.guaranteed_cap
    
    for _ in range(sla.guaranteed_cap):
        best_k = -1
        best_gain = -1
        for k in range(n):
            if base_shares[k] < sla.guaranteed_cap:
                gain = deficit[k] / max(1, base_shares[k] + 1)
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
        if best_k >= 0:
            base_shares[best_k] += 1
        else:
            break
    if sum(base_shares) <= sla.guaranteed_cap:
        candidates.add(tuple(base_shares))
    
    # 3. Smooth transitions from previous action (reduce churn)
    if prev_action is not None:
        for delta in range(-2, 3):
            for k in range(n):
                trial = list(prev_action)
                trial[k] = max(0, trial[k] + delta)
                if sum(trial) <= sla.guaranteed_cap:
                    candidates.add(tuple(trial))
    
    # 4. Keep previous action (stability bonus)
    if prev is not None:
        candidates.add(tuple(prev))
    
    # 5. Proportional allocation by weight with boosted targets
    total_w = sum(W)
    prop_shares = [max(0, int(sla.guaranteed_cap * W[k] / total_w)) for k in range(n)]
    if sum(prop_shares) <= sla.guaranteed_cap:
        candidates.add(tuple(prop_shares))
    
    # Select best candidate
    if candidates:
        selected = list(max(candidates, key=lambda s: _composite_tick(est_eff, list(s), prev)))
    else:
        selected = [0] * n
    
    state.set('prior_observed_cqi', list(obs))
    state.set('previous_action', list(selected))
    
    return selected