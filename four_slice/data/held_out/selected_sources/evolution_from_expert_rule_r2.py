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
    """Improved controller with better demand estimation, adaptive scaling, and focus on stability."""
    n = sla.n
    
    # Observe current CQI and delivered throughput
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    recent_mbps = [history.slices[k].latest_mbps() if history.slices[k].last() else 0.0 for k in range(n)]
    last_shares = [history.slices[k].last().prb_share if history.slices[k].last() else 0.0 for k in range(n)]
    
    prev_cqi = state.get('prior_observed_cqi', None)
    prev_action = state.get('previous_action', None)
    recent_history = state.get('recent_mbps_history', [[] for _ in range(n)])
    
    # Maintain rolling history of delivered throughput per slice
    for k in range(n):
        recent_history[k].append(recent_mbps[k])
        if len(recent_history[k]) > 5:
            recent_history[k].pop(0)
    
    # Estimate efficiency per share with adaptive scaling based on variance
    pred = list(obs)
    if prev_cqi is not None:
        momentum_factor = 0.15
        pred = [min(CQI_HI, max(CQI_LO, obs[k] + momentum_factor * (obs[k] - prev_cqi[k])))
                for k in range(n)]
    
    # Adaptive scaling: increase confidence in good channels, decrease in bad ones
    scale_factors = []
    for k in range(n):
        if len(recent_history[k]) > 0:
            hist_mean = sum(recent_history[k]) / len(recent_history[k])
            hist_var = sum((x - hist_mean) ** 2 for x in recent_history[k]) / len(recent_history[k])
            stability = 1.0 / (1.0 + 0.5 * math.sqrt(hist_var))
            scale_factors.append(0.82 + 0.12 * stability)
        else:
            scale_factors.append(0.87)
    
    est_eff = [e_of_cqi(pred[k]) * scale_factors[k] for k in range(n)]
    
    # Compute minimum viable share to meet each target
    required = [max(0, math.ceil((sla.slices[k].target_mbps - OK_EPS) / max(est_eff[k], 0.01))) for k in range(n)]
    for k in range(n):
        required[k] = min(required[k], sla.guaranteed_cap)
    
    # Check feasibility of meeting all targets
    total_required = sum(required)
    feasible_all = total_required <= sla.guaranteed_cap
    
    candidates = []
    
    # Candidate 1: Meet all targets if possible, otherwise proportional deficit reduction
    if feasible_all:
        candidates.append(list(required))
    else:
        deficit_alloc = list(required)
        remaining = sla.guaranteed_cap - sum(deficit_alloc)
        if remaining < 0:
            overshoot = -remaining
            deficit_indices = sorted(range(n), key=lambda k: W[k], reverse=True)
            for k in deficit_indices:
                cut = min(deficit_alloc[k], overshoot)
                deficit_alloc[k] -= cut
                overshoot -= cut
                if overshoot == 0:
                    break
        candidates.append(deficit_alloc)
    
    # Candidate 2: Greedy allocation favoring high-weight slices with good efficiency
    greedy = [0] * n
    remaining = sla.guaranteed_cap
    indices = sorted(range(n), key=lambda k: W[k] * est_eff[k], reverse=True)
    for k in indices:
        alloc = min(required[k], remaining)
        greedy[k] = alloc
        remaining -= alloc
    candidates.append(greedy)
    
    # Candidate 3: Weighted proportional with efficiency and demand awareness
    total_weight_eff = sum(W[k] * est_eff[k] for k in range(n))
    if total_weight_eff > 0:
        proportional = [0] * n
        remaining = sla.guaranteed_cap
        allocs_float = [(W[k] * est_eff[k] / total_weight_eff) * sla.guaranteed_cap for k in range(n)]
        for k in range(n):
            alloc = int(allocs_float[k])
            proportional[k] = min(alloc, remaining)
            remaining -= proportional[k]
        for k in sorted(range(n), key=lambda x: (allocs_float[x] - int(allocs_float[x])), reverse=True):
            if remaining > 0:
                proportional[k] += 1
                remaining -= 1
        candidates.append(proportional)
    
    # Candidate 4: Stability-biased (minimize changes from previous action)
    if prev_action is not None:
        candidates.append(list(prev_action))
    
    # Candidate 5: Focus on under-performing slices with available pool
    if sum(required) < sla.guaranteed_cap:
        focused = list(required)
        bonus_pool = sla.guaranteed_cap - sum(focused)
        underperforming = [(W[k], k) for k in range(n) if recent_mbps[k] < T[k] - OK_EPS]
        underperforming.sort(reverse=True)
        for _, k in underperforming:
            if bonus_pool > 0:
                boost = min(8, bonus_pool)
                focused[k] += boost
                bonus_pool -= boost
        if bonus_pool > 0:
            for k in sorted(range(n), key=lambda x: W[x], reverse=True):
                focused[k] += bonus_pool
                bonus_pool = 0
                break
        candidates.append(focused)
    
    # Candidate 6: Efficiency-maximizing (maximize best-effort share)
    be_max = [0] * n
    remaining = sla.guaranteed_cap
    indices = sorted(range(n), key=lambda k: est_eff[k], reverse=True)
    for k in indices:
        min_alloc = min(1, remaining)
        be_max[k] = min_alloc
        remaining -= min_alloc
    candidates.append(be_max)
    
    # Remove duplicates and invalid candidates
    seen = set()
    valid_candidates = []
    for cand in candidates:
        cand_tuple = tuple(cand)
        if cand_tuple not in seen and sum(cand) <= sla.guaranteed_cap and all(x >= 0 for x in cand):
            seen.add(cand_tuple)
            valid_candidates.append(cand)
    
    # Select best candidate
    if valid_candidates:
        selected = max(valid_candidates, key=lambda s: _composite_tick(est_eff, s, prev_action))
    else:
        selected = [0] * n
    
    # Soft stability constraint: nudge towards previous if scores are close
    if prev_action is not None:
        score_selected = _composite_tick(est_eff, selected, prev_action)
        score_prev = _composite_tick(est_eff, prev_action, prev_action)
        if abs(score_prev - score_selected) < 0.01:
            selected = list(prev_action)
    
    state.set('prior_observed_cqi', list(obs))
    state.set('previous_action', list(selected))
    state.set('recent_mbps_history', recent_history)
    
    return list(selected)