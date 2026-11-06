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
    """Enhanced controller with better attainment targeting and aggressive fairness."""
    n = sla.n
    
    # Observe current CQI
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    prev_action = state.get('previous_action', None)
    tick_count = state.get('tick_count', 0)
    attainment_history = state.get('attainment_history', [])
    
    # Estimate effective throughput per share with adaptive pessimism
    # Higher scale when channels are stable, lower when degraded
    avg_cqi = sum(obs) / n
    scale = 0.82 if avg_cqi < 10 else 0.88
    est_eff = [e_of_cqi(c) * scale for c in obs]
    
    # Calculate shares needed to hit each target
    target_shares = []
    for k in range(n):
        if est_eff[k] > 0.1:
            needed = math.ceil(T[k] / est_eff[k])
            target_shares.append(min(needed, GCAP))
        else:
            target_shares.append(GCAP)
    
    total_needed = sum(target_shares)
    
    # Strategy: if all targets fit in GCAP, allocate to meet them; otherwise prioritize by w_k
    candidates = []
    
    # Candidate 1: Allocate exactly to target if possible
    if total_needed <= GCAP:
        candidates.append(target_shares)
    else:
        # Candidate 1b: Scale down proportionally by weight
        scale_factor = GCAP / total_needed
        scaled = [max(1, int(target_shares[k] * scale_factor)) for k in range(n)]
        s = sum(scaled)
        if s > GCAP:
            for k in range(n):
                scaled[k] = max(0, int(scaled[k] * GCAP / s))
        candidates.append(scaled)
    
    # Candidate 2: Prioritize by weight × effectiveness
    priority = [(W[k] * est_eff[k], k) for k in range(n)]
    priority.sort(reverse=True)
    weighted = [0] * n
    remaining = GCAP
    for _, k in priority:
        alloc = min(remaining, target_shares[k])
        weighted[k] = alloc
        remaining -= alloc
    candidates.append(weighted)
    
    # Candidate 3: Prioritize unmet slices (those farthest from target)
    deficit = []
    for k in range(n):
        delivered = est_eff[k] * target_shares[k]
        gap = max(0, T[k] - delivered)
        deficit.append((W[k] * gap, k))
    deficit.sort(reverse=True)
    deficit_alloc = [0] * n
    remaining = GCAP
    for _, k in deficit:
        alloc = min(remaining, target_shares[k] + 2)
        deficit_alloc[k] = alloc
        remaining -= alloc
    candidates.append(deficit_alloc)
    
    # Candidate 4: Conservative equal split with slight bias to high-weight slices
    base_share = GCAP // n
    biased = [base_share] * n
    remaining = GCAP - sum(biased)
    sorted_by_w = sorted(range(n), key=lambda k: W[k], reverse=True)
    for k in sorted_by_w:
        if remaining <= 0:
            break
        biased[k] += 1
        remaining -= 1
    candidates.append(biased)
    
    # Candidate 5: Previous action with slight adjustment
    if prev_action is not None and tick_count > 0:
        adjusted = list(prev_action)
        # Marginal adjustments: increase shares where attainment is failing
        for k in range(n):
            delivered = est_eff[k] * adjusted[k]
            if delivered < T[k] - 1 and sum(adjusted) < GCAP:
                adjusted[k] = min(adjusted[k] + 1, GCAP - sum(adjusted) + adjusted[k])
        if sum(adjusted) <= GCAP:
            candidates.append(adjusted)
        candidates.append(list(prev_action))
    
    # Validate and clamp candidates
    valid = []
    for s in candidates:
        clamped = [max(0, int(x)) for x in s]
        total = sum(clamped)
        if total <= GCAP:
            valid.append(clamped)
        elif total > 0:
            scaled = [int(c * GCAP / total) for c in clamped]
            if sum(scaled) <= GCAP:
                valid.append(scaled)
    
    # Select best candidate using lookahead scoring
    if valid:
        best_score = None
        best_action = valid[0]
        for s in valid:
            score = _composite_tick(est_eff, s, prev_action)
            if best_score is None or score > best_score:
                best_score = score
                best_action = s
        selected = best_action
    else:
        selected = [0] * n
    
    # Update state
    state.set('previous_action', list(selected))
    state.set('tick_count', tick_count + 1)
    
    # Track attainment for future decisions
    new_attainment = []
    for k in range(n):
        delivered = est_eff[k] * selected[k]
        new_attainment.append(1 if delivered >= T[k] - OK_EPS else 0)
    attainment_history.append(new_attainment)
    if len(attainment_history) > 10:
        attainment_history.pop(0)
    state.set('attainment_history', attainment_history)
    
    return list(selected)