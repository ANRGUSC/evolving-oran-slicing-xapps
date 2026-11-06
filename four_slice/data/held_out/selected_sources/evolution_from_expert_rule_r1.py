def controller(history, state, sla, predictors):
    """
    Improved controller with dynamic allocation, attainment-weighted sharing, 
    and aggressive best-effort maximization when targets cannot be met.
    """
    # Constants
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
    
    n = sla.n
    
    # Observe current CQI
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    prev_cqi = state.get('prior_observed_cqi', None)
    
    # CQI prediction with moderate momentum
    alpha = 0.12
    pred = []
    for k, c in enumerate(obs):
        if prev_cqi is not None:
            momentum = alpha * (c - prev_cqi[k])
            pred.append(min(CQI_HI, max(CQI_LO, c + momentum)))
        else:
            pred.append(c)
    
    # Estimate efficiency per share for each slice
    est_eff = [e_of_cqi(pred[k]) for k in range(n)]
    
    # Compute minimum shares needed to meet targets
    min_shares = []
    for k in range(n):
        if est_eff[k] > 0.01:
            min_share = (T[k] + 0.5) / est_eff[k]
            min_shares.append(max(1, int(min_share + 0.9999)))  # Ceiling
        else:
            min_shares.append(GCAP + 1)  # Impossible to meet
    
    # Compute total capacity and deficit
    total_min = sum(min_shares)
    deficit = max(0, total_min - GCAP)
    
    prev_action = state.get('previous_action', None)
    
    # Strategy selection based on deficit
    if deficit <= 0:
        # All targets can be met with minimum shares; allocate remaining capacity
        shares = list(min_shares)
        spare = GCAP - sum(shares)
        
        # Distribute spare capacity by weighted efficiency to maximize composite score
        if spare > 0:
            # Allocate spare proportionally to W[k] * est_eff[k]
            priority_scores = [W[k] * est_eff[k] for k in range(n)]
            priority_sum = sum(priority_scores)
            
            if priority_sum > 0:
                for k in range(n):
                    allocation = int((priority_scores[k] / priority_sum) * spare)
                    shares[k] += allocation
            
            # Ensure we use all spare (handle rounding)
            remaining_spare = GCAP - sum(shares)
            if remaining_spare > 0:
                # Give to highest weight
                shares[0] += remaining_spare
        
        selected = shares
    else:
        # Deficit exists: prioritize by weighted attainment potential
        # Focus allocation on slices most likely to contribute to score
        priority = [(W[k] * est_eff[k], k) for k in range(n)]
        priority.sort(reverse=True)
        
        shares = [0] * n
        remaining = GCAP
        
        # Allocate to meet targets in priority order
        for _, k in priority:
            if remaining > 0 and min_shares[k] <= remaining:
                alloc = min(min_shares[k], remaining)
                shares[k] = alloc
                remaining -= alloc
        
        # Distribute remaining capacity to highest priority slices
        for _, k in priority:
            if remaining > 0:
                boost = min(remaining, 3)
                shares[k] += boost
                remaining -= boost
        
        selected = shares
    
    # Clamp to valid range
    selected = [max(0, min(GCAP, s)) for s in selected]
    total = sum(selected)
    if total > GCAP:
        scale_factor = GCAP / total
        selected = [int(s * scale_factor) for s in selected]
        delta = GCAP - sum(selected)
        if delta > 0:
            selected[0] += delta
    
    # Consider inertia: penalize excessive churn relative to score gain
    if prev_action is not None:
        alt_score = _composite_tick(est_eff, selected, prev_action)
        prev_score = _composite_tick(est_eff, prev_action, prev_action)
        
        # Require significant score improvement to justify change
        if alt_score < prev_score + 0.02:
            selected = list(prev_action)
    
    state.set('prior_observed_cqi', list(obs))
    state.set('previous_action', list(selected))
    
    return selected