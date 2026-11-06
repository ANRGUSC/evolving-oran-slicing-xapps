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
    """Improved controller with better attainment prioritization and dynamic reallocation."""
    n = sla.n
    
    # Observe current CQI and delivered throughput
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    delivered = [(history.slices[k].latest_mbps() or 0.0) for k in range(n)]
    
    # Retrieve state
    prev_action = state.get('previous_action', None)
    cqi_history = state.get('cqi_history', [])
    fail_count = state.get('fail_count', [0] * n)
    
    # Build CQI history for trend detection
    cqi_history.append(list(obs))
    if len(cqi_history) > 8:
        cqi_history.pop(0)
    
    # Compute CQI trends with longer lookback
    pred = []
    volatility = []
    trend_strength = []
    for k in range(n):
        if len(cqi_history) >= 3:
            recent = [h[k] for h in cqi_history[-5:]]
            trend = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
            vol = max(0, max(recent) - min(recent))
            volatility.append(vol)
            
            # Measure trend consistency
            if len(recent) >= 3:
                diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
                trend_consistency = 1.0 - (sum(abs(d - trend) for d in diffs) / (len(diffs) * 2.0))
                trend_strength.append(max(0, trend_consistency))
            else:
                trend_strength.append(0.5)
            
            # Conservative trend: use strength-weighted adjustment
            trend_adj = trend * trend_strength[-1] * (1.0 - 0.25 * min(1.0, vol / 2.5))
            pred_c = obs[k] + 0.15 * trend_adj
        else:
            pred_c = obs[k]
            volatility.append(0.0)
            trend_strength.append(0.5)
        
        pred.append(min(CQI_HI, max(CQI_LO, pred_c)))
    
    # Estimate efficiency with volatility-aware scaling
    base_scale = 0.93
    est_eff = []
    for k in range(n):
        vol_factor = 1.0 - 0.18 * min(1.0, volatility[k] / 3.5)
        eff = e_of_cqi(pred[k]) * base_scale * vol_factor
        est_eff.append(eff)
    
    # Track attainment failures for adaptive allocation
    for k in range(n):
        if delivered[k] < T[k] - OK_EPS:
            fail_count[k] += 1
        else:
            fail_count[k] = max(0, fail_count[k] - 1)
    
    # Calculate minimum viable shares with failure history consideration
    min_shares = []
    for k in range(n):
        if est_eff[k] > 0.02:
            # Add margin for slices with repeated failures
            failure_margin = 0.5 + 0.3 * min(1.0, fail_count[k] / 3.0)
            min_s = (T[k] - OK_EPS + failure_margin) / est_eff[k]
            min_shares.append(max(1, int(math.ceil(min_s))))
        else:
            min_shares.append(GCAP)
    
    total_min = sum(min_shares)
    
    candidates = []
    
    # Strategy 1: Prioritize by weighted attainment impact
    # Key insight: maximize sum of (W[k] * attain_k) by focusing on slices where
    # we can achieve attainment with reasonable shares
    
    priority_idx = []
    for k in range(n):
        # Score = weight / shares_needed, adjusted by failure history
        if min_shares[k] < GCAP:
            score = W[k] / (min_shares[k] + 1.0)
            # Boost priority for slices failing frequently
            score *= (1.0 + 0.4 * min(1.0, fail_count[k] / 2.0))
        else:
            score = 0.0
        priority_idx.append((score, k))
    
    priority_idx.sort(reverse=True)
    
    # Allocate greedily by priority
    attain_focused = [0] * n
    remaining = GCAP
    
    for score, k in priority_idx:
        if score > 0:
            alloc = min(min_shares[k], remaining)
            attain_focused[k] = alloc
            remaining -= alloc
    
    # Distribute any leftover capacity to highest-weight slices that need it
    for score, k in priority_idx:
        if remaining <= 0:
            break
        add = min(3, remaining)  # Up to 3 extra shares per slice
        attain_focused[k] += add
        remaining -= add
    
    candidates.append(attain_focused)
    
    # Strategy 2: Efficiency maximization with weight weighting
    base_alloc = [0] * n
    used = 0
    remaining = GCAP
    
    # First, allocate minimum to ensure targets (where achievable)
    for k in range(n):
        if min_shares[k] < GCAP and used + min_shares[k] <= GCAP:
            base_alloc[k] = min_shares[k]
            used += min_shares[k]
            remaining = GCAP - used
    
    # Distribute residual by weight * efficiency
    if remaining > 0:
        weight_eff = [(W[k] * est_eff[k], k) for k in range(n)]
        weight_eff.sort(reverse=True)
        
        for _ in range(remaining):
            best_idx = weight_eff[0][1]
            base_alloc[best_idx] += 1
    
    candidates.append(base_alloc)
    
    # Strategy 3: Conservative: slightly favor lower-weight slices if they're failing
    conservative = [0] * n
    remaining = GCAP
    
    # Allocate in reverse priority order (help struggling slices)
    for score, k in reversed(priority_idx):
        if remaining > 0:
            alloc = min(max(min_shares[k], 8), remaining)
            conservative[k] = alloc
            remaining -= alloc
    
    candidates.append(conservative)
    
    # Strategy 4: Stability with small perturbations
    if prev_action is not None:
        candidates.append(list(prev_action))
        
        # Try small adjustments
        for k in range(n):
            for delta in [-2, -1, 1, 2]:
                perturb = list(prev_action)
                perturb[k] = max(0, perturb[k] + delta)
                if sum(perturb) <= GCAP and perturb not in candidates:
                    candidates.append(list(perturb))
    
    # Strategy 5: Best-effort maximization (minimize guaranteed shares)
    min_guaranteed = [max(0, int(min_shares[k] * 0.7)) for k in range(n)]
    be_max = list(min_guaranteed)
    remaining = GCAP - sum(be_max)
    
    # Add back carefully to prevent failures
    for score, k in priority_idx:
        if remaining <= 0:
            break
        add = min(min_shares[k] - be_max[k], remaining)
        be_max[k] += add
        remaining -= add
    
    candidates.append(be_max)
    
    # Evaluate all candidates
    best_score = float('-inf')
    best_action = [0] * n
    stability_bonus = 0.008
    
    for candidate in candidates:
        if sum(candidate) <= GCAP:
            score = _composite_tick(est_eff, candidate, prev_action)
            
            if prev_action is not None and candidate == prev_action:
                score += stability_bonus
            
            if score > best_score:
                best_score = score
                best_action = list(candidate)
    
    # Enforce constraints
    final_action = best_action
    if sum(final_action) > GCAP:
        total = sum(final_action)
        final_action = [max(0, int(x * GCAP / total)) for x in final_action]
        remainder = GCAP - sum(final_action)
        # Distribute remainder to highest-priority slices
        for _, k in priority_idx:
            if remainder <= 0:
                break
            final_action[k] += 1
            remainder -= 1
    
    # Update state
    state.set('previous_action', list(final_action))
    state.set('cqi_history', cqi_history)
    state.set('fail_count', fail_count)
    
    return final_action