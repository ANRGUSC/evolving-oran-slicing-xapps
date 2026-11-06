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
    """Enhanced controller with improved CQI prediction, dynamic risk-aware allocation,
    and exploration-exploitation balance for sustained high score."""
    
    n = sla.n
    
    # Adaptive smoothing parameters based on channel stability
    alpha = 0.25  # increased prediction weight
    scale = 0.88  # slightly less conservative
    momentum = 0.65  # more responsive to changes while reducing churn
    
    # Observe current CQI and delivered throughput
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    obs_mbps = [(history.slices[k].latest_mbps() or 0.0) for k in range(n)]
    
    prev_cqi = state.get('prior_observed_cqi', None)
    prev_action = state.get('previous_action', None)
    prev_mbps = state.get('prior_mbps', None)
    episode_count = state.get('episode_count', 0)
    
    # Enhanced CQI prediction with velocity and stability detection
    pred = []
    velocity = []
    stability = []
    
    for k, c in enumerate(obs):
        if prev_cqi is not None:
            delta = c - prev_cqi[k]
            velocity.append(delta)
            # Predict with momentum in CQI trajectory
            predicted = c + alpha * delta
        else:
            velocity.append(0.0)
            predicted = c
        
        # Stability: low velocity = more stable channel
        stab = 1.0 - min(1.0, abs(velocity[-1]) / 3.0)
        stability.append(stab)
        
        pred.append(min(CQI_HI, max(CQI_LO, predicted)))
    
    # Estimate effective rates with adaptive scale based on stability
    est_eff = []
    for k, c in enumerate(pred):
        base_eff = e_of_cqi(c)
        # More conservative on unstable channels
        adaptive_scale = scale * (0.85 + 0.15 * stability[k])
        est_eff.append(base_eff * adaptive_scale)
    
    # Calculate minimum viable shares considering historical delivery
    required = []
    for k in range(n):
        if est_eff[k] > 0.01:
            min_share = max(0, math.ceil((T[k] - OK_EPS) / est_eff[k]))
        else:
            min_share = GCAP  # channel very bad, needs full cap to attempt target
        required.append(min_share)
    
    # Generate diverse candidate allocations
    candidates = []
    
    # 1. Minimal required allocation
    min_alloc = [required[k] for k in range(n)]
    if sum(min_alloc) <= GCAP:
        candidates.append(min_alloc)
    
    # 2. Weighted greedy by per-share efficiency and weight product
    total_required = sum(required)
    if total_required > GCAP:
        # Weighted priority: high weight + high efficiency slices get priority
        efficiency_weight = [(W[k] * est_eff[k] / (T[k] + 1), k) for k in range(n)]
        efficiency_weight.sort(reverse=True)
        greedy_shares = [0] * n
        remaining = GCAP
        for _, k in efficiency_weight:
            alloc = min(required[k], remaining)
            greedy_shares[k] = alloc
            remaining -= alloc
        candidates.append(greedy_shares)
    
    # 3. Equal-proportional across targets (load-balanced)
    total_targets = sum(T)
    balanced = [max(required[k], int(GCAP * T[k] / total_targets)) for k in range(n)]
    s_bal = sum(balanced)
    if s_bal > GCAP:
        scale_factor = GCAP / s_bal
        balanced = [int(balanced[k] * scale_factor) for k in range(n)]
    if sum(balanced) <= GCAP:
        candidates.append(balanced)
    
    # 4. Maximize best-effort (minimum guaranteed that still meets some targets)
    min_guaranteed = [min(required[k], max(1, int(GCAP * 0.1))) for k in range(n)]
    if sum(min_guaranteed) <= GCAP:
        candidates.append(min_guaranteed)
    
    # 5. Inertia: keep previous action with small perturbations
    if prev_action is not None:
        candidates.append(list(prev_action))
        # Tiny adjustments around previous
        for k in range(n):
            perturbed = list(prev_action)
            for delta in [-2, -1, 1, 2]:
                perturbed[k] = max(0, min(GCAP, prev_action[k] + delta))
                if sum(perturbed) <= GCAP:
                    candidates.append(list(perturbed))
                perturbed = list(prev_action)
    
    # 6. Risk-aware: on high instability, bias toward allocations that leave BE room
    if any(s < 0.5 for s in stability):
        conservative = [max(0, int(required[k] * 0.9)) for k in range(n)]
        if sum(conservative) <= GCAP:
            candidates.append(conservative)
    
    # Remove duplicates and invalid candidates
    candidates = [c for c in candidates if sum(c) <= GCAP and all(x >= 0 for x in c)]
    candidates = [list(x) for x in set(tuple(c) for c in candidates)]
    
    if not candidates:
        candidates = [[0] * n]
    
    # Score each candidate
    best_score = -float('inf')
    best_shares = candidates[0]
    
    for s in candidates:
        score = _composite_tick(est_eff, s, prev_action)
        if score > best_score:
            best_score = score
            best_shares = s
    
    # Smooth with momentum, but allow larger jumps if score improves significantly
    if prev_action is not None:
        smoothed = [int(momentum * prev_action[k] + (1 - momentum) * best_shares[k]) 
                    for k in range(n)]
        # Clamp to cap
        s_smooth = sum(smoothed)
        if s_smooth > GCAP:
            scale_factor = GCAP / s_smooth
            smoothed = [int(smoothed[k] * scale_factor) for k in range(n)]
        
        smooth_score = _composite_tick(est_eff, smoothed, prev_action)
        # Accept smoothed if very close to best, otherwise use best (reduce over-smoothing)
        if smooth_score > best_score - 0.08:
            best_shares = smoothed
    
    # Update state
    state.set('prior_observed_cqi', list(obs))
    state.set('previous_action', list(best_shares))
    state.set('prior_mbps', list(obs_mbps))
    state.set('episode_count', episode_count + 1)
    
    return list(best_shares)