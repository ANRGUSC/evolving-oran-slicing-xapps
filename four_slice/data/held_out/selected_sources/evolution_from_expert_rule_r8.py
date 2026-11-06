import math

# --- env_d constants + functions inlined verbatim so this seed is self-contained ---
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
    """Hand-written expert rule (subset enumeration + CQI-informed min-viable funding + hold candidate),
    tuned settings alpha=0, scale=0.9 -> worst-env 0.5413 on DEV 0-15."""
    alpha, scale = 0.0, 0.9
    n = sla.n
    obs = [(history.slices[k].last().cqi if history.slices[k].last() else 15.0) for k in range(n)]
    prev_cqi = state.get('prior_observed_cqi', None)
    pred = [min(CQI_HI, max(CQI_LO, c + (alpha * (c - prev_cqi[k]) if prev_cqi is not None else 0.0)))
            for k, c in enumerate(obs)]
    est_eff = [e_of_cqi(c) * scale for c in pred]
    required = [max(0, math.ceil((sla.slices[k].target_mbps - OK_EPS) / est_eff[k])) for k in range(n)]
    prev = state.get('previous_action', None)
    candidates = []
    for mask in range(1 << n):
        shares = [required[k] if mask & (1 << k) else 0 for k in range(n)]
        if sum(shares) <= sla.guaranteed_cap:
            candidates.append(shares)
    if prev is not None:
        candidates.append(list(prev))
    selected = max(candidates, key=lambda s: _composite_tick(est_eff, s, prev))
    state.set('prior_observed_cqi', list(obs))
    state.set('previous_action', list(selected))
    return list(selected)
