"""EXP11 Env D — INJECTED SEED (GEN-0 scoring pool only; NOT shown in any prompt).

A hand-written FACTORED per-slice threshold holder derived from the Phase-1 best factored baseline
(taus = (9, 9, 11, 11); worst-env composite best_factored = -0.0485). It is competent — it knows the
channel physics (a monotone CQI->rate estimate) and funds each slice its minimum viable share when that
slice's own CQI clears its threshold — but it is DECOUPLED: each slice's decision depends only on its OWN
CQI, never on the other three. That is exactly the tested dimension (cross-slice arbitration), so its
coupling-probe rows are flat by construction. eta ~ 0 by design (it reproduces the factored baseline).

Interface = the Phase-2 prompt's controller(history, state, sla, predictors) -> list of 4 guaranteed shares.
"""
import math

TAU = (9.0, 9.0, 11.0, 11.0)          # per-slice CQI thresholds (Phase-1 best factored, taus 9/9/11/11)


def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate (the sealed calibrated curve; the seed 'knows' the physics)."""
    c = 5.0 if cqi < 5.0 else 15.0 if cqi > 15.0 else cqi
    return 0.30 + ((2.39 - 0.30) / (10.0 ** 1.578)) * (c - 5.0) ** 1.578


def controller(history, state, sla, predictors):
    out = []
    for k in range(sla.n):
        h = history.slices[k].last()
        cqi = h.cqi if h is not None else 15.0
        tgt = sla.slices[k].target_mbps
        if cqi >= TAU[k]:
            rate = _rate_of_cqi(cqi)
            out.append(int(math.ceil(tgt / rate)) if rate > 0 else sla.guaranteed_cap)
        else:
            out.append(0)                 # own CQI below threshold -> release (per-slice, no cross-slice look)
    return out
