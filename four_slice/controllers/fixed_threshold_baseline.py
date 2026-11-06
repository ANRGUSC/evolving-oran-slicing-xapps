"""Independent CQI-threshold allocation with the calibrated efficiency curve."""
import math

TAU = (9.0, 9.0, 11.0, 11.0)          # per-slice CQI thresholds


def _rate_of_cqi(cqi):
    """Monotone CQI->Mbps/share estimate (the calibrated curve; the seed 'knows' the physics)."""
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
