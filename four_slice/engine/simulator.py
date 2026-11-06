"""Four-slice simulator (paper Sec. VI-A, Eq. 11).

Five slices share one pool of 100 integer share-points: four guaranteed slices G1..G4 (weights and
targets below) plus one saturated best-effort slice that takes the remainder. The controller emits four
integer shares s1..s4, sum <= GCAP (= 70); the best-effort slice receives BE = POOL - sum(s).

Guaranteed demand is saturated, so delivered_k = e_k(t) * s_k. Each guaranteed slice's channel is a
phase-staggered fade cycle on CQI in [5, 15] with a sustained deep-null plateau (rotating fades). The
per-point efficiency e_k(CQI) is the curve of Eq. 11: 2.39 Mbps/point at CQI 15 (the two-slice testbed
calibration), collapsing toward a floor at low CQI, so a slice in a deep null cannot be served at any
share. Observations reach the controller one tick late.

At CQI 15 the four minimum funding costs already sum to more than GCAP, so not all four targets can be
met every tick: the controller must choose which slices to fund, and the best subset rotates with the
fades. The reference anchors (analysis/anchors.py) and the scorer (scorer.py) use this module.
"""
import math, random, statistics

# ----------------------------------------------------------------------------- world constants (tunable)
POOL   = 100          # total share-points
GCAP   = 70           # guaranteed cap: sum(s1..s4) <= GCAP  -> BE >= POOL-GCAP = 30  (tuned from 90: scarcity)
NG     = 4            # number of guaranteed slices
# Weights strictly decreasing, targets strictly increasing -> value density w_k/T_k strictly decreasing
# (G1 densest). With this monotone w/T the oracle-vs-greedy gap comes from the channel rotation alone,
# not from a static weight-vs-target misalignment.
W      = (1.00, 0.90, 0.80, 0.70)     # guaranteed weights  G1..G4  (tuned)
T      = (44.0, 46.0, 48.0, 50.0)     # guaranteed targets (Mbps)   (tuned; heaviest slice, lowest rate)
TICKS  = 360
DEV    = list(range(16))              # env seed set (worst-env aggregation is a MIN over these)

# efficiency curve e_k(CQI) (Eq. 11): e(15)=E_MAX (= the two-slice GOOD_SLOPE 2.39), e(5)=E_FLOOR,
# convex ("collapsing toward the floor") with exponent E_EXP.
CQI_LO, CQI_HI = 5.0, 15.0
E_MAX  = 2.39         # Mbps/point at CQI 15  (two-slice simulator GOOD_SLOPE)
E_FLOOR = 0.30        # Mbps/point at CQI 5   (deep-null floor: 0.30*GCAP = 27 < every target -> unaffordable)
E_EXP  = 1.578        # convexity; e(10) ~= 1.00 with these anchors
_E_C   = (E_MAX - E_FLOOR) / (CQI_HI - CQI_LO) ** E_EXP

# best-effort slice: saturated, on a stable good channel (spare-capacity soak). Its per-point rate is fixed
# so the BE composite term is a clean function of the RESIDUAL share -> this is what the pool is traded against.
E_BE   = 2.39
CEIL   = E_BE * POOL          # measured-and-pinned ceiling for BE normalisation (max BE_Mbps = E_BE*POOL)

# composite (generalised paper C_N)
BE_W   = 0.6          # best-effort term weight
CHURN  = 0.02         # penalty per changed share (per tick)
OK_EPS = 0.5          # delivered >= T_k - OK_EPS counts as delivered (binary)

# fade schedule (deterministic waveform -> deep nulls guaranteed & controls re-derivable; + seeded jitter)
DWELL_HI = 10         # ticks pinned at CQI_HI (peak)              (tuned from 15)
RAMP_DN  = 30         # ticks ramping CQI_HI -> CQI_LO             (tuned from 25: more contested mid-band)
DWELL_LO = 10         # ticks pinned at CQI_LO (SUSTAINED DEEP NULL plateau) (tuned from 15; still sustained)
RAMP_UP  = 30         # ticks ramping CQI_LO -> CQI_HI            (tuned from 25)
PERIOD   = DWELL_HI + RAMP_DN + DWELL_LO + RAMP_UP     # = 80
JIT_SD   = 0.5        # CQI jitter sd (applied off the deep-null plateau only)
PHASE_OFF = None      # explicit per-slice phase offsets (ticks); None => auto stagger k*PERIOD/NG


def _recompute_derived():
    """Recompute constants derived from the tunable primitives (call after reconfigure)."""
    global _E_C, CEIL, PERIOD
    _E_C = (E_MAX - E_FLOOR) / (CQI_HI - CQI_LO) ** E_EXP
    CEIL = E_BE * POOL
    PERIOD = DWELL_HI + RAMP_DN + DWELL_LO + RAMP_UP


def reconfigure(**kw):
    """Set any tunable module constant(s) and recompute derived quantities."""
    g = globals()
    for k, v in kw.items():
        assert k in g, f"unknown simulator parameter {k!r}"
        g[k] = v
    _recompute_derived()


def e_of_cqi(cqi):
    """Per-point efficiency (Mbps/point) at a given CQI. Monotone, convex, e(5)=E_FLOOR, e(15)=E_MAX."""
    c = min(max(cqi, CQI_LO), CQI_HI)
    return E_FLOOR + _E_C * (c - CQI_LO) ** E_EXP


def _base_cqi(k, t, align=False):
    """Deterministic staggered fade waveform for guaranteed slice k at tick t (no jitter).

    One PERIOD: DWELL_HI at 15, ramp down to 5, DWELL_LO at 5 (deep null), ramp up to 15. Phase offset
    k*PERIOD/NG staggers the slices so the deep nulls ROTATE. align=True zeroes the stagger (control a)."""
    if align:
        phase = 0
    elif PHASE_OFF is not None:
        phase = PHASE_OFF[k]
    else:
        phase = k * PERIOD // NG
    u = (t + phase) % PERIOD
    if u < DWELL_HI:                                   # peak plateau
        return CQI_HI
    u -= DWELL_HI
    if u < RAMP_DN:                                    # 15 -> 5
        return CQI_HI + (CQI_LO - CQI_HI) * (u / RAMP_DN)
    u -= RAMP_DN
    if u < DWELL_LO:                                   # deep-null plateau
        return CQI_LO
    u -= DWELL_LO
    return CQI_LO + (CQI_HI - CQI_LO) * (u / RAMP_UP)  # 5 -> 15


def cqi_series(seed, align=False, common_phase=False):
    """Per-slice CQI series [k][t]. Deterministic waveform + seeded jitter (jitter suppressed on the deep-null
    plateau so nulls stay 'sustained floor pinning'). common_phase==align (control a: all fades aligned)."""
    align = align or common_phase
    out = []
    for k in range(NG):
        rng = random.Random(4200 + seed * 37 + k)
        row = []
        for t in range(TICKS):
            base = _base_cqi(k, t, align=align)
            if base <= CQI_LO + 1e-9:                  # on the deep-null plateau: keep it deep
                row.append(CQI_LO)
            else:
                row.append(min(max(base + rng.gauss(0.0, JIT_SD), CQI_LO), CQI_HI))
        out.append(row)
    return out


def eff_series(seed, **kw):
    """True per-slice efficiencies e_k(t) [k][t] for the given env seed."""
    cq = cqi_series(seed, **kw)
    return [[e_of_cqi(c) for c in row] for row in cq]


# ----------------------------------------------------------------------------- composite
def deliver(e_t, s):
    """Delivered Mbps per guaranteed slice this tick (demand saturated -> no cap)."""
    return [e_t[k] * s[k] for k in range(NG)]


def be_share(s):
    return POOL - sum(s)


def composite_tick(e_t, s, prev):
    """Per-tick composite contribution C_t for guaranteed shares s (list of NG ints), given true e_k(t).

        C_t = sum_k [ +w_k if delivered_k >= T_k-0.5 else -0.5*w_k ]        (guaranteed, binary)
            + 0.6 * (BE_Mbps / CEIL)                                        (best-effort)
            - 0.02 * (#shares changed vs previous tick)                     (churn)
    """
    d = deliver(e_t, s)
    g = 0.0
    for k in range(NG):
        g += W[k] if d[k] >= T[k] - OK_EPS else -0.5 * W[k]
    be = BE_W * (E_BE * be_share(s)) / CEIL
    changed = 0
    if prev is not None:
        full = list(s) + [be_share(s)]
        pfull = list(prev) + [be_share(prev)]
        changed = sum(1 for a, b in zip(full, pfull) if a != b)
    return g + be - CHURN * changed


def feasible(s):
    """Clamp a raw guaranteed action to non-negative integers with sum <= GCAP (BE absorbs the rest).
    If the raw sum exceeds GCAP, scale down proportionally then integer-round with a largest-remainder pass."""
    s = [max(0, int(round(x))) for x in s]
    tot = sum(s)
    if tot <= GCAP:
        return s
    scaled = [x * GCAP / tot for x in s]
    fl = [int(math.floor(x)) for x in scaled]
    rem = GCAP - sum(fl)
    for _, i in sorted(((scaled[i] - fl[i], i) for i in range(NG)), reverse=True):
        if rem <= 0:
            break
        fl[i] += 1
        rem -= 1
    return fl


def eval_controller(ctrl, seed, lag=True, perfect=False, **kw):
    """Score a controller over one env seed. Returns mean per-tick composite.

    ctrl(obs, state) -> raw 4-vector of guaranteed shares. obs is a dict with 'e' (efficiencies the controller
    is allowed to see) and 'cqi'. With lag=True the controller sees e_k(t-1)/cqi(t-1) (one-tick lag); the
    FIRST tick sees tick 0 (no history). perfect=True hands the controller the TRUE e_k(t) (for the
    perfect-knowledge baselines). Scoring always uses TRUE e_k(t)."""
    E = eff_series(seed, **kw)
    C = cqi_series(seed, **kw)
    state, prev, comps = {}, None, []
    for t in range(TICKS):
        if perfect:
            oe = [E[k][t] for k in range(NG)]
            oc = [C[k][t] for k in range(NG)]
        else:
            ti = t - 1 if (lag and t > 0) else t
            oe = [E[k][ti] for k in range(NG)]
            oc = [C[k][ti] for k in range(NG)]
        s = feasible(ctrl({"e": oe, "cqi": oc, "t": t}, state))
        e_true = [E[k][t] for k in range(NG)]
        comps.append(composite_tick(e_true, s, prev))
        prev = s
    return statistics.fmean(comps)


def worst_env(ctrl, seeds=DEV, **kw):
    """Worst-env aggregation: MIN mean-composite over the env seed set (the paper's fitness)."""
    return min(eval_controller(ctrl, s, **kw) for s in seeds)


# ----------------------------------------------------------------------------- diagnostics
def cost_table_at(cqi):
    """Minimum funding cost ceil(T_k/e_k) for each guaranteed slice at a single CQI (diagnostic)."""
    e = e_of_cqi(cqi)
    return [math.ceil(T[k] / e) for k in range(NG)]


if __name__ == "__main__":
    print(f"four-slice simulator  POOL={POOL} GCAP={GCAP}  W={W}  T={T}  CEIL={CEIL}")
    print(f"efficiency curve: e(5)={e_of_cqi(5):.3f}  e(10)={e_of_cqi(10):.3f}  e(15)={e_of_cqi(15):.3f} Mbps/pt")
    for cq in (15, 13, 11, 10, 8, 5):
        ct = cost_table_at(cq)
        print(f"  CQI {cq:>2}: min-cost G1..G4 = {ct}  sum={sum(ct)}  (GCAP={GCAP})")
    print(f"PERIOD={PERIOD}  deep-null plateau={DWELL_LO} ticks/period  ~{TICKS/PERIOD:.1f} nulls/slice/episode")
