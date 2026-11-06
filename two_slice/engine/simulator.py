"""Two-slice benchmark simulator (paper Eq. 8), calibrated to the POWDER 5G testbed.
One guaranteed slice (SLICE1) plus a best-effort slice (SLICE2), scored over a fixed demand schedule.
Throughput moves via mean-zero multiplicative delivery noise on SLICE1 (the mechanism the live cell
exhibited: CQI stayed saturated, so goodput varied through delivery noise rather than CQI).

Delivery surface (fitted to measured throughputs):
  Good SLICE1 = min(2.39*share*(1+eps_t), demand)   Null SLICE1 = max(0, 0.567*share - 8.1)*(1+eps_t)
  SLICE2      = 2.107*share2 + 31.2                 (deterministic)
  eps_t ~ N(0, SIGMA^2), iid and seeded/reproducible.
Scoring: need = min(demand, target); attained iff thp >= need - 0.5;
  composite = attain - 0.5*viol + 0.6*mean_s2/CEIL - 0.02*churn ; CEIL = 220.
"""
import json, statistics, random
from pathlib import Path
SCHED = Path(__file__).resolve().parent / "demand_schedule.json"
TARGET = 80; POOL = 100; CEIL = 220.0; NULL_CQI = 2; GOOD_CQI = 15
ALPHA, S2W, GAMMA = 0.5, 0.6, 0.02
S2_MEAS_LO, S2_MEAS_HI = 10, 68
GOOD_SLOPE = 2.39
SIGMA = 0.0302   # delivery-noise standard deviation, calibrated to the testbed

def s1_null(s):    return max(0.0, 0.567*float(s) - 8.1)
def s1_good(s, d): return min(GOOD_SLOPE*float(s), float(d))
def s2_thp(share2, variant):
    lin = 2.107*float(share2) + 31.2
    flagged = not (S2_MEAS_LO <= share2 <= S2_MEAS_HI)
    return (min(lin, 200.0) if variant == "cap200" else lin), flagged

class Tick:
    def __init__(s, cqi, prb, mbps, dl_bytes=0.0, t=0.0): s.cqi, s.prb_share, s.mbps, s.dl_bytes, s.t = cqi, prb, mbps, dl_bytes, t
class UEHist:
    def __init__(s): s._t = []
    def last(s): return s._t[-1] if s._t else None
    def latest_mbps(s): return s._t[-1].mbps if s._t else 0.0
    def push(s, t): s._t.append(t)
    @property
    def samples(s): return s._t
class Hist:
    def __init__(s): s.ue1, s.ue2 = UEHist(), UEHist()
    @property
    def slices(s): return [s.ue1, s.ue2]          # cold-draw prompt interface: history.slices[k]
class SlaOne:
    def __init__(s, tgt, mx=90): s.min_share, s.max_share, s.target_mbps = 5, mx, tgt
class Sla:
    n = 2
    def __init__(s, tgt): s.pool_max = POOL; s.sla_ue1, s.sla_ue2 = SlaOne(tgt), SlaOne(0)
    @property
    def priority(s): return s.sla_ue1             # cold-draw prompt interface: sla.priority.*
    @property
    def best_effort(s): return [s.sla_ue2]
class State:
    def __init__(s): s.d = {}
    def get(s, k, d=None): return s.d.get(k, d)
    def set(s, k, v): s.d[k] = v

def ticks(sched_path=SCHED):
    s = json.loads(sched_path.read_text()); out = []
    for seg in s["segments"]:
        for _ in range(seg["dur"]):
            out.append({"demand": seg["s1_rate_mbps"], "act": seg["act"], "null": seg["act"] == "B_channel_null"})
    return out, s["sla_target_mbps"]

def _share_of(o): return int(o[0]) if isinstance(o, (tuple, list)) else int(o)

def replay(kind, variant, seed=0, sigma=None, corrupt=None):
    """One noise realization. kind: ('const', share) | ('oracle',) | ('source', controller_fn)."""
    sig = SIGMA if sigma is None else sigma
    rng = random.Random(1000 + seed)
    tk, target = ticks()
    if corrupt and corrupt[0] == "episode":
        tk = [dict(t) for t in tk]
        for j in corrupt[1]:
            tk[j]["null"] = not tk[j]["null"]
    sla = Sla(target); st = State(); h = Hist()
    prev_share = 35; prev_s1 = 0.0; prev_s2 = 0.0
    ctrl = None; pred = {}
    faults = 0
    attain = viol = churn = 0; s2s = []; shares = []; margin_swings = 0; miss96 = tot96 = 0
    per_act = {}
    for i, t in enumerate(tk):
        cqi = NULL_CQI if t["null"] else GOOD_CQI
        # feed per-tick dl_bytes = mbps*1e6/8 so a controller doing dl_bytes*8/1e6 (the prompt's
        # documented per-tick interface) reads true Mbps.
        h.ue1.push(Tick(cqi, prev_share, prev_s1, prev_s1*125000.0))
        h.ue2.push(Tick(GOOD_CQI, POOL-prev_share, prev_s2, prev_s2*125000.0))
        prev_viol = st.get("violations", 0)
        if kind[0] == "const":      s1 = kind[1]
        elif kind[0] == "oracle":   s1 = 5 if t["null"] else 35
        elif kind[0] == "source":                                   # arbitrary drawn controller
            try: s1 = _share_of(kind[1](h, st, sla, {}))
            except Exception: s1 = prev_share; faults += 1
        s1 = max(5, min(90, int(s1))); s2 = POOL - s1
        eps = rng.gauss(0.0, sig)                                  # delivery noise on CAPACITY (both regimes)
        if t["null"]:
            d1 = max(0.0, s1_null(s1) * (1.0 + eps))
        else:
            d1 = max(0.0, min(GOOD_SLOPE * s1 * (1.0 + eps), float(t["demand"])))  # noise then demand cap
            # -> demand-24 ticks are demand-limited (capacity>>24) hence noise-immune, matching live;
            #    only demand-96 ticks (capacity ~ need) feel the noise -> the ~5% boundary misses
        d2, _flag = s2_thp(s2, variant)
        if corrupt and corrupt[0] == "tick" and i in corrupt[1]: d1 = corrupt[2]
        need = min(t["demand"], target); ok = d1 >= need - 0.5
        attain += ok; viol += (not ok); churn += (s1 != prev_share)
        if (not t["null"]) and t["demand"] == 96:
            tot96 += 1; miss96 += (not ok)
        if st.get("violations", 0) > prev_viol: margin_swings += 1     # some controllers track 'violations' in state
        s2s.append(d2); shares.append(s1)
        pa = per_act.setdefault(t["act"], {"share": []}); pa["share"].append(s1)
        prev_share, prev_s1, prev_s2 = s1, d1, d2
    n = len(tk)
    comp = attain/n - ALPHA*viol/n + S2W*statistics.fmean(s2s)/CEIL - GAMMA*churn/n
    return {"composite": comp, "attain": attain, "viol": viol, "churn": churn, "n": n,
            "share_sd": statistics.pstdev(shares), "share_distinct": sorted(set(shares)),
            "mean_s2": statistics.fmean(s2s), "shares": shares, "margin_swings": margin_swings,
            "miss96": miss96, "tot96": tot96, "faults": faults,
            "per_act": {a: statistics.fmean(v["share"]) for a, v in per_act.items()}}

def score_source(src, variant="linear", nseeds=10):
    """Score an arbitrary drawn controller on the bench. A load failure or >5% runtime
    fault rate -> UNSCOREABLE (recorded, never silently fixed)."""
    ns = {}
    try:
        exec(src, ns)
    except Exception as e:
        return {"status": "unscoreable", "reason": f"load:{type(e).__name__}"}
    ctrl = ns.get("controller")
    if ctrl is None:
        return {"status": "unscoreable", "reason": "no_controller"}
    comps = []; fault_ticks = 0; total_ticks = 0
    for s in range(nseeds):
        try:
            r = replay(("source", ctrl), variant, seed=s)
        except Exception as e:
            return {"status": "unscoreable", "reason": f"replay:{type(e).__name__}"}
        comps.append(r["composite"]); fault_ticks += r["faults"]; total_ticks += r["n"]
    fr = fault_ticks / max(1, total_ticks)
    if fr > 0.05:
        return {"status": "unscoreable", "reason": f"fault_rate={fr:.0%}"}
    return {"status": "ok", "composite": statistics.fmean(comps), "fault_rate": fr}

def replay_mean(kind, variant, nseeds=30, sigma=None):
    """Average metrics over nseeds noise realizations (stable gate metrics)."""
    rs = [replay(kind, variant, seed=s, sigma=sigma) for s in range(nseeds)]
    agg = {"composite": statistics.fmean(r["composite"] for r in rs),
           "composite_sd": statistics.pstdev(r["composite"] for r in rs),
           "share_sd": statistics.fmean(r["share_sd"] for r in rs),
           "attain": statistics.fmean(r["attain"] for r in rs),
           "mean_s2": statistics.fmean(r["mean_s2"] for r in rs),
           "margin_swings": statistics.fmean(r["margin_swings"] for r in rs),
           "miss96": statistics.fmean(r["miss96"] for r in rs),
           "tot96": rs[0]["tot96"],
           "share_distinct_union": sorted(set().union(*[set(r["share_distinct"]) for r in rs]))}
    return agg

if __name__ == "__main__":
    print("static share=35 linear mean composite:", round(replay_mean(("const",35),"linear")["composite"],3))
