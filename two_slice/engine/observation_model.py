"""CQI observation model and worst-floor fitness of the two-slice benchmark (paper Sec. V-B).
Delivery physics (s1_null / s1_good / s2_thp, SIGMA) come unchanged from simulator.py; only the CQI
the controller observes is modelled here:
  null CQI ~ clipped-normal(null_floor, 0.5)  [null_floor in {4.5, 5.1 (nominal), 5.8}]
  good CQI ~ clipped-normal(15.0, 0.79)        [measured sd 0.79; clipped to valid CQI 1..15]
Fitness = MIN over the three null-floor environments of the mean composite (worst-floor selection).
"""
import sys, statistics, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
# simulator.py lives in this same engine/ directory.
sys.path.insert(0, str(HERE))
import simulator as B   # physics + schedule + Tick/Hist/Sla/State reused

NULL_FLOOR_NOMINAL = 5.1
NULL_SD = 0.5
GOOD_MEAN, GOOD_SD = 15.0, 0.79
ENVS = (4.5, 5.1, 5.8)

def _clip(x, lo, hi): return lo if x < lo else hi if x > hi else x
def _obs_cqi(rng, is_null, null_floor):
    if is_null: return _clip(rng.gauss(null_floor, NULL_SD), 1.0, 15.0)
    return _clip(rng.gauss(GOOD_MEAN, GOOD_SD), 1.0, 15.0)

def replay(kind, variant, seed=0, null_floor=NULL_FLOOR_NOMINAL, sigma=None, corrupt=None):
    """Mirror of B.replay with the ONLY change being the per-tick CQI observation (sampled, not constant).
    Delivery (d1/d2), scoring, noise and corruptions are those of simulator.py."""
    sig = B.SIGMA if sigma is None else sigma
    rng = random.Random(1000 + seed)
    crng = random.Random(7000 + seed)                 # separate stream for CQI obs -> delivery seed unchanged
    tk, target = B.ticks()
    if corrupt and corrupt[0] == "episode":
        tk = [dict(t) for t in tk]
        for j in corrupt[1]: tk[j]["null"] = not tk[j]["null"]
    sla = B.Sla(target); st = B.State(); h = B.Hist()
    prev_share = 35; prev_s1 = 0.0; prev_s2 = 0.0
    faults = 0
    attain = viol = churn = 0; s2s = []; shares = []; margin_swings = 0; miss96 = tot96 = 0; per_act = {}
    for i, t in enumerate(tk):
        cqi = _obs_cqi(crng, t["null"], null_floor)                       # <-- v3 OBSERVATION MODEL
        h.ue1.push(B.Tick(cqi, prev_share, prev_s1, prev_s1*125000.0))
        h.ue2.push(B.Tick(_obs_cqi(crng, False, null_floor), B.POOL-prev_share, prev_s2, prev_s2*125000.0))
        prev_viol = st.get("violations", 0)
        if kind[0] == "const":      s1 = kind[1]
        elif kind[0] == "oracle":   s1 = 5 if t["null"] else 35
        elif kind[0] == "source":
            try: s1 = B._share_of(kind[1](h, st, sla, {}))
            except Exception: s1 = prev_share; faults += 1
        s1 = max(5, min(90, int(s1))); s2 = B.POOL - s1
        eps = rng.gauss(0.0, sig)
        if t["null"]: d1 = max(0.0, B.s1_null(s1) * (1.0 + eps))
        else:         d1 = max(0.0, min(B.GOOD_SLOPE * s1 * (1.0 + eps), float(t["demand"])))
        d2, _flag = B.s2_thp(s2, variant)
        if corrupt and corrupt[0] == "tick" and i in corrupt[1]: d1 = corrupt[2]
        need = min(t["demand"], target); ok = d1 >= need - 0.5
        attain += ok; viol += (not ok); churn += (s1 != prev_share)
        if (not t["null"]) and t["demand"] == 96: tot96 += 1; miss96 += (not ok)
        if st.get("violations", 0) > prev_viol: margin_swings += 1
        s2s.append(d2); shares.append(s1)
        pa = per_act.setdefault(t["act"], {"share": []}); pa["share"].append(s1)
        prev_share, prev_s1, prev_s2 = s1, d1, d2
    n = len(tk)
    comp = attain/n - B.ALPHA*viol/n + B.S2W*statistics.fmean(s2s)/B.CEIL - B.GAMMA*churn/n
    return {"composite": comp, "attain": attain, "viol": viol, "churn": churn, "n": n,
            "share_sd": statistics.pstdev(shares), "share_distinct": sorted(set(shares)),
            "mean_s2": statistics.fmean(s2s), "shares": shares, "margin_swings": margin_swings,
            "miss96": miss96, "tot96": tot96, "faults": faults,
            "per_act": {a: statistics.fmean(v["share"]) for a, v in per_act.items()}}

def replay_mean(kind, variant, nseeds=30, null_floor=NULL_FLOOR_NOMINAL, sigma=None):
    rs = [replay(kind, variant, seed=s, null_floor=null_floor, sigma=sigma) for s in range(nseeds)]
    return {"composite": statistics.fmean(r["composite"] for r in rs),
            "composite_sd": statistics.pstdev(r["composite"] for r in rs),
            "share_sd": statistics.fmean(r["share_sd"] for r in rs),
            "attain": statistics.fmean(r["attain"] for r in rs),
            "mean_s2": statistics.fmean(r["mean_s2"] for r in rs),
            "margin_swings": statistics.fmean(r["margin_swings"] for r in rs),
            "miss96": statistics.fmean(r["miss96"] for r in rs), "tot96": rs[0]["tot96"],
            "share_distinct_union": sorted(set().union(*[set(r["share_distinct"]) for r in rs]))}

def score_source_env(src_ctrl, variant, null_floor, nseeds=8):
    comps = [replay(("source", src_ctrl), variant, seed=s, null_floor=null_floor)["composite"] for s in range(nseeds)]
    return statistics.fmean(comps)

def fitness_min(src_ctrl, variant="linear", nseeds=8, envs=ENVS):
    """Design choice (c): worst-case selection = MIN composite over the null-floor environments."""
    per = {e: score_source_env(src_ctrl, variant, e, nseeds) for e in envs}
    return min(per.values()), per

if __name__ == "__main__":
    print(f"sensor-calibrated bench: envs={ENVS} null~N(floor,{NULL_SD}) good~N({GOOD_MEAN},{GOOD_SD})")
    for e in ENVS:
        print(f"  frozen const-35 @floor {e}: linear={replay_mean(('const',35),'linear',40,e)['composite']:+.4f}")
