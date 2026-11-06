"""Two-slice controller fitness: minimum mean episode score across CQI floors."""
import os, sys, statistics
# simulator.py and observation_model.py live in this same engine/ directory.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path: sys.path.insert(0, HERE)
import simulator as B, observation_model as B3

DEV = list(range(16)); ENVS = B3.ENVS; VARIANTS = ("linear", "cap200")
SLA_MIN = 5                                   # sla.priority.min_share (bench SlaOne)
CANON = dict(variant="linear", seed=0, null_floor=5.1)   # canonical trace realization (declared)

def _null_mask():
    tk, _ = B.ticks(); return [bool(t["null"]) for t in tk]
NULL_MASK = _null_mask()

def score_source_v3(src, *, seeds=DEV, envs=ENVS, include_trace=True):
    """Execute a controller on every requested floor, then select the lowest mean."""
    seeds, envs = tuple(seeds), tuple(envs)
    if not seeds or not envs:
        raise ValueError("At least one seed and CQI floor are required")
    ns = {}
    try:
        exec(src, ns)
    except Exception as e:
        return {"status": "VOID", "reason": f"load:{type(e).__name__}"}
    ctrl = ns.get("controller")
    if ctrl is None:
        return {"status": "VOID", "reason": "no_controller"}
    per = {}
    for floor in envs:                                   # per-floor fault gate, as in the evolution
        comps = []; ft = tt = 0
        for v in VARIANTS:
            for s in seeds:
                try:
                    r = B3.replay(("source", ctrl), v, seed=s, null_floor=floor)
                except Exception as e:
                    return {"status": "VOID", "reason": f"replay:{type(e).__name__}"}
                comps.append(r["composite"]); ft += r["faults"]; tt += r["n"]
        if ft / max(1, tt) > 0.05:
            return {"status": "VOID", "reason": f"fault_rate={ft/max(1,tt):.0%}@floor{floor}"}
        per[floor] = statistics.fmean(comps)
    comp = min(per.values())                             # worst-env selection metric (== fitness)
    result = {"status": "ok", "composite": comp,
              "per_env": {str(k): v for k, v in per.items()},
              "worst_floor": min(per, key=per.get)}
    if include_trace:
        try:
            result["trace"] = [int(x) for x in B3.replay(
                ("source", ctrl), CANON["variant"], seed=CANON["seed"],
                null_floor=CANON["null_floor"])["shares"]]
        except Exception as e:
            return {"status": "VOID", "reason": f"trace:{type(e).__name__}"}
    return result


def score_static_share(share, *, seeds=DEV, envs=ENVS):
    """Use exactly the controller evaluation objective for an admissible constant share."""
    bounds = B.Sla(B.TARGET).priority
    if type(share) is not int or not bounds.min_share <= share <= bounds.max_share:
        raise ValueError("Static share must be an admissible integer")
    seeds, envs = tuple(seeds), tuple(envs)
    if not seeds or not envs:
        raise ValueError("At least one seed and CQI floor are required")
    return min(statistics.fmean(
        B3.replay(("const", share), variant, seed=seed, null_floor=floor)["composite"]
        for variant in VARIANTS for seed in seeds
    ) for floor in envs)


def classify_null(trace):
    """From the ACTION TRACE only: ABANDONS iff min null-tick share reaches s_min; HOLDS iff null mean
    within +/-5 of good mean; else OTHER (incl. increasing in the null)."""
    nsh = [s for s, m in zip(trace, NULL_MASK) if m]
    gsh = [s for s, m in zip(trace, NULL_MASK) if not m]
    if not nsh or not gsh: return "OTHER"
    nmean = statistics.fmean(nsh); gmean = statistics.fmean(gsh)
    if min(nsh) <= SLA_MIN: return "ABANDONS"
    if abs(nmean - gmean) <= 5: return "HOLDS"
    return "OTHER"

if __name__ == "__main__":   # smoke: score the evolved champion in the public layout
    champ = os.path.join(os.path.dirname(HERE), "controllers", "evolved_claude.py")
    ok = score_source_v3(open(champ).read())
    print("evolved_claude:", ok["status"], round(ok["composite"], 4), classify_null(ok["trace"]),
          "null_mask_ticks=", sum(NULL_MASK), "/", len(NULL_MASK))
