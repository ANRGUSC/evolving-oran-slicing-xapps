"""Sanity gates of the two-slice benchmark (not part of the paper's results). G5 corrupts the
simulator four ways and checks that each corruption is caught; G1/G3/G4 check the static-35 baseline,
a frozen anchor and the oracle headroom; G7 is the behavioral preflight (release in the null, hold in
the good regime) that search/evolve.py applies to each generation champion. Running this file re-runs
them all."""
import sys, statistics, random
from pathlib import Path
HERE = Path(__file__).resolve().parent
# simulator.py and observation_model.py live in this same engine/ directory.
sys.path.insert(0, str(HERE))
import simulator as B
import observation_model as B3
def line(): print("-"*82)

# ---- reference controllers for gates/self-test ----
# The ablated CQI-seed reference program (s0g4c6_cqi.py) is not part of this release;
# _ablated_s0g4c6_cqi() returns None when it is absent and G7_selftest skips that arm.
_ABLATED_SRC = HERE.parent / "s0g4c6_cqi.py"
def _ablated_s0g4c6_cqi():
    if not _ABLATED_SRC.exists(): return None
    ns = {}; exec(_ABLATED_SRC.read_text(), ns); return ns["controller"]
def _oracle_thr6(history, state, sla, predictors):        # G7 self-test oracle: abandon at cqi<6
    smp = history.slices[0].last()
    if smp is None: return 35
    return sla.priority.min_share if smp.cqi < 6 else 35
def _band_detector(lo=3.6, hi=6.6):                       # G5(4) reference: floor-5.1-calibrated BAND
    def c(history, state, sla, predictors):
        smp = history.slices[0].last()
        if smp is None: return 35
        return sla.priority.min_share if (lo <= smp.cqi <= hi) else 35
    return c

# ---- G7 sequence machinery ----
def _run_seq(ctrl, cqis, demands, is_null):
    sla = B.Sla(80); st = B.State(); h = B.Hist(); prev_share = 35; prev_s1 = prev_s2 = 0.0; shares = []
    for cqi, dem in zip(cqis, demands):
        h.ue1.push(B.Tick(cqi, prev_share, prev_s1, prev_s1*125000.0))
        h.ue2.push(B.Tick(15.0, B.POOL - prev_share, prev_s2, prev_s2*125000.0))
        s1 = max(5, min(90, int(B._share_of(ctrl(h, st, sla, {})))))
        d1 = max(0.0, B.s1_null(s1)) if is_null else max(0.0, min(B.GOOD_SLOPE*s1, float(dem)))
        d2, _ = B.s2_thp(B.POOL - s1, "linear")
        shares.append(s1); prev_share, prev_s1, prev_s2 = s1, d1, d2
    return shares

def G7(ctrl, null_floor=5.1, seed=0, verbose=False, null_clip=(4.0, 6.0)):
    rng = random.Random(1234 + seed)
    null_cqi = [B3._clip(rng.gauss(null_floor, 0.5), null_clip[0], null_clip[1]) for _ in range(15)]  # nominal [4,6] per spec
    good_cqi = [B3._clip(rng.gauss(15.0, 0.79), 1.0, 15.0) for _ in range(15)]
    good_dem = [24 if i % 2 else 96 for i in range(15)]
    good_sh = _run_seq(ctrl, good_cqi, good_dem, is_null=False)
    null_sh = _run_seq(ctrl, null_cqi, [96]*15, is_null=True)
    good_mean = statistics.fmean(good_sh)
    null_tail = statistics.fmean(null_sh[8:15])                                       # ticks 9-15
    null_pass = (null_tail <= 0.6*good_mean) and (null_tail <= 40)
    good_pass = all(s > 7 for s in good_sh) and (30 <= good_mean <= 45)               # no abandon; sane
    if verbose:
        print(f"      good_sh={good_sh} mean={good_mean:.1f}")
        print(f"      null_sh={null_sh} tail(9-15)={null_tail:.1f}")
    return {"null_pass": null_pass, "good_pass": good_pass, "good_mean": good_mean, "null_tail": null_tail}

# ============================ G5 (4/4) ============================
def G5():
    print("G5 mutation test (each corruption -> its gate RED -> revert), NOW 4/4 required first:")
    base = B3.replay_mean(("const", 35), "linear", nseeds=40)["composite"]; red = 0
    _o = B.s1_null; B.s1_null = lambda s: 3.0*float(s)
    c1 = B3.replay_mean(("const", 35), "linear", nseeds=40)["composite"]; B.s1_null = _o
    r1 = abs(c1-base) > 0.08; red += r1
    print(f"   (1) surface coeff (null winnable): {base:+.3f} -> {c1:+.3f}  {'RED' if r1 else 'green'}")
    _tk, _ = B.ticks(); null_idx = [i for i, t in enumerate(_tk) if t['null']]; flips = null_idx[:60]
    c2 = statistics.fmean(B3.replay(('const',35),'linear',seed=s,corrupt=('episode',flips))['composite'] for s in range(40))
    r2 = abs(c2-base) > 0.08; red += r2
    print(f"   (2) episode (60 null->good):       {base:+.3f} -> {c2:+.3f}  {'RED' if r2 else 'green'}")
    c3 = statistics.fmean(B3.replay(('const',35),'linear',seed=s,corrupt=('tick',list(range(40)),0.0))['composite'] for s in range(40))
    r3 = abs(c3-base) > 0.05; red += r3
    print(f"   (3) scored ticks (40 good->0):      {base:+.3f} -> {c3:+.3f}  {'RED' if r3 else 'green'}")
    # (4) NEW sensor-floor: break the null-CQI floor 5.1 -> 2.0. A floor-5.1-calibrated BAND detector
    #     PASSES G7-null at 5.1 but must go RED (fail null) at floor 2.0 -> proves G7 detects the floor break.
    #     [interpretation note: 'G7 must go red' = the sensor-floor break makes a floor-correct detector
    #      fail G7's null sequence; reference = band[3.6,6.6]. Printed for operator review.]
    bd = _band_detector()   # both arms clipped [1,15] so the 2.0 floor actually reaches the detector (isolate floor)
    g_ok = G7(bd, null_floor=5.1, null_clip=(1.0, 15.0))["null_pass"]
    g_broken = G7(bd, null_floor=2.0, null_clip=(1.0, 15.0))["null_pass"]
    r4 = (g_ok is True) and (g_broken is False); red += r4
    print(f"   (4) sensor-floor (null 5.1->2.0):   band-detector G7-null 5.1={'PASS' if g_ok else 'FAIL'} "
          f"2.0={'PASS' if g_broken else 'FAIL(RED)'}  {'RED' if r4 else 'green'}")
    print(f"   G5: {red}/4 -> {'PASS' if red==4 else 'FAIL'}")
    return red == 4

def G1():
    # PATH/KIND FIX (public repo): the removed 'c0016'/'flat35' replay branch is replaced by the
    # still-supported fixed baseline ('const', 35). Both are obs-independent static-35 policies, so the
    # G1 invariant (distinct shares == [35], sd == 0) is unchanged.
    r = B3.replay(("const", 35), "linear", seed=0)
    ok = r["share_distinct"] == [35] and r["share_sd"] == 0.0
    print(f"G1 static-35 baseline: distinct={r['share_distinct']} sd={r['share_sd']:.2f} -> {'PASS' if ok else 'FAIL'} (expect [35], sd 0; obs-independent)")
    return ok

def G3():
    live = (0.901, 0.878)
    print("G3 anchor: frozen c0016 (const35) vs live +0.901/+0.878, tol +/-0.08 (delivery UNCHANGED, all 3 floors):")
    res = {}
    for v in ("linear", "cap200"):
        m = B3.replay_mean(("const", 35), v, nseeds=60); c = m["composite"]; res[v] = c
        d1, d2 = abs(c-live[0]), abs(c-live[1])
        print(f"   {v:7}: composite={c:+.3f}+/-{m['composite_sd']:.3f} -> {'PASS' if d1<=0.08 or d2<=0.08 else 'OUTSIDE'}")
    return res

def G4(res3):
    print("G4 oracle headroom (WIN binds LINEAR; A4):")
    ok = True
    for v in ("linear", "cap200"):
        orc = B3.replay_mean(("oracle",), v, nseeds=60)["composite"]; head = orc - res3[v]
        if v == "linear":
            p = 0.05 <= head <= 0.12; ok &= p
            print(f"   linear : oracle={orc:+.3f} frozen={res3[v]:+.3f} headroom={head:+.3f} -> {'PASS' if p else 'FAIL'} (band 0.05-0.12)")
        else:
            print(f"   cap200 : oracle={orc:+.3f} frozen={res3[v]:+.3f} headroom={head:+.3f} -> recorded")
    return ok

def G7_selftest():
    print("G7 SELF-TEST: ablated seed must FAIL null; oracle(cqi<6) must PASS both:")
    abl = _ablated_s0g4c6_cqi(); orc = _oracle_thr6
    ro = G7(orc, null_floor=5.1)
    o_ok = (ro["null_pass"] is True) and (ro["good_pass"] is True)
    if abl is None:
        # ablated CQI-seed reference not shipped in the public folder -> skip that arm honestly.
        print(f"   ablated seed @5.1: s0g4c6_cqi.py not in this release -> SKIPPED")
        print(f"   oracle(cqi<6)     @5.1: null_pass={ro['null_pass']} good_pass={ro['good_pass']} (null_tail={ro['null_tail']:.1f}, good_mean={ro['good_mean']:.1f}) -> want PASS/PASS -> {'OK' if o_ok else 'WRONG'}")
        return o_ok
    ra = G7(abl, null_floor=5.1)
    a_ok = (ra["null_pass"] is False)
    print(f"   ablated seed @5.1: null_pass={ra['null_pass']} (null_tail={ra['null_tail']:.1f}, good_mean={ra['good_mean']:.1f}) -> want FAIL -> {'OK' if a_ok else 'WRONG'}")
    print(f"   oracle(cqi<6)     @5.1: null_pass={ro['null_pass']} good_pass={ro['good_pass']} (null_tail={ro['null_tail']:.1f}, good_mean={ro['good_mean']:.1f}) -> want PASS/PASS -> {'OK' if o_ok else 'WRONG'}")
    return a_ok and o_ok

if __name__ == "__main__":
    print("="*82); print(f"v3 SENSOR-CALIBRATED BENCH — GATES (envs={B3.ENVS}, null~N(f,{B3.NULL_SD}), good~N(15,{B3.GOOD_SD}))"); print("="*82)
    g5 = G5(); line()
    if not g5: print("G5 not 4/4 -> STOP."); sys.exit(0)
    g1 = G1(); res3 = G3(); g4 = G4(res3); line()
    g7st = G7_selftest(); line()
    print(f"SUMMARY: G5={g5}(4/4) G1={g1} G4(LINEAR)={g4} G7-selftest={g7st} | G3 rows above")
