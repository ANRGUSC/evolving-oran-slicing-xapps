#!/usr/bin/env python3
"""
Offline reproduction of the editability result (paper Sec. V-G: "Editability: Repairing a
Calibration Error").

What this does (NO network, NO API calls):
  1. Loads the evolved two-slice controller (../two_slice/controllers/evolved_claude.py) and
     applies the one-line provisioning-margin edit of patches/E1_provisioning_margin.diff:
         prb_needed = target_mbps / mbps_per_prb
      -> prb_needed = target_mbps / (mbps_per_prb * 0.9)
  2. Evaluates the original and the edited controller with the two-slice simulator and
     observation model (../two_slice/engine), on the 96 development episodes (3 CQI floors x
     2 best-effort profiles x 16 seeds), under
       - the nominal delivery model (good-channel efficiency 2.39 Mbps per share point), and
       - a -10% efficiency stress (actual good-channel efficiency 0.9 x 2.39, controller unchanged).
  3. Prints the good-channel SLA miss rate, the worst-floor composite and the mean best-effort
     throughput of each case next to the paper's values, and self-checks them.

Self-check: exits non-zero unless the three miss rates are within 0.1 point of 2.4 / 79.9 / 2.2 %
and the best-effort change is within 0.1 Mbps of -6.3.

Run from inside this folder:
    python reproduce.py
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TWO_SLICE = os.path.join(os.path.dirname(HERE), "two_slice")
sys.path.insert(0, os.path.join(TWO_SLICE, "engine"))

import simulator as B           # noqa: E402
import observation_model as B3  # noqa: E402
import scorer as Z              # noqa: E402  (ENVS, VARIANTS, DEV)

CONTROLLER = os.path.join(TWO_SLICE, "controllers", "evolved_claude.py")
OLD = "    prb_needed = target_mbps / mbps_per_prb\n"
NEW = "    prb_needed = target_mbps / (mbps_per_prb * 0.9)  # provisioning margin (efficiency overestimation)\n"

PAPER = {"miss_nominal_orig": 2.4, "miss_stress_orig": 79.9, "miss_stress_edit": 2.2, "be_delta": -6.3}


def load(src):
    ns = {}
    exec(compile(src, "controller", "exec"), ns)
    return ns["controller"]


def evaluate(ctrl):
    """96 development episodes: worst-floor composite, good-channel SLA miss rate, mean BE throughput."""
    per_floor, miss, tot, be = {}, 0, 0, []
    for floor in Z.ENVS:
        comps = []
        for variant in Z.VARIANTS:
            for seed in Z.DEV:
                r = B3.replay(("source", ctrl), variant, seed=seed, null_floor=floor)
                comps.append(r["composite"])
                miss += r["miss96"]            # good-channel ticks where the 80 Mbps target was missed
                tot += r["tot96"]
                be.append(r["mean_s2"])
        per_floor[floor] = statistics.fmean(comps)
    return {"composite": min(per_floor.values()), "miss_pct": 100.0 * miss / tot,
            "be_mbps": statistics.fmean(be)}


def main():
    original = open(CONTROLLER).read()
    if original.count(OLD) != 1:
        print("FAIL: the provisioning line was not found exactly once in evolved_claude.py")
        return 1
    edited = original.replace(OLD, NEW, 1)

    print("=" * 78)
    print("EDITABILITY: REPAIRING A CALIBRATION ERROR  (paper Sec. V-G)")
    print("=" * 78)
    print("controller : two_slice/controllers/evolved_claude.py (fitness 0.9398)")
    print("edit       : patches/E1_provisioning_margin.diff  (divide the efficiency estimate by 0.9 -> ~11% more share)")
    print("evaluation : 96 development episodes; stress = actual good-channel efficiency reduced by 10%")
    print()

    nominal_slope = B.GOOD_SLOPE
    results = {}
    for cond, slope in (("nominal", nominal_slope), ("stress", 0.9 * nominal_slope)):
        B.GOOD_SLOPE = slope
        for name, src in (("original", original), ("edited", edited)):
            results[(cond, name)] = evaluate(load(src))
    B.GOOD_SLOPE = nominal_slope

    print(f"    {'condition':<22}{'controller':<12}{'SLA miss %':>11}{'composite':>11}{'BE Mbps':>10}     paper")
    print("-" * 78)
    rows = [("nominal (eff 2.39)", "original", PAPER["miss_nominal_orig"]),
            ("stress (eff -10%)", "original", PAPER["miss_stress_orig"]),
            ("stress (eff -10%)", "edited", PAPER["miss_stress_edit"]),
            ("nominal (eff 2.39)", "edited", None)]
    ok = True
    for cond_label, name, paper_miss in rows:
        r = results[(cond_label.split()[0], name)]
        if paper_miss is None:
            flag, paper_txt = "", "--"
        else:
            flag = "" if abs(r["miss_pct"] - paper_miss) <= 0.1 else "   <-- MISMATCH"
            paper_txt = f"{paper_miss:.1f}%"
        ok &= flag == ""
        print(f"    {cond_label:<22}{name:<12}{r['miss_pct']:>11.1f}{r['composite']:>11.4f}{r['be_mbps']:>10.1f}     {paper_txt}{flag}")
    be_delta = results[("nominal", "edited")]["be_mbps"] - results[("nominal", "original")]["be_mbps"]
    flag = "" if abs(be_delta - PAPER["be_delta"]) <= 0.1 else "   <-- MISMATCH"
    ok &= flag == ""
    print()
    print(f"    best-effort throughput, edited - original: {be_delta:+.1f} Mbps     paper: {PAPER['be_delta']:+.1f} Mbps{flag}")
    print(f"    fitness, nominal: {results[('nominal', 'original')]['composite']:.4f} -> "
          f"{results[('nominal', 'edited')]['composite']:.4f}; no re-evolution.")
    print("=" * 78)
    if ok:
        print("SELF-CHECK PASS: SLA misses 2.4% -> 79.9% under the -10% stress, 2.2% after the edit; BE -6.3 Mbps.")
        return 0
    print("SELF-CHECK FAIL: recomputed values do not match the paper within tolerance.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
