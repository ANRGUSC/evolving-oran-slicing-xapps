"""Deterministic 75-step CQI fixture (paper Sec. V-E, "Selected Program and Source-Based Predictions").

The paper derives three predictions from the evolved controller's source and checks them on a
fixed CQI sequence: 30 observations at CQI 15, 30 at CQI 5, 15 at CQI 15.

  1. Release timing  -- from the CQI-15 steady state the EMA follows c = 5 + 10*(0.7)^k after k low
                        observations, so the release predicate (EMA < 6.5 and >=2 of the last 3 < 6)
                        first holds at k = 6: the controller clamps to the SLA minimum (5) on the SIXTH
                        low observation.
  2. Restoration     -- no latched release state and no exit threshold: a share above the floor is
                        commanded on the FIRST recovered observation.
  3. Comparison bug  -- the Gemini comparison controller never releases, even at a CQI floor of 2.

This file replays the fixture through the same harness objects the benchmark uses (simulator.Tick /
Hist / Sla / State) with deterministic delivery (Eq. 8 physics, no noise, demand 96 Mbps) so that
the only driver is the CQI sequence. Standard library only.

Usage:  python cqi_fixture.py            (from engine/ or anywhere; paths are resolved here)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulator as B  # noqa: E402

CONTROLLERS = os.path.join(os.path.dirname(HERE), "controllers")
SEQ_GOOD, SEQ_NULL, SEQ_RECOVER = 30, 30, 15
DEMAND = 96.0


def load(name):
    ns = {}
    exec(compile(open(os.path.join(CONTROLLERS, name)).read(), name, "exec"), ns)
    return ns["controller"]


def run_fixture(ctrl, low_cqi):
    """Return the list of commanded SLICE1 shares, one per observation."""
    seq = [B.GOOD_CQI] * SEQ_GOOD + [low_cqi] * SEQ_NULL + [B.GOOD_CQI] * SEQ_RECOVER
    sla, st, h = B.Sla(B.TARGET), B.State(), B.Hist()
    prev_share, prev_s1, prev_s2 = 35, 0.0, 0.0
    shares = []
    for cqi in seq:
        h.ue1.push(B.Tick(cqi, prev_share, prev_s1, prev_s1 * 125000.0))
        h.ue2.push(B.Tick(B.GOOD_CQI, B.POOL - prev_share, prev_s2, prev_s2 * 125000.0))
        s1 = max(5, min(90, B._share_of(ctrl(h, st, sla, {}))))
        null = cqi < 6
        d1 = B.s1_null(s1) if null else B.s1_good(s1, DEMAND)
        d2, _ = B.s2_thp(B.POOL - s1, "linear")
        shares.append(s1)
        prev_share, prev_s1, prev_s2 = s1, d1, d2
    return shares


def analyse(shares):
    null = shares[SEQ_GOOD:SEQ_GOOD + SEQ_NULL]
    rec = shares[SEQ_GOOD + SEQ_NULL:]
    k_release = next((k + 1 for k, s in enumerate(null) if s == 5), None)   # 1-based low-observation index
    restore_first = rec[0] > 5 if rec else None
    return k_release, restore_first, null, rec


def main():
    ok = True
    print("75-step CQI fixture: 30 x CQI15, 30 x CQI5, 15 x CQI15")

    champ = load("evolved_claude.py")
    sh = run_fixture(champ, 5)
    k, restore, null, rec = analyse(sh)
    print(f"  Claude: release on low observation {k}; first recovered share {rec[0]}")
    ok &= (k == 6) and bool(restore)

    gem = load("comparison_gemini.py")
    for floor in (5, 2):
        shg = run_fixture(gem, floor)
        kg, _, nullg, _ = analyse(shg)
        print(f"  Gemini, CQI floor {floor}: release {'never' if kg is None else f'on observation {kg}'}")
        ok &= (kg is None)

    print("Fixture:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
