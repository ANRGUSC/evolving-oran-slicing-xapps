#!/usr/bin/env python3
"""
Offline reproduction of the one-constant source repair (paper Sec. VI-E, Fig. 7).

Runs repair.py: it applies the single edit 11.8 -> 2.19 to inputs/controller.py (the controller
returned by the prompting-seeded arm, run 3), replays both versions on the 16 development traces
with the four-slice simulator and scorer (../four_slice/engine, unchanged) and writes results/
(summary.json, comparison.json, per-seed and paired CSVs, one_parameter.diff). With --figure it
also draws Fig. 7 (needs matplotlib).

Self-check: exits non-zero unless the worst-episode fitness is within 1e-3 of 0.3226 (original)
and 0.7255 (edited) and all 16 traces improve.

Run from inside this folder:
    python reproduce.py [--figure]
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', action='store_true', help='Draw Fig. 7 with matplotlib.')
    args = parser.parse_args()
    out = ROOT / 'results'
    command = [sys.executable, str(ROOT / 'repair.py'), '--out', str(out)]
    if not args.figure:
        command.append('--skip-figure')
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        return result.returncode
    summary = {row['variant']: row for row in json.loads((out / 'summary.json').read_text())}
    comparison = json.loads((out / 'comparison.json').read_text())
    original, edited = summary['original'], summary['edited']
    print()
    print("=" * 78)
    print("ONE-CONSTANT SOURCE REPAIR  (paper Sec. VI-E, Fig. 7): 11.8 -> 2.19 in the CQI rate fallback")
    print("=" * 78)
    print(f"    {'metric':<34}{'original':>12}{'edited':>12}     paper")
    print("    " + "-" * 74)
    for label, key, digits, paper in [
        ('worst-episode fitness', 'fitness', 4, '0.3226 -> 0.7255'),
        ('target attainment (%)', 'attainment_pct', 1, '37.7 -> 45.2'),
        ('allocation changes / tick', 'churn_count', 2, '3.76 -> 2.26'),
        ('restoration within 7 ticks (%)', 'restore_pct', 1, '41.6 -> 50.5'),
        ('best-effort throughput (Mbps)', 'be_mbps', 1, '75.1 -> 72.1'),
    ]:
        print(f"    {label:<34}{original[key]:>12.{digits}f}{edited[key]:>12.{digits}f}     {paper}")
    print(f"    {'traces improved / worsened':<34}{comparison['improved_traces']:>12}{comparison['worsened_traces']:>12}     16 / 0")
    ok = (abs(original['fitness'] - 0.3226) <= 1e-3 and abs(edited['fitness'] - 0.7255) <= 1e-3
          and comparison['improved_traces'] == 16 and comparison['worsened_traces'] == 0)
    print("=" * 78)
    print("SELF-CHECK PASS: fitness 0.3226 -> 0.7255, 16/16 traces improved." if ok
          else "SELF-CHECK FAIL: recomputed values do not match the paper within tolerance.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
