#!/usr/bin/env python3
"""
Offline reproduction of the two-slice results (paper Sec. V-C and V-E).

What this does (NO network, NO API calls; standard library only):
  1. Scores the evolved controller (controllers/evolved_claude.py) and the Gemini comparison
     controller (controllers/comparison_gemini.py) with the scorer used during evolution
     (engine/scorer.py: 3 CQI floors x 2 best-effort profiles x 16 seeds, worst floor):
     0.9398 and 0.8313, against the static 35/65 allocation at 0.9300.
  2. Reads the score records of the 279 zero-shot calls (data/zero_shot_scores.jsonl):
     144 valid programs, 72 per model; the best scores 0.9054 and none reaches 0.9300;
     Gemini voided 65% of its calls, Claude 5%.
  3. Reads the per-generation records of the two evolution runs (data/search_history/):
     the Claude run rises from a generation-0 best of 0.8729 to 0.9398 (+0.034 over the best
     zero-shot program); the Gemini run ends at 0.8313, 0.073 below its best zero-shot program.
  4. Reads the 20 initial programs drawn under the confident prompt
     (data/cold_draws_confident.jsonl): 13 of 20 propose a release rule.
  5. Replays the 75-step CQI fixture of Sec. V-E (engine/cqi_fixture.py): the evolved
     controller releases on the 6th low observation and restores on the first recovered one;
     the comparison controller never releases, even at a CQI floor of 2.

Self-check: exits non-zero unless every value above matches the paper.

Run from inside this folder:
    python reproduce.py
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine"))
import scorer       # noqa: E402
import cqi_fixture  # noqa: E402

DATA = os.path.join(HERE, "data")
PAPER = {"claude": 0.9398, "gemini": 0.8313, "static": 0.9300, "zero_shot_best": 0.9054,
         "zero_shot_n": 144, "void_claude": 5, "void_gemini": 65, "claude_gain": 0.034,
         "gemini_gap": -0.073, "confident": (13, 20), "gen0_claude": 0.8729}


def jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    ok = True
    print("=" * 78)
    print("TWO-SLICE BENCHMARK: OFFLINE REPRODUCTION  (paper Sec. V-C and V-E)")
    print("=" * 78)

    # ---- 1. the two controllers and the static bar
    scores = {}
    for key, fname, label in (("claude", "evolved_claude.py", "evolved controller (Claude)"),
                              ("gemini", "comparison_gemini.py", "comparison controller (Gemini)")):
        r = scorer.score_source_v3(open(os.path.join(HERE, "controllers", fname)).read())
        if r["status"] != "ok":
            print(f"FAIL: {fname} did not score ({r.get('reason')})")
            return 1
        scores[key] = r["composite"]
        flag = "" if abs(r["composite"] - PAPER[key]) <= 1e-3 else "   <-- MISMATCH"
        ok &= flag == ""
        print(f"    {label:<34}{r['composite']:>8.4f}   null behavior {scorer.classify_null(r['trace']):<9}"
              f"[paper {PAPER[key]}]{flag}")
    static = scorer.score_static_share(35)
    flag = "" if abs(static - PAPER["static"]) <= 1e-3 else "   <-- MISMATCH"
    ok &= flag == ""
    print(f"    {'static 35/65 allocation':<34}{static:>8.4f}   [paper {PAPER['static']}]{flag}")

    # ---- 2. zero-shot pool
    records = jsonl(os.path.join(DATA, "zero_shot_scores.jsonl"))
    valid = [r for r in records if r["status"] == "ok"]
    best_zs = max(float(r["composite"]) for r in valid)
    above = sum(float(r["composite"]) >= static for r in valid)
    print()
    print(f"Zero-shot pool: {len(valid)} valid programs from {len(records)} calls   [paper 144]")
    voids = {}
    for model in sorted({r["model_alias"] for r in records}):
        calls = [r for r in records if r["model_alias"] == model]
        good = [float(r["composite"]) for r in calls if r["status"] == "ok"]
        voids[model] = 100.0 * (len(calls) - len(good)) / len(calls)
        print(f"    {model:<24} {len(good):>3} valid of {len(calls):>3} calls; voided {voids[model]:4.1f}%; best {max(good):.4f}")
    print(f"    best zero-shot {best_zs:.4f} [paper 0.9054]; programs at or above the static bar: {above} [paper none]")
    print(f"    void rates: Gemini {voids['gemini-2.5-flash-lite']:.0f}% [paper 65%], Claude {voids['claude-haiku-4-5']:.0f}% [paper 5%]")
    ok &= len(valid) == PAPER["zero_shot_n"] and above == 0 and abs(best_zs - PAPER["zero_shot_best"]) <= 1e-3
    ok &= round(voids["gemini-2.5-flash-lite"]) == PAPER["void_gemini"] and round(voids["claude-haiku-4-5"]) == PAPER["void_claude"]

    # ---- 3. evolution runs
    print()
    print("Evolution runs (data/search_history/, 3 seeds x 5 generations each)")
    gen = {}
    for model in ("claude", "gemini"):
        rows = jsonl(os.path.join(DATA, "search_history", f"{model}_generations.jsonl"))
        gen0 = {r["best"] for r in rows if r["gen"] == 0}
        if len(rows) != 15 or len(gen0) != 1:
            print(f"FAIL: unexpected generation records for {model}")
            return 1
        gen[model] = {"gen0": gen0.pop(), "peak": max(r["best"] for r in rows), "calls": max(r["calls"] for r in rows)}
        print(f"    {model:<8} generation-0 best {gen[model]['gen0']:.4f} -> peak {gen[model]['peak']:.4f}  ({gen[model]['calls']} calls)")
    claude_gain = scores["claude"] - best_zs
    gemini_best_zs = max(float(r["composite"]) for r in valid if r["model_alias"] == "gemini-2.5-flash-lite")
    gemini_gap = scores["gemini"] - gemini_best_zs
    print(f"    Claude: evolved - best zero-shot = {claude_gain:+.3f}   [paper +0.034]")
    print(f"    Gemini: evolved - best zero-shot = {gemini_gap:+.3f}   [paper -0.073]")
    ok &= abs(claude_gain - PAPER["claude_gain"]) <= 1e-3 and abs(gemini_gap - PAPER["gemini_gap"]) <= 1e-3
    ok &= abs(gen["claude"]["gen0"] - PAPER["gen0_claude"]) <= 1e-3 and abs(gen["claude"]["peak"] - scores["claude"]) <= 1e-3

    # ---- 4. confident prompt
    draws = jsonl(os.path.join(DATA, "cold_draws_confident.jsonl"))
    for d in draws:
        if hashlib.sha256(d["source"].encode()).hexdigest() != d["source_sha256"]:
            print(f"FAIL: source digest mismatch in confident draw {d['i']}")
            return 1
    release = sum(d["abandons_u"] and d["abandons_c"] for d in draws)
    print()
    print(f"Confident prompt: {release} of {len(draws)} initial programs propose a release rule   [paper 13 of 20]")
    ok &= (release, len(draws)) == PAPER["confident"]

    # ---- 5. CQI fixture
    print()
    fixture_rc = cqi_fixture.main()
    ok &= fixture_rc == 0

    print()
    print("=" * 78)
    print("SELF-CHECK PASS: scores, zero-shot pool, search statistics and fixture as in the paper." if ok
          else "SELF-CHECK FAIL: see the MISMATCH lines above.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
