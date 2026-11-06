#!/usr/bin/env python3
"""
Offline reproduction of the hardware panel (paper Sec. V-F, Table I) and the PPO comparator
summary (Sec. V-D).

What this does (standard library only):
  1. Recomputes Table I from the per-tick KPM logs of the POWDER session in logs_312891_rep1/:
     for each of the five controllers, the mean commanded guaranteed-slice share under the deep
     null (B_channel_null) and under the good channel (A_good_demand), and the mean best-effort
     throughput under the deep null. As in the paper, these are full-run arithmetic means
     including the startup samples (about 120 s, 110-115 samples per run).
  2. Prints the headline: best-effort throughput 158.2 Mbps (static 35/65) to 228.6 Mbps
     (evolved controller), +44.5%.
  3. Summarizes the recorded scores of the five PPO training seeds
     (ppo_baseline/ppo_experiment_records.json): 0.9534 +/- 0.0208, four of five above the
     static 35/65 score, and checks that ppo_baseline/ppo_policy.zip is the recorded deployed
     checkpoint (SHA-256 and seed metadata).

Self-check: exits non-zero unless every Table I cell is within 0.05 of the paper, the headline
within 0.1 point, and the PPO summary within 1e-3.

Run from inside this folder:
    python reproduce.py
"""
import hashlib
import json
import math
import os
import statistics
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "logs_312891_rep1")
PPO_DIR = os.path.join(HERE, "ppo_baseline")

# (log-file token, paper label, paper: null share %, good-channel share %, null BE Mbps)
ROWS = [
    ("evolved_champion_claude", "Claude evolved", 7.9, 35.4, 228.6),
    ("rl_policy_ppo", "RL policy (PPO)", 5.0, 36.6, 235.6),
    ("static_bar_35_65", "Static (35/65)", 35.0, 35.0, 158.2),
    ("comparison_champion_gemini", "Gemini evolved", 86.6, 40.3, 38.6),
    ("over_provisioning_90_10", "Static (90/10)", 90.0, 90.0, 30.1),
]
PAPER_GAIN = 44.5
PAPER_PPO = {"mean": 0.9534, "sd": 0.0208, "clears": 4}


def load_log(name):
    rows = [json.loads(line) for line in open(os.path.join(LOGDIR, name)) if line.strip()]
    for row in rows:
        for field in ("s1_commanded", "slice2_thp"):
            if not isinstance(row.get(field), (int, float)) or not math.isfinite(row[field]):
                raise ValueError(f"{name}: invalid {field}")
    return rows


def table_i():
    out = {}
    for token, label, *_ in ROWS:
        null = load_log(f"rep1_B_channel_null_{token}.jsonl")
        good = load_log(f"rep1_A_good_demand_{token}.jsonl")
        out[token] = {"null_share": statistics.fmean(r["s1_commanded"] for r in null),
                      "good_share": statistics.fmean(r["s1_commanded"] for r in good),
                      "null_be": statistics.fmean(r["slice2_thp"] for r in null),
                      "n_null": len(null), "n_good": len(good)}
    return out


def ppo_summary():
    rec = json.load(open(os.path.join(PPO_DIR, "ppo_experiment_records.json")))
    runs = rec["per_seed_scores"]
    scores = [r["worstenv"] for r in runs]
    deployed = next(r for r in runs if r["seed"] == rec["deployed_seed"])
    checkpoint = os.path.join(PPO_DIR, rec["deployed_checkpoint"])
    digest = hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()
    with zipfile.ZipFile(checkpoint) as z:          # plain JSON metadata; the policy is not unpickled
        meta = json.loads(z.read("data"))
    return {"seeds": [r["seed"] for r in runs], "scores": scores,
            "mean": statistics.fmean(scores), "sd": statistics.pstdev(scores),
            "clears": sum(s > rec["static_35_65_score"] for s in scores),
            "static": rec["static_35_65_score"],
            "deployed_seed": deployed["seed"], "deployed_score": deployed["worstenv"],
            "checkpoint_ok": digest == rec["deployed_checkpoint_sha256"] and meta.get("seed") == deployed["seed"]}


def main():
    ok = True
    print("=" * 78)
    print("TWO-SLICE HARDWARE PANEL  (paper Sec. V-F, Table I; POWDER session 312891)")
    print("=" * 78)
    print("Full-run arithmetic means including the startup samples, one run per controller and condition.")
    print()
    t = table_i()
    print(f"    {'controller':<18}{'null share %':>13}{'good share %':>13}{'null BE Mbps':>13}{'samples':>12}   paper")
    print("    " + "-" * 74)
    for token, label, p_null, p_good, p_be in ROWS:
        r = t[token]
        flag = "" if (abs(r["null_share"] - p_null) <= 0.05 and abs(r["good_share"] - p_good) <= 0.05
                      and abs(r["null_be"] - p_be) <= 0.05) else "   <-- MISMATCH"
        ok &= flag == ""
        print(f"    {label:<18}{r['null_share']:>13.1f}{r['good_share']:>13.1f}{r['null_be']:>13.1f}"
              f"{str(r['n_null']) + '/' + str(r['n_good']):>12}   {p_null} / {p_good} / {p_be}{flag}")
    gain = 100.0 * (t["evolved_champion_claude"]["null_be"] / t["static_bar_35_65"]["null_be"] - 1.0)
    flag = "" if abs(gain - PAPER_GAIN) <= 0.1 else "   <-- MISMATCH"
    ok &= flag == ""
    print()
    print(f"    best-effort throughput under the deep null: static 35/65 {t['static_bar_35_65']['null_be']:.1f} Mbps "
          f"-> evolved {t['evolved_champion_claude']['null_be']:.1f} Mbps, {gain:+.1f}%   [paper +{PAPER_GAIN}%]{flag}")

    print()
    p = ppo_summary()
    print("PPO comparator (paper Sec. V-D): five training seeds, 10^6 steps each")
    print(f"    seeds {p['seeds']}: scores {[round(s, 4) for s in p['scores']]}")
    flag = "" if (abs(p["mean"] - PAPER_PPO["mean"]) <= 1e-3 and abs(p["sd"] - PAPER_PPO["sd"]) <= 1e-3
                  and p["clears"] == PAPER_PPO["clears"]) else "   <-- MISMATCH"
    ok &= flag == ""
    print(f"    mean {p['mean']:.4f} +/- {p['sd']:.4f}; {p['clears']} of 5 above the static 35/65 score "
          f"{p['static']}   [paper 0.9534 +/- 0.0208, 4 of 5]{flag}")
    print(f"    deployed checkpoint ppo_baseline/ppo_policy.zip: seed {p['deployed_seed']} "
          f"(score {p['deployed_score']}); SHA-256 and seed metadata "
          f"{'match' if p['checkpoint_ok'] else 'DO NOT match'} the record")
    ok &= p["checkpoint_ok"]

    print()
    print("=" * 78)
    print("SELF-CHECK PASS: Table I, the +44.5% headline and the PPO summary reproduce." if ok
          else "SELF-CHECK FAIL: see the MISMATCH lines above.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
