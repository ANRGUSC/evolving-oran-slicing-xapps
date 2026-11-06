#!/usr/bin/env python3
"""
Offline reproduction of the four-slice results (paper Sec. VI, Tables II and III, Fig. 6).

What this does (NO network, NO API calls; standard library only):
  1. Rescores the five starting programs in controllers/ on the 16 development traces
     (seeds 0-15, 360 ticks, one-tick observation lag): fixed-threshold baseline -0.0485,
     expert rule 0.541, greedy seed 0.204, prompting seed 0.297, champion seed 0.678.
  2. Recomputes the Table II anchors with analysis/anchors.py: oracle reference 1.0555,
     weight-ordered greedy 0.9104 (true current channel) and -0.0974 (one-tick lag).
  3. Checks the sealed prompt: the SYSTEM and USER blocks of search/prompt.txt hash to the
     values journaled on every API call of the 30 primary-arm runs (data/runs/*.prompts.jsonl).
  4. Table III: for each of the 70 selected programs (data/held_out/selected_sources/)
     recomputes the selection fitness (minimum over the 16 development traces) and the
     held-out mean over the 200 sealed traces (seeds 20000-20199, data/held_out/
     held_out_seeds.json). The 30 primary-arm programs are checked against the complete
     run archives in data/runs/ (same source, same score as the archived best record).
     eta_held uses the recomputed held-out means of the fixed-threshold baseline (-0.0164)
     and of the reference policy funding each slice at ceil(T_k / e_k) (1.0523).
  5. Checks that every Fig. 6 curve (data/search_histories.json) ends at the fitness of the
     selected program of that run, and that the curves of the three primary arms equal the
     best-so-far records of the run archives.

Self-check: exits non-zero unless every recomputed value matches the paper (fitness within
1e-3, eta within 0.1 point). Step 4 evaluates 70 programs on 216 traces and takes a few minutes.

Run from inside this folder:
    python reproduce.py
"""
import glob
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "engine"))
sys.path.insert(0, os.path.join(HERE, "analysis"))
import simulator as E   # noqa: E402
import scorer as H      # noqa: E402
import anchors as A     # noqa: E402

DATA = os.path.join(HERE, "data")
CONTROLLERS = os.path.join(HERE, "controllers")
SELECTED = os.path.join(DATA, "held_out", "selected_sources")
TOL, ETA_TOL = 1e-3, 0.1

# (file, label, paper value)  -- Table II and the starting fitness in Table III
STARTING = [
    ("fixed_threshold_baseline.py", "fixed-threshold baseline (Table II)", -0.0485),
    ("expert_rule.py", "hand-written expert rule (Table II)", 0.541),
    ("greedy_seed.py", "greedy seed (Table III)", 0.204),
    ("prompting_seed.py", "prompting seed (Table III)", 0.297),
    ("champion_seed.py", "champion seed (Table III)", 0.678),
]
# (paper row label, run-name token, paper: selection mean, held-out mean, SD, eta_held %)
METHODS = [
    ("Prompting alone", "prompting_alone", 0.111, 0.158, 0.160, 16.3),
    ("Evolution from scratch", "evolution_from_scratch", 0.284, 0.327, 0.192, 32.1),
    ("Evolution: fixed-threshold seed (-0.048)", "evolution_from_baseline", 0.485, 0.529, 0.219, 51.0),
    ("Evolution: greedy seed (0.204)", "evolution_from_greedy_seed", 0.564, 0.608, 0.081, 58.5),
    ("Evolution: prompting seed (0.297)", "evolution_from_prompting_seed", 0.394, 0.436, 0.104, 42.3),
    ("Evolution: expert rule (0.541)", "evolution_from_expert_rule", 0.617, 0.677, 0.045, 64.9),
    ("Evolution: champion seed (0.678)", "evolution_from_champion_seed", 0.681, 0.712, 0.007, 68.2),
]
PAPER_EXPERT_HELDOUT, PAPER_EXPERT_ETA = 0.613, 58.9
PAPER_ANCHORS = {"oracle": 1.0555, "greedy_current": 0.9104, "greedy_lagged": -0.0974,
                 "heldout_baseline": -0.0164, "heldout_reference": 1.0523}
ARCHIVED_ARMS = ("prompting_alone", "evolution_from_scratch", "evolution_from_baseline")


def load_controller(path):
    ns = {"__name__": os.path.splitext(os.path.basename(path))[0], "__file__": path}
    exec(compile(open(path).read(), path, "exec"), ns)
    fn = ns.get("controller")
    if not callable(fn):
        raise ValueError(f"{path}: no controller function")
    return fn


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def check_prompt():
    """SYSTEM and USER blocks of search/prompt.txt vs the hashes journaled on every API call."""
    lines = open(os.path.join(HERE, "search", "prompt.txt")).read().split("\n")
    sys_block = "\n".join(lines[1:lines.index("USER") - 1])
    user_block = lines[lines.index("USER") + 1]
    sys_hashes, user_hashes, calls = set(), set(), 0
    for f in sorted(glob.glob(os.path.join(DATA, "runs", "*.prompts.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            calls += 1
            sys_hashes.add(r["sys_sha256"])
            if r["call_id"].split("::")[-1].startswith(("ind_", "init_")):   # task-prompt draws
                user_hashes.add(r["user_sha256"])
    ok = sys_hashes == {sha256(sys_block)} and user_hashes == {sha256(user_block)}
    return ok, calls, sha256(sys_block)[:8], sha256(user_block)[:8]


def archived_best(arm, run):
    """(source, score) of the best valid record in the complete archive of one primary-arm run."""
    meta = json.load(open(os.path.join(DATA, "runs", f"{arm}_r{run}.json")))["run"]
    for line in open(os.path.join(DATA, "runs", f"{arm}_r{run}.jsonl")):
        if line.strip():
            r = json.loads(line)
            if r.get("stage") == "done" and r["cid"] == meta["best_cid"]:
                return r["source"], r["score"]
    raise ValueError(f"{arm}_r{run}: best record {meta['best_cid']} not in the archive")


def main():
    t0 = time.time()
    ok = True
    print("=" * 78)
    print("FOUR-SLICE SIMULATION: OFFLINE REPRODUCTION  (paper Sec. VI, Tables II and III)")
    print("=" * 78)

    # ---- 1. starting programs on the development traces
    print("[1] Starting programs  (fitness = min over the 16 development traces of the mean composite)")
    print(f"    {'program':<40}{'fitness':>10}   paper")
    for fname, label, paper in STARTING:
        score = H.worst_env_prompt(load_controller(os.path.join(CONTROLLERS, fname)))
        flag = "" if abs(score - paper) <= TOL else "   <-- MISMATCH"
        ok &= flag == ""
        print(f"    {label:<40}{score:>10.4f}   {paper}{flag}")

    # ---- 2. anchors
    print()
    print("[2] Reference policies (analysis/anchors.py, development traces)")
    anchors = {"oracle": A.subset_reference(), "greedy_current": A.greedy_current(),
               "greedy_lagged": A.greedy_lagged()}
    for key, label in (("oracle", "oracle reference (true current channel)"),
                       ("greedy_current", "weight-ordered greedy, true current channel"),
                       ("greedy_lagged", "weight-ordered greedy, one-tick lag")):
        flag = "" if abs(anchors[key] - PAPER_ANCHORS[key]) <= TOL else "   <-- MISMATCH"
        ok &= flag == ""
        print(f"    {label:<48}{anchors[key]:>10.4f}   {PAPER_ANCHORS[key]}{flag}")

    # ---- 3. sealed prompt
    print()
    prompt_ok, calls, sys8, user8 = check_prompt()
    ok &= prompt_ok
    print(f"[3] Sealed prompt: SYSTEM {sys8}, USER {user8} -- "
          f"{'match' if prompt_ok else 'DO NOT match'} the hashes journaled on all {calls} API calls")

    # ---- 4. Table III
    print()
    seeds_file = json.load(open(os.path.join(DATA, "held_out", "held_out_seeds.json")))
    heldout = seeds_file["seeds"]
    assert seeds_file["n_seeds"] == len(heldout) == 200
    recorded = {r["run"]: r for r in json.load(open(os.path.join(DATA, "held_out", "held_out_scores.json")))["records"]}
    print(f"[4] Table III: 70 selected programs on the 16 development traces and the {len(heldout)} held-out traces")
    results, n = {}, 0
    for _, tok, *_ in METHODS:
        for run in range(1, 11):
            name = f"{tok}_r{run}"
            path = os.path.join(SELECTED, f"{name}.py")
            fn = load_controller(path)
            dev = [H.eval_prompt_controller(fn, s) for s in E.DEV]
            held = [H.eval_prompt_controller(fn, s) for s in heldout]
            results[name] = {"selection": min(dev), "heldout": statistics.fmean(held)}
            rec = recorded[name]
            if abs(results[name]["selection"] - rec["selection_min"]) > 5e-5 or \
               abs(results[name]["heldout"] - rec["heldout_mean_200"]) > 5e-5:
                ok = False
                print(f"    MISMATCH vs data/held_out/held_out_scores.json: {name}")
            if tok in ARCHIVED_ARMS:
                src, score = archived_best(tok, run)
                if src.strip() != open(path).read().strip() or abs(score - results[name]["selection"]) > 5e-5:
                    ok = False
                    print(f"    MISMATCH vs the run archive data/runs/{tok}_r{run}.jsonl: {name}")
            n += 1
            if n % 10 == 0:
                print(f"    {n}/70 programs evaluated ({time.time() - t0:.0f} s)", flush=True)
    base_fn = load_controller(os.path.join(CONTROLLERS, "fixed_threshold_baseline.py"))
    expert_fn = load_controller(os.path.join(CONTROLLERS, "expert_rule.py"))
    base_held = statistics.fmean(H.eval_prompt_controller(base_fn, s) for s in heldout)
    expert_held = statistics.fmean(H.eval_prompt_controller(expert_fn, s) for s in heldout)
    ref_held = statistics.fmean(A._subset_episode(s, funding_tolerance=0.0) for s in heldout)

    def eta(x):
        return 100.0 * (x - base_held) / (ref_held - base_held)

    print()
    print("    eta_held anchors (held-out means): fixed-threshold baseline "
          f"{base_held:.4f} [paper {PAPER_ANCHORS['heldout_baseline']}], reference policy "
          f"{ref_held:.4f} [paper {PAPER_ANCHORS['heldout_reference']}]")
    ok &= abs(base_held - PAPER_ANCHORS["heldout_baseline"]) <= TOL
    ok &= abs(ref_held - PAPER_ANCHORS["heldout_reference"]) <= TOL
    print()
    print(f"    {'method (starting fitness)':<42}{'select.':>8}{'held-out':>9}{'SD':>7}{'eta%':>7}   paper")
    print("    " + "-" * 74)
    for label, tok, p_sel, p_held, p_sd, p_eta in METHODS:
        sel = statistics.fmean(results[f"{tok}_r{r}"]["selection"] for r in range(1, 11))
        hv = [results[f"{tok}_r{r}"]["heldout"] for r in range(1, 11)]
        hm, sd = statistics.fmean(hv), statistics.stdev(hv)
        flag = "" if (abs(sel - p_sel) <= TOL and abs(hm - p_held) <= TOL and abs(sd - p_sd) <= TOL
                      and abs(eta(hm) - p_eta) <= ETA_TOL) else "   <-- MISMATCH"
        ok &= flag == ""
        print(f"    {label:<42}{sel:>8.3f}{hm:>9.3f}{sd:>7.3f}{eta(hm):>7.1f}   "
              f"{p_sel:.3f} {p_held:.3f} {p_sd:.3f} {p_eta}{flag}")
    flag = "" if (abs(expert_held - PAPER_EXPERT_HELDOUT) <= TOL
                  and abs(eta(expert_held) - PAPER_EXPERT_ETA) <= ETA_TOL) else "   <-- MISMATCH"
    ok &= flag == ""
    print(f"    {'Hand-written expert rule':<42}{0.541:>8.3f}{expert_held:>9.3f}{'--':>7}{eta(expert_held):>7.1f}   "
          f"0.541 {PAPER_EXPERT_HELDOUT} --    {PAPER_EXPERT_ETA}{flag}")

    # ---- 5. Fig. 6 curves
    print()
    hist = json.load(open(os.path.join(DATA, "search_histories.json")))
    points, curves_ok = 0, True
    for arm, block in hist.items():
        start = block["start_budget"]
        for run in range(1, 11):
            curve = block["runs"][str(run)]
            points += len(curve)
            valid = [v for v in curve if v is not None]
            if abs(valid[-1] - results[f"{arm}_r{run}"]["selection"]) > 5e-5:
                curves_ok = False
            if arm in ARCHIVED_ARMS:
                meta = json.load(open(os.path.join(DATA, "runs", f"{arm}_r{run}.json")))
                archived = {c["budget"]: c["best_valid_score"] for c in meta["budget_curve"]}
                for i, v in enumerate(curve):
                    a = archived.get(start + i)
                    if (a is None) != (v is None) or (a is not None and abs(a - v) > 1e-9):
                        curves_ok = False
    ok &= curves_ok
    print(f"[5] Fig. 6 curves: {points} points, {len(hist)} methods x 10 runs; "
          f"{'end at the selected programs and match the run archives' if curves_ok else 'MISMATCH'}")

    print()
    print("=" * 78)
    if ok:
        print(f"SELF-CHECK PASS: Tables II and III and Fig. 6 reproduce ({time.time() - t0:.0f} s).")
    else:
        print("SELF-CHECK FAIL: see the MISMATCH lines above.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
