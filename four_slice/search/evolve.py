#!/usr/bin/env python3
"""Corrected, OFFLINE search harness (Phase 1 fix + Phase 2 arms). No API client at import. Generated candidate
code is evaluated in a credential-free subprocess worker with a wall-clock timeout (parent SIGKILLs on
overrun). (source, score, validity, provenance) is ONE atomic candidate record; a child is eligible only after
a successful finite evaluation; on any child failure the parent record is retained and a fallback event logged.

Model interface is pluggable: MockModel (deterministic, offline — used for validation NOW) or a real API model
(NOT imported/instantiated here; wired only when a spend limit + model id are frozen). Arms: independent /
unseeded / seeded. Ledger separates logical draws, API attempts, successful responses, and evaluation ticks.

Usage (offline mock):
  .venv/bin/python research_upgrade/scripts/evolve_offline.py --arm unseeded --runs 1 --mock --nseeds 4
"""
import argparse, hashlib, json, math, os, random, subprocess, sys, tempfile, time
from dataclasses import dataclass, field, asdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKER = os.path.join(ROOT, "research_upgrade/scripts/_eval_worker.py")
PY = sys.executable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_model import CapReached as _CapReached   # no anthropic import at module load (lazy inside real_model)

# --------------------------------------------------------------------------- atomic candidate record
@dataclass
class Candidate:
    cid: str
    source: str
    score: float | None
    valid: bool
    reason: str
    ticks: int
    provenance: dict = field(default_factory=dict)

# --------------------------------------------------------------------------- isolated evaluator
def evaluate(source: str, nseeds: int, timeout: float = 60.0) -> tuple:
    """Return (valid, score, ticks, reason). Subprocess, credentials scrubbed, wall-clock bounded."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(source); path = fh.name
    env = {k: v for k, v in os.environ.items()
           if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "ANTHROPIC", "OPENAI", "AWS", "GOOGLE"))}
    try:
        p = subprocess.run([PY, WORKER, path, str(nseeds)], capture_output=True, text=True,
                           timeout=timeout, env=env)
        line = (p.stdout or "").strip().splitlines()[-1] if p.stdout.strip() else ""
        d = json.loads(line) if line else {"valid": False, "score": None, "ticks": 0, "reason": "no_output"}
        return d["valid"], d["score"], d["ticks"], d["reason"]
    except subprocess.TimeoutExpired:
        return False, None, 0, "timeout"
    except Exception as e:
        return False, None, 0, f"worker:{type(e).__name__}"
    finally:
        try: os.unlink(path)
        except OSError: pass

# --------------------------------------------------------------------------- model interface
class MockModel:
    """Deterministic offline model. Emits Env-D controllers from a fixed library, plus (deterministically) some
    invalid/failing proposals so the failure paths are exercised. No network, no API. Seeded by orchestration
    RNG only — NOTE a real API is NOT deterministic even under a seeded orchestrator (documented in the plan)."""
    def __init__(self, rng): self.rng = rng
    def draw(self, prompt_kind, parent_src=None, feedback=True, call_key=None, on_response=None):
        src = self._draw(prompt_kind, parent_src, feedback)
        if on_response is not None: on_response(src, 0, 0)   # exercise the durable-source hook (offline, $0)
        return src
    def _draw(self, prompt_kind, parent_src=None, feedback=True):
        roll = self.rng.random()
        # ~15% invalid to exercise the corrected rule; deterministic given the seed
        if roll < 0.08:  return "def controller(h,s,sla,p):\n    retrn [1,2,3,4]\n"          # syntax error
        if roll < 0.15:  return "def controller(h,s,sla,p):\n    raise ValueError('boom')\n"  # runtime raise
        if roll < 0.18:  return "x = 1\n"                                                     # no controller
        # valid library: target-cost-ish allocators with varied constants
        a = self.rng.choice([14, 16, 17, 18, 20]); b = self.rng.choice([14, 16, 17, 18])
        c = self.rng.choice([12, 15, 17]); d = self.rng.choice([10, 14, 16])
        floor = self.rng.choice([0, 5])
        return (f"def controller(history, state, sla, predictors):\n"
                f"    cap = sla.guaranteed_cap\n"
                f"    base = [{a},{b},{c},{d}]\n"
                f"    cqis = []\n"
                f"    for k in range(4):\n"
                f"        smp = history.slices[k].last()\n"
                f"        cqis.append(smp.cqi if smp and smp.cqi is not None else 10.0)\n"
                f"    out = [base[k] if cqis[k] > 6.0 else {floor} for k in range(4)]\n"
                f"    tot = sum(out)\n"
                f"    if tot > cap:\n"
                f"        out = [int(x*cap/tot) for x in out]\n"
                f"    return out\n")

# --------------------------------------------------------------------------- ledger
def new_ledger():
    return {"logical_draws": 0, "api_attempts": 0, "successful_responses": 0,
            "eval_ticks_allocated": 0, "actual_executed_ticks": 0, "unique_evals": 0, "fallback_events": 0,
            "invalid_children": 0, "full_charge_evals": 0, "zero_charge_evals": 0, "reasons": {}}

# --------------------------------------------------------------------------- durable journal + resume
class Journal:
    """Append-only JSONL of per-candidate records. Stage 'drawn' (raw source+usage, written the instant a
    response is received, BEFORE evaluation) then 'done' (adds validity/score/ticks). Enables loss-free resume:
    a killed process leaves every received source on disk; re-invoking the SAME command replays the (seeded,
    deterministic) orchestration and reuses journaled candidates as cache hits -> no re-draw, no re-charge."""
    def __init__(self, path):
        self.path = path; self.done = {}; self.drawn = {}
        if path and os.path.exists(path):
            for ln in open(path):
                ln = ln.strip()
                if not ln: continue
                try: r = json.loads(ln)
                except Exception: continue
                if r.get("stage") == "done": self.done[r["cid"]] = r
                elif r.get("stage") == "drawn": self.drawn.setdefault(r["cid"], r)
    def _append(self, rec):
        if not self.path: return
        with open(self.path, "a") as f: f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
    def mark_drawn(self, cid, source, in_tok, out_tok):
        rec = {"cid": cid, "stage": "drawn", "source": source, "in_tok": in_tok, "out_tok": out_tok}
        self.drawn[cid] = rec; self._append(rec)
    def mark_done(self, cand):
        rec = asdict(cand); rec["stage"] = "done"; self.done[cand.cid] = rec; self._append(rec)


def _cand_from_rec(d):
    return Candidate(cid=d["cid"], source=d["source"], score=d.get("score"), valid=d["valid"],
                     reason=d["reason"], ticks=d.get("ticks", 0), provenance=d.get("provenance", {}))


def _resolve(cid, kind, parent, feedback, model, nseeds, ledger, evaluator, journal, prov):
    """Return (Candidate, origin). Resume-aware: cache-hit 'done' reuses (no draw, no eval, no charge);
    'drawn'-only re-evaluates the SAVED source (no API call); else a fresh draw (durable source hook) + eval."""
    if journal and cid in journal.done:
        return _cand_from_rec(journal.done[cid]), "cache_done"
    if journal and cid in journal.drawn:
        cand = make_candidate(cid, journal.drawn[cid]["source"], nseeds, ledger, prov, evaluator)
        if journal: journal.mark_done(cand)
        return cand, "cache_drawn_reeval"
    on_resp = (lambda s, i, o: journal.mark_drawn(cid, s, i, o)) if journal else None
    src = model.draw(kind, parent, feedback=feedback, call_key=cid, on_response=on_resp)
    cand = make_candidate(cid, src, nseeds, ledger, prov, evaluator)
    if journal: journal.mark_done(cand)
    return cand, "fresh"


# --------------------------------------------------------------------------- arms
def make_candidate(cid, source, nseeds, ledger, prov, evaluator=None):
    valid, score, ticks, reason = (evaluator or evaluate)(source, nseeds)
    ledger["eval_ticks_allocated"] += ticks; ledger["unique_evals"] += 1
    # ACTUAL executed ticks: a valid eval runs all nseeds x 360; a runtime-raise/nonfinite/timeout faults early
    # so its actual executed ticks are ~unknown-but-small -> counted 0 here and flagged as allocation-only.
    ledger["actual_executed_ticks"] += (ticks if valid else 0)
    ledger["full_charge_evals"] += (1 if ticks > 0 else 0)
    ledger["zero_charge_evals"] += (1 if ticks == 0 else 0)
    ledger["reasons"][reason] = ledger["reasons"].get(reason, 0) + 1
    if not valid: ledger["invalid_children"] += 1
    prov = dict(prov); prov["src_sha8"] = hashlib.sha256(source.encode()).hexdigest()[:8]
    return Candidate(cid, source, score, valid, reason, ticks, prov)

def run_independent(model, nseeds, budget, ledger, log, evaluator=None, journal=None):
    """64 independent task-prompt draws; no parent/feedback; best valid by DEV score. Resume-safe: cids ind_c{i}
    are deterministic in the loop index, so re-invoking reuses journaled candidates and continues the remainder."""
    archive = []
    for i in range(budget):
        cid = f"ind_c{i}"
        ledger["logical_draws"] += 1
        try:
            cand, origin = _resolve(cid, "task", None, True, model, nseeds, ledger, evaluator, journal,
                                    {"arm": "independent", "draw": i})
        except _CapReached:
            return archive, {"status": "STOPPED_AT_CAP", "at": "independent", "draw": i}
        if origin == "fresh": ledger["api_attempts"] += 1; ledger["successful_responses"] += 1
        archive.append(cand)
        log.append({"budget": i + 1, "best_valid_score": _best(archive)})
    return archive, {"status": "ok"}

def run_evolution(model, nseeds, budget, ledger, log, seeded=False, seed_src=None,
                  pop_size=8, elite=4, rng=None, feedback=True, selection="fitness", evaluator=None, journal=None):
    """Unseeded/seeded evolution: 16 initial + (GENS x (elite carried + child proposals)); atomic records;
    complete archive kept; best DEV candidate chosen from the archive at the end (for ALL arms).
    Controls (item 3): feedback=False omits numeric scores/diagnostics/traces from the mutation prompt
    (selection is STILL fitness-based -> "no EXPLICIT feedback"); selection="random" chooses the population
    and surviving parents UNIFORMLY (not fitness-ranked), but the best-ever is still taken from the complete
    archive so search effectiveness is not confounded with accidental loss of a good final candidate."""
    rng = rng or random.Random(0)
    archive = []; used = 0
    # initial draws (16); seeded arm additionally makes the factored seed eligible
    n_init = 16
    init = []
    if seeded and seed_src is not None:
        if journal and "seed_factored" in journal.done:
            c = _cand_from_rec(journal.done["seed_factored"])
        else:
            c = make_candidate("seed_factored", seed_src, nseeds, ledger, {"arm": "seeded", "role": "seed"}, evaluator)
            if journal: journal.mark_done(c)
        init.append(c); archive.append(c)   # BUGFIX: the seed must be in the COMPLETE archive used for the
        #   final best-ever selection (else a seeded run could report worse-than-seed, which is impossible if
        #   the seed is eligible under the same evaluator). Seed eval counts as evaluator work, not a model draw.
        log.append({"budget": used, "best_valid_score": _best(archive)})
    for i in range(n_init):
        if used >= budget: break
        cid = f"init_c{i}"; ledger["logical_draws"] += 1; used += 1
        try:
            c, origin = _resolve(cid, "task", None, True, model, nseeds, ledger, evaluator, journal,
                                 {"arm": "seeded" if seeded else "unseeded", "gen": 0})
        except _CapReached:
            return archive, {"status": "STOPPED_AT_CAP", "at": "init", "i": i}
        if origin == "fresh": ledger["api_attempts"] += 1; ledger["successful_responses"] += 1
        init.append(c); archive.append(c)
        log.append({"budget": used, "best_valid_score": _best(archive)})
    archive_init_valid = [c for c in init if c.valid]
    if not archive_init_valid:
        return archive, {"status": "FAILED_RUN_no_valid_initial"}   # do NOT inject a hand-written fallback
    # population = top `pop_size` valid by score (seed eligible in seeded arm); random-selection control picks
    # the population UNIFORMLY among valid candidates (best-ever still taken from the complete archive).
    valid0 = [c for c in init if c.valid]
    if selection == "random":
        pop = rng.sample(valid0, min(pop_size, len(valid0)))
    else:
        pop = sorted(valid0, key=lambda c: -c.score)[:pop_size]
    gen = 1
    while used < budget:
        if selection == "random":                           # surviving parents chosen uniformly (no ranking)
            parents = rng.sample(pop, min(elite, len(pop)))
        else:
            parents = sorted(pop, key=lambda c: -c.score)[:elite]
        newpop = list(parents)                              # exact elitism: carry elite forward
        while len(newpop) < pop_size and used < budget:
            par = parents[rng.randrange(len(parents))]
            cid = f"g{gen}c{len(newpop)}"; ledger["logical_draws"] += 1; used += 1
            try:
                child, origin = _resolve(cid, "mutation", par.source, feedback, model, nseeds, ledger, evaluator,
                                         journal, {"gen": gen, "parent": par.cid})
            except _CapReached:
                return archive, {"status": "STOPPED_AT_CAP", "at": "mutation", "gen": gen}
            if origin == "fresh": ledger["api_attempts"] += 1; ledger["successful_responses"] += 1
            archive.append(child)
            if child.valid:                                  # ATOMIC: eligible only after successful eval
                newpop.append(child)
            else:                                            # retain parent; log fallback; never borrow fitness
                ledger["fallback_events"] += 1
                newpop.append(par)                           # a copy of the (valid) parent record
                log_fallback = {"gen": gen, "reason": child.reason, "parent": par.cid, "child_cid": child.cid}
            log.append({"budget": used, "best_valid_score": _best(archive)})
        pop = newpop; gen += 1
    return archive, {"status": "ok"}

def _best(cands):
    v = [c.score for c in cands if c.valid and c.score is not None]
    return max(v) if v else None

# --------------------------------------------------------------------------- paid wiring (gated)
def pilot_commands():
    """Exact, supported terminal commands for the authorized 192-proposal pilot (3 arms share ONE durable
    ledger -> the $30 cap spans all 192 proposals across the three invocations)."""
    base = (".venv/bin/python research_upgrade/scripts/evolve_offline.py --paid --runs 1 --budget 64 "
            "--nseeds 16 --ledger research_upgrade/results/pilot/cost_ledger.json")
    return "\n".join(f"{base} --arm {arm} --out research_upgrade/results/pilot/pilot_{arm}.json"
                     for arm in ("independent", "unseeded", "seeded"))


def _build_paid(a, client_override=None):
    """Construct (RealModel, CostLedger, rpc_evaluator, model_id). GATED: requires paid_execution_authorized +
    positive spend_cap_usd in pilot_config.json AND a present API key (unless a test client is injected).
    Evaluation is routed through the trusted-RPC evaluator (sandboxed worker). No historical runner is used."""
    import real_model as RM, cost_ledger as CL, trusted_evaluator as TE
    cfg = json.load(open(os.path.join(ROOT, "research_upgrade/configs/pilot_config.json")))
    assert cfg.get("paid_execution_authorized") is True, "paid_execution_authorized is not true in pilot_config.json"
    cap = cfg.get("spend_cap_usd")
    assert isinstance(cap, (int, float)) and cap > 0, "spend_cap_usd must be a positive number in pilot_config.json"
    pricing = cfg["pricing_usd_per_million_tokens"]; max_out = cfg["sampling"]["max_tokens"]
    model_id = a.model or cfg["generation_model"]
    ledger_path = os.path.join(ROOT, a.ledger); os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    ledger = CL.CostLedger(ledger_path, cap, pricing["input"], pricing["output"], max_out)
    if client_override is not None:
        client = client_override
    else:
        key = RM.load_api_key()
        if not key:
            print("EXECUTION BLOCKED: no ANTHROPIC_API_KEY in the environment or research_upgrade/.env.\n"
                  "  Provide it privately (NEVER commit; both .env paths are gitignored), e.g.:\n"
                  "    export ANTHROPIC_API_KEY=sk-ant-...      # current shell only\n"
                  "  or add a line  ANTHROPIC_API_KEY=sk-ant-...  to research_upgrade/.env\n"
                  "  then run the exact pilot commands:\n\n" + pilot_commands() +
                  "\n\nNo paid call was made.")
            raise SystemExit(3)
        client = RM.make_client(key)
    model = RM.RealModel(client, ledger, model=model_id, max_tokens=max_out,
                         temperature=cfg["sampling"]["temperature"], max_attempts=a.max_attempts,
                         call_prefix=f"{a.arm}::")   # arm-namespaced ledger call-ids (cids collide across arms)

    def rpc_evaluator(source, nseeds):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(source); path = fh.name
        try:
            rr = TE.evaluate_rpc(path, list(range(nseeds)), sandboxed=True)
            valid = rr["status"] == "ok" and rr["worst_env"] is not None
            return valid, rr["worst_env"], rr["executed_ticks"], ("ok" if valid else rr["reason"])
        finally:
            try: os.unlink(path)
            except OSError: pass
    return model, ledger, rpc_evaluator, model_id


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["independent", "unseeded", "seeded", "no_feedback", "random_selection"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--nseeds", type=int, default=4, help="DEV seeds for evaluation (mock uses fewer for speed)")
    ap.add_argument("--mock", action="store_true", help="use offline MockModel (no API)")
    ap.add_argument("--paid", action="store_true", help="use the REAL model (gated: authorized+cap+key required)")
    ap.add_argument("--model", default=None, help="override model id (default from pilot_config.json)")
    ap.add_argument("--max-attempts", type=int, default=3, help="SDK attempts per logical draw (each reserved)")
    ap.add_argument("--ledger", default="research_upgrade/results/pilot/cost_ledger.json", help="durable $ ledger path")
    ap.add_argument("--out", default="research_upgrade/results/search/mock_run.json")
    ap.add_argument("--print-command", action="store_true", help="print the exact authorized pilot commands and exit")
    a = ap.parse_args()
    if a.print_command:
        print(pilot_commands()); return
    assert a.arm, "--arm is required (unless --print-command)"
    assert a.mock or a.paid, "choose --mock (offline) or --paid (real model, gated)"
    assert not (a.mock and a.paid), "--mock and --paid are mutually exclusive"

    real_model = None; cost_ledger = None; rpc_evaluator = None; model_id = "MockModel(deterministic,offline)"
    if a.paid:
        real_model, cost_ledger, rpc_evaluator, model_id = _build_paid(a)

    seed_src = open(os.path.join(ROOT, "experiments/exp11/env_d/injected_seed.py")).read()
    runs = []
    for r in range(a.runs):
        rng = random.Random(20260906 + r)
        model = real_model if a.paid else MockModel(rng)
        ledger = new_ledger(); log = []
        journal = Journal(os.path.join(ROOT, a.out) + f".run{r}.jsonl")   # durable per-candidate record; enables resume
        t0 = time.time()
        if a.arm == "independent":
            archive, status = run_independent(model, a.nseeds, a.budget, ledger, log, rpc_evaluator, journal)
        else:
            archive, status = run_evolution(
                model, a.nseeds, a.budget, ledger, log,
                seeded=(a.arm == "seeded"), seed_src=seed_src, rng=rng,
                feedback=(a.arm != "no_feedback"),
                selection=("random" if a.arm == "random_selection" else "fitness"),
                evaluator=rpc_evaluator, journal=journal)
        best = max([c for c in archive if c.valid], key=lambda c: c.score, default=None)
        seed_prov = None; invariant = None
        if a.arm == "seeded":
            seed_cand = next((c for c in archive if c.cid == "seed_factored"), None)
            seed_prov = {"in_archive": seed_cand is not None,
                         "seed_valid": bool(seed_cand and seed_cand.valid),
                         "seed_dev_score": (seed_cand.score if seed_cand else None),
                         "seed_src_sha8": (seed_cand.provenance.get("src_sha8") if seed_cand else None)}
            if seed_cand and seed_cand.valid and best is not None:
                # INVARIANT (same DEV instrument only; never DEV-vs-TEST): best-ever >= seed - tol
                invariant = {"assert": "best_ever_dev >= seed_dev - 1e-9",
                             "best_ever_dev": best.score, "seed_dev": seed_cand.score,
                             "holds": best.score >= seed_cand.score - 1e-9}
                assert invariant["holds"], f"seeded invariant VIOLATED: best {best.score} < seed {seed_cand.score}"
        runs.append({"run": r, "arm": a.arm, "status": status, "wall_s": round(time.time() - t0, 2),
                     "ledger": ledger, "n_archive": len(archive),
                     "n_valid": sum(1 for c in archive if c.valid),
                     "best_cid": best.cid if best else None, "best_score": best.score if best else None,
                     "best_src_sha8": (best.provenance.get("src_sha8") if best else None),
                     "seed_provenance": seed_prov, "seeded_invariant": invariant,
                     "budget_curve": log})
    out = {"arm": a.arm, "budget": a.budget, "nseeds": a.nseeds, "model": model_id,
           "paid": bool(a.paid),
           "cost_ledger": (cost_ledger.summary() if cost_ledger else None),
           "attempts_log": (real_model.attempts_log if real_model else None),
           "evaluator": ("trusted_rpc" if a.paid else "subprocess"), "runs": runs}
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=2)
    for rr in runs:
        L = rr["ledger"]
        print(f"[{a.arm} run{rr['run']}] status={rr['status']['status']} best={rr['best_score']} "
              f"valid={rr['n_valid']}/{rr['n_archive']} logical={L['logical_draws']} "
              f"eval_ticks_alloc={L['eval_ticks_allocated']} actual_ticks={L['actual_executed_ticks']} "
              f"invalid={L['invalid_children']} fallbacks={L['fallback_events']} "
              f"wall={rr['wall_s']}s")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
