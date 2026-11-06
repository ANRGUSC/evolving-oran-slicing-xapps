"""Sensor-calibrated evolution. 3 seeds x pop8 x 5 gens.
Prompt = facts block; bench = observation model; fitness = MIN
composite over null-floor envs {4.5,5.1,5.8} (worst-case selection), mean over DEV seeds + both variants.
GEN-0 = 16 fresh cold draws under the prompt + one ablated seed program as a declared seed member.
Directive-free mutation (diagnostics only). Per-gen: best/mean, null-ratio (realistic sensor), good-act
spurious-abandon count, detector form. Gate check on each gen champion. DEV vs held-out TEST split.

REFERENCE ONLY (not run by reproduce.py): this is the original two-slice evolution loop, kept so the
exact prompt and search procedure are inspectable. To actually re-run it you need: an Anthropic API key
(ANTHROPIC_API_KEY in the environment or a .env at the repo root), the helper
`evolution/llm_client.extract_code_block`, and an ablated seed program -- none of which
ship in this public repo. The task prompt itself IS included: search/prompt.txt."""
import os, sys, json, statistics, hashlib, random, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))   # simulator, observation_model, gates live in ../engine
import simulator as B, observation_model as B3, gates as G
from evolution.llm_client import extract_code_block   # helper; not shipped (reference-only)
_env = os.path.join(ROOT, ".env")
if os.path.exists(_env):
    for ln in open(_env):
        if ln.startswith("ANTHROPIC_API_KEY"): os.environ["ANTHROPIC_API_KEY"] = ln.split("=",1)[1].strip().strip('"')
import anthropic
CLIENT = anthropic.Anthropic()

PV3 = os.path.join(HERE, "prompt.txt")
raw = open(PV3).read()
SYS = raw.split("# ================= SYSTEM =================")[1].split("# ================= USER =================")[0].strip()
USER0 = raw.split("# ================= USER =================")[1].strip()
assert "MINIMUM share" not in SYS and "HARDWARE TELEMETRY" in SYS   # facts present, no strategy directive
POP, GENS, SEEDS = 8, 5, 3
DEV = list(range(0,16)); TEST = list(range(200,216)); ENVS = B3.ENVS
LEDGER = {"calls":0}
def llm(user):
    r = CLIENT.messages.create(model="claude-haiku-4-5", max_tokens=8192, system=SYS,
                               messages=[{"role":"user","content":user}]); LEDGER["calls"]+=1
    return r.content[0].text

def _load(src):
    ns={}
    try: exec(src, ns)
    except Exception: return None
    return ns.get("controller")
def score_env(c, seeds, floor):
    comps=[]; ft=tt=0
    for v in ("linear","cap200"):
        for s in seeds:
            try: r=B3.replay(("source",c), v, seed=s, null_floor=floor)
            except Exception: return None
            comps.append(r["composite"]); ft+=r["faults"]; tt+=r["n"]
    if ft/max(1,tt) > 0.05: return None
    return statistics.fmean(comps)
def fitness(src, seeds):
    """min over envs of mean(seeds,variants) composite (worst-case selection). None if unscoreable."""
    c=_load(src)
    if c is None: return None
    per={}
    for e in ENVS:
        s=score_env(c, seeds, e)
        if s is None: return None
        per[e]=s
    return min(per.values())
def diag(src, seeds):
    c=_load(src); rs=[B3.replay(("source",c),"linear",seed=s,null_floor=5.1) for s in seeds]
    return {"comp": fitness(src,seeds) or 0.0, "attain": statistics.fmean(r["attain"]/r["n"] for r in rs),
            "viol": statistics.fmean(100*r["viol"]/r["n"] for r in rs),
            "be": statistics.fmean(r["mean_s2"] for r in rs), "churn": statistics.fmean(r["churn"] for r in rs)}
def user_mutation(src, bd):   # directive-free: diagnostics only, strategy hints stripped
    return (f"Here is a controller with worst-case fitness {bd['comp']:+.4f}.\n"
            f"Diagnostics (means over evaluation, nominal sensor): priority attainment={bd['attain']:.3f}, "
            f"violation%={bd['viol']:.2f}, best-effort Mbps={bd['be']:.1f}, priority share-changes/episode={bd['churn']:.1f}.\n\n"
            f"```python\n{src}\n```\n\nProduce an IMPROVED variant that scores higher on the average score. "
            "Return ONLY a fenced ```python code block.")

def behav(src, floor=5.1):
    """null-ratio (realistic sensor) + good-act spurious-abandon count (share<=min+2 in a good act)."""
    c=_load(src); tk,_=B.ticks(); sla=B.Sla(80); st=B.State(); h=B.Hist()
    crng=random.Random(9999); prev=35; ps1=ps2=0.0; nb=[]; gb=[]; spur=0
    for t in tk:
        cqi=B3._obs_cqi(crng,t["null"],floor)
        h.ue1.push(B.Tick(cqi,prev,ps1,ps1*125000.0)); h.ue2.push(B.Tick(15,100-prev,ps2,ps2*125000.0))
        try: s1=max(5,min(90,int(B._share_of(c(h,st,sla,{})))))
        except Exception: s1=prev
        if t["null"]: nb.append(s1)
        else:
            gb.append(s1)
            if s1<=7: spur+=1
        eps=crng.gauss(0,B.SIGMA)
        ps1=max(0,B.s1_null(s1)*(1+eps)) if t["null"] else max(0,min(B.GOOD_SLOPE*s1*(1+eps),float(t["demand"])))
        ps2=B.s2_thp(100-s1,"linear")[0]; prev=s1
    gm=statistics.fmean(gb) if gb else 1
    return {"null_ratio": (statistics.fmean(nb)/gm if gb else float("nan")), "spurious_abandon": spur,
            "good_share": round(gm,1), "null_share": round(statistics.fmean(nb),1) if nb else 0}
def detector_form(src):
    forms=[]
    if re.search(r'cqi\s*[<>]=?\s*\d', src): forms.append("absolute-threshold")
    if re.search(r'(mean|avg|sum|history\[|cqi_hist|/\s*len)', src) and re.search(r'cqi', src): forms.append("relative/avg")
    if re.search(r'(counter|window|deque|debounce|consecutive|streak|>=\s*\d.*counter)', src, re.I): forms.append("debounced")
    return "+".join(dict.fromkeys(forms)) or "none/other"

# ---- GEN-0: 16 fresh cold draws under the prompt + one ablated seed ----
print("=== GEN-0: 16 cold draws + seed ===")
gen0=[]
for i in range(16):
    try: src=extract_code_block(llm(USER0)); f=fitness(src,DEV)
    except Exception as e: src=None; f=None
    if src and f is not None: gen0.append(src); print(f"  draw{i:2d}: fitness(worst-env)={f:+.4f} detector={detector_form(src)}")
    else: print(f"  draw{i:2d}: unscoreable/crash")
abl=open(os.path.join(ROOT,"ablated_seed.py")).read()
gen0.append(abl); print(f"  SEED: ablated seed fitness(worst-env)={fitness(abl,DEV):+.4f} detector={detector_form(abl)}")
scored0=sorted(((fitness(s,DEV),s) for s in gen0), key=lambda x:-(x[0] or -9))
top8=[s for f,s in scored0 if f is not None][:8]
json.dump({"gen0_n":len(gen0),"top8_fit":[round(fitness(s,DEV),4) for s in top8]},
          open(os.path.join(HERE,"gen0_v3.json"),"w"), indent=2)

# ---- evolve ----
ALL=[]; genlog=open(os.path.join(HERE,"evolve_v3.jsonl"),"w")
for seed in range(SEEDS):
    rng=random.Random(42+seed); pop=[(f"s{seed}g0c{i}",s,fitness(s,DEV)) for i,s in enumerate(top8)]
    for gen in range(GENS):
        pop=[p for p in pop if p[2] is not None]; pop.sort(key=lambda p:-p[2]); best=pop[0]
        bm=behav(best[1]); df=detector_form(best[1])
        rec={"seed":seed,"gen":gen,"best":round(best[2],4),"mean":round(statistics.fmean(p[2] for p in pop),4),
             "null_ratio":round(bm["null_ratio"],3),"spurious_abandon":bm["spurious_abandon"],
             "good_share":bm["good_share"],"null_share":bm["null_share"],"detector":df,"calls":LEDGER["calls"]}
        genlog.write(json.dumps(rec)+"\n"); genlog.flush()
        print(f"  seed{seed} gen{gen}: best={rec['best']:+.3f} mean={rec['mean']:+.3f} null_ratio={rec['null_ratio']:.2f} "
              f"spur_abandon={rec['spurious_abandon']} detector={df} calls={LEDGER['calls']}")
        ALL.extend([(p[0],p[1],p[2]) for p in pop])
        if gen==GENS-1: break
        parents=pop[:max(2,POP//2)]; newpop=list(parents)
        while len(newpop)<POP:
            par=parents[rng.randrange(len(parents))]
            try: off=extract_code_block(llm(user_mutation(par[1],diag(par[1],DEV))))
            except Exception: off=None
            sc=fitness(off,DEV) if off else None
            newpop.append((f"s{seed}g{gen+1}c{len(newpop)}", off if off else par[1], sc if sc is not None else par[2]))
        pop=newpop
genlog.close()

# ---- champion + TEST (once) + G7 + criteria ----
ALL=[a for a in ALL if a[2] is not None]; champ=max(ALL,key=lambda a:a[2])
csrc=champ[1]; c=_load(csrc)
test={}
for e in [5.1, min(ENVS)]:
    test[e]={v: statistics.fmean(B3.replay(("source",c),v,seed=s,null_floor=e)["composite"] for s in TEST) for v in ("linear","cap200")}
g7=G.G7(c, null_floor=5.1); bm=behav(csrc)
res={"champion_id":champ[0],"dev_fitness_worstenv":round(champ[2],4),
     "TEST_nominal_5.1":{k:round(v,4) for k,v in test[5.1].items()},
     "TEST_worstenv":{k:round(v,4) for k,v in test[min(ENVS)].items()},
     "frozen_const35":0.933,"WIN_threshold":0.963,
     "WIN_nominal_linear":bool(test[5.1]["linear"]>=0.963),"WIN_worst_linear":bool(test[min(ENVS)]["linear"]>=0.963),
     "null_ratio":round(bm["null_ratio"],3),"spurious_abandon_good":bm["spurious_abandon"],
     "G7_null_pass":g7["null_pass"],"G7_good_pass":g7["good_pass"],"detector":detector_form(csrc),
     "total_calls":LEDGER["calls"]}
json.dump({**res,"champion_source":csrc}, open(os.path.join(HERE,"RESULT_v3.json"),"w"), indent=2)
print("\n=== CHAMPION",champ[0],"===")
print(json.dumps(res, indent=2))
