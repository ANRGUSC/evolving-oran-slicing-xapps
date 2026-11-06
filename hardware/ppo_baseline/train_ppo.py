#!/usr/bin/env python3
"""Train the PPO comparator of the hardware experiment (paper Sec. V-D).

Trains one PPO agent (Stable-Baselines3, MLP policy) per seed on the two-slice simulator with
the observation model (ppo_env.DemandEnvNull), 10^6 environment steps each at learning rate
3e-4, and scores each agent with the same worst-floor fitness as the evolved controller:
min over the three null floors of the mean over the 16 DEV seeds and both best-effort variants.
The paper's five seeds are 42, 7, 123, 2024 and 5 (scores in ppo_experiment_records.json);
the deployed checkpoint ppo_policy.zip is the seed-42 agent.

Writes retrained/ppo_seed<seed>.zip for every seed and retrained/ppo_result.json with the
per-seed scores and the best seed. Needs torch, stable-baselines3 and gymnasium
(requirements.txt); the run takes a while. The new agents are not bit-identical to the
archived ones (training is not deterministic across library versions and hardware).

    python train_ppo.py [--seeds 42 7 123 2024 5] [--steps 1000000] [--out retrained]
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("two_slice/engine", "hardware/ppo_baseline"):
    q = str(ROOT / p)
    if q not in sys.path:
        sys.path.insert(0, q)
import observation_model as B3   # noqa: E402
from scorer import NULL_MASK     # noqa: E402  (null mask of the development schedule)
from ppo_env import DemandEnvNull, make_null_ppo_controller, ENVS, VARIANTS, DEV_SEEDS  # noqa: E402

STATIC_BAR, EVOLVED = 0.9300, 0.9398
DEFAULT_SEEDS = (42, 7, 123, 2024, 5)


def worst_floor_fitness(ctrl):
    """DEV worst-floor composite of a callable controller (the scorer's aggregation)."""
    per = {}
    for floor in ENVS:
        cs = [B3.replay(("source", ctrl), v, seed=s, null_floor=floor)["composite"]
              for v in VARIANTS for s in DEV_SEEDS]
        per[floor] = statistics.fmean(cs)
    return min(per.values()), {str(k): round(v, 4) for k, v in per.items()}


def null_behavior(ctrl):
    tr = [int(x) for x in B3.replay(("source", ctrl), "linear", seed=0, null_floor=5.1)["shares"]]
    n = [s for s, m in zip(tr, NULL_MASK) if m]
    g = [s for s, m in zip(tr, NULL_MASK) if not m]
    return round(statistics.fmean(n), 1), round(statistics.fmean(g), 1), min(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--steps", type=int, default=int(os.environ.get("PPO_STEPS", "1000000")))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "retrained")
    a = ap.parse_args()
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    a.out.mkdir(parents=True, exist_ok=True)

    print(f"=== PPO on the two-slice simulator: seeds {a.seeds}, {a.steps} steps each, lr {a.lr} ===")
    runs = []
    for seed in a.seeds:
        env = DummyVecEnv([lambda: DemandEnvNull()])
        model = PPO("MlpPolicy", env, learning_rate=a.lr, seed=seed, verbose=0)
        t0 = time.time()
        model.learn(total_timesteps=a.steps)
        dt = time.time() - t0
        ctrl = make_null_ppo_controller(model)
        fitness, per_floor = worst_floor_fitness(ctrl)
        null_share, good_share, null_min = null_behavior(ctrl)
        path = a.out / f"ppo_seed{seed}.zip"
        model.save(str(path))
        runs.append({"seed": seed, "worstenv": round(fitness, 4), "per_floor": per_floor,
                     "null_share": null_share, "good_share": good_share, "null_min": null_min,
                     "clears_bar": fitness > STATIC_BAR, "checkpoint": path.name, "train_s": round(dt, 1)})
        print(f"  seed {seed}: fitness {fitness:+.4f}  per floor {per_floor}  "
              f"null share {null_share} good share {good_share}  ({dt:.0f} s)")
    scores = [r["worstenv"] for r in runs]
    best = max(runs, key=lambda r: r["worstenv"])
    result = {"steps": a.steps, "learning_rate": a.lr, "runs": runs,
              "mean": round(statistics.fmean(scores), 4),
              "sd": round(statistics.pstdev(scores), 4) if len(scores) > 1 else 0.0,
              "clears_bar": sum(r["clears_bar"] for r in runs), "best_seed": best["seed"],
              "static_bar": STATIC_BAR, "evolved_controller": EVOLVED}
    json.dump(result, open(a.out / "ppo_result.json", "w"), indent=2)
    print(f"\n=== RESULT: mean {result['mean']:+.4f} +/- {result['sd']:.4f}; "
          f"{result['clears_bar']} of {len(runs)} above the static bar {STATIC_BAR}; "
          f"best seed {best['seed']} ({best['worstenv']:+.4f}); evolved controller {EVOLVED} ===")
    print(f"written: {a.out / 'ppo_result.json'}")


if __name__ == "__main__":
    main()
