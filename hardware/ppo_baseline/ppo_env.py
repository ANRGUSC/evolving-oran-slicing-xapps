#!/usr/bin/env python3
"""DemandEnvNull: Gymnasium environment for training the PPO comparator (paper Sec. V-D) on the
two-slice simulator with the observation model, the same surface the evolved controller is selected on:
  (i)  each episode follows the demand schedule (120 null / 240 good of 360 ticks) with a DEV seed (0-15),
       a null floor from {4.5, 5.1, 5.8} and a best-effort variant from {linear, cap200};
  (ii) the observation carries the live per-tick CQI (the signal the evolved controller reads);
  (iii) the reward is the simulator's per-tick composite, so in the null (where attainment is 0 at any
        share) the only lever is best-effort throughput and releasing the share is rewarded."""
from __future__ import annotations
import os, sys, random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in ("two_slice/engine",):
    q = str(ROOT / p);  sys.path.insert(0, q) if q not in sys.path else None
import simulator as B, observation_model as B3

PRB_MIN, PRB_MAX = 5, 90
TARGET = B.TARGET                    # 80
CEIL = B.CEIL                        # 220
ALPHA, S2_W, GAMMA, SIGMA = B.ALPHA, B.S2W, B.GAMMA, B.SIGMA
ENVS = B3.ENVS                       # (4.5, 5.1, 5.8)
VARIANTS = ("linear", "cap200")
DEV_SEEDS = list(range(16))          # DEV split (the seeds the evolved controller was selected on)
# Observation scales: cqi, cqi, Mbps, Mbps, PRB, PRB, target, target, pool.
_NM = float(CEIL)
_NORM = np.array([15.0, 15.0, _NM, _NM, 100.0, 100.0, _NM, _NM, 100.0], dtype=np.float32)

def make_obs(cqi1, cqi2, mbps1, mbps2, prb1, prb2, t1, t2, pool):
    return np.array([cqi1, cqi2, mbps1, mbps2, prb1, prb2, t1, t2, pool], dtype=np.float32) / _NORM

class DemandEnvNull(gym.Env):
    """Per-tick mirror of observation_model.replay. Observation carries the live CQI; reward = the composite."""
    def __init__(self, seeds=DEV_SEEDS):
        super().__init__()
        self.seeds = list(seeds)
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, (9,), np.float32)

    def _sample_tick_cqi(self):
        t = self.tk[self.i]
        self.cqi1 = B3._obs_cqi(self.crng, t["null"], self.floor)     # LIVE priority CQI (null-aware)
        self.cqi2 = B3._obs_cqi(self.crng, False, self.floor)          # best-effort CQI (good)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        s = self.seeds[self.np_random.integers(len(self.seeds))]
        self.floor = ENVS[self.np_random.integers(len(ENVS))]
        self.variant = VARIANTS[self.np_random.integers(len(VARIANTS))]
        self.rng = random.Random(1000 + s)                            # delivery-noise stream (== replay)
        self.crng = random.Random(7000 + s)                           # CQI-obs stream (== replay)
        self.tk, self.target = B.ticks()
        self.n = len(self.tk); self.i = 0
        self.prev_share = 35; self.mbps1 = 0.0; self.mbps2 = 0.0
        self._sample_tick_cqi()
        return self._obs(), {}

    def _obs(self):
        return make_obs(self.cqi1, self.cqi2, self.mbps1, self.mbps2,
                        self.prev_share, 100 - self.prev_share, self.target, 0.0, 100)

    def step(self, action):
        t = self.tk[self.i]
        frac = (float(action[0]) + 1.0) / 2.0
        s1 = max(5, min(90, int(round(PRB_MIN + frac * (PRB_MAX - PRB_MIN)))))
        s2 = B.POOL - s1
        eps = self.rng.gauss(0.0, SIGMA)
        if t["null"]:
            d1 = max(0.0, B.s1_null(s1) * (1.0 + eps))
        else:
            d1 = max(0.0, min(B.GOOD_SLOPE * s1 * (1.0 + eps), float(t["demand"])))
        d2, _ = B.s2_thp(s2, self.variant)
        need = min(t["demand"], self.target)
        ok = 1.0 if d1 >= need - 0.5 else 0.0
        churn = 1.0 if s1 != self.prev_share else 0.0
        reward = ok - ALPHA * (1.0 - ok) + S2_W * (d2 / CEIL) - GAMMA * churn   # = the simulator's per-tick composite
        self.mbps1, self.mbps2, self.prev_share = d1, d2, s1
        self.i += 1
        done = self.i >= self.n
        if not done:
            self._sample_tick_cqi()
            obs = self._obs()
        else:
            obs = np.zeros(9, dtype=np.float32)
        return obs, float(reward), done, False, {}

def make_null_ppo_controller(model):
    """Wrap a trained PPO policy as a controller(history, state, sla, predictors) that reads the live CQI."""
    def controller(history, state, sla, predictors):
        l1, l2 = history.ue1.last(), history.ue2.last()
        if l1 is None or l2 is None:
            h = sla.pool_max // 2; return h
        obs = make_obs(l1.cqi, l2.cqi, history.ue1.latest_mbps(), history.ue2.latest_mbps(),
                       l1.prb_share if l1.prb_share else 35,
                       (sla.pool_max - (l1.prb_share if l1.prb_share else 35)),
                       sla.sla_ue1.target_mbps, 0.0, sla.pool_max)
        a, _ = model.predict(obs, deterministic=True)
        frac = (float(a[0]) + 1.0) / 2.0
        return max(5, min(90, int(round(PRB_MIN + frac * (PRB_MAX - PRB_MIN)))))
    return controller
