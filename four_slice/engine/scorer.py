"""Scorer of the four-slice experiment: evaluates a controller(history, state, sla, predictors) under
simulator.py with one-tick-lagged observations, guaranteed-cap feasibility, the composite reward of Eq. 3 and
worst-env aggregation (min over the 16 development seeds of the episode-mean composite). It exposes the
history / sla / state objects described in the task prompt (search/prompt.txt). No API calls."""
import statistics
import simulator as E


class _Sample:
    __slots__ = ("cqi", "prb_share", "dl_bytes")
    def __init__(self, cqi, share, mbps):
        self.cqi, self.prb_share, self.dl_bytes = cqi, share, mbps * 1e6 / 8.0


class _Hist:
    def __init__(self): self._t = []
    def push(self, smp): self._t.append(smp)
    def last(self): return self._t[-1] if self._t else None
    def latest_mbps(self): return (self._t[-1].dl_bytes * 8 / 1e6) if self._t else 0.0


class _Slices:
    def __init__(self, n): self.slices = [_Hist() for _ in range(n)]


class _SlaSlice:
    def __init__(self, w, t): self.weight, self.target_mbps = w, t


class _Sla:
    def __init__(self):
        self.slices = [_SlaSlice(E.W[k], E.T[k]) for k in range(E.NG)]
        self.pool_max, self.guaranteed_cap, self.n, self.best_effort = E.POOL, E.GCAP, E.NG, object()


class _State:
    def __init__(self): self.d = {}
    def get(self, k, d=None): return self.d.get(k, d)
    def set(self, k, v): self.d[k] = v


def eval_prompt_controller(ctrl_fn, seed, **kw):
    """One env seed -> mean per-tick composite. Lagged obs (cqi/share/delivery of tick t-1); scoring
    always uses the TRUE e_k(t)."""
    EFF, CQI = E.eff_series(seed, **kw), E.cqi_series(seed, **kw)
    hist, sla, st = _Slices(E.NG), _Sla(), _State()
    prev, prev_deliv, comps = None, [0.0] * E.NG, []
    for t in range(E.TICKS):
        ti = t - 1 if t > 0 else 0
        for k in range(E.NG):
            hist.slices[k].push(_Sample(CQI[k][ti], (prev[k] if prev else 0), prev_deliv[k]))
        s = E.feasible(ctrl_fn(hist, st, sla, None))
        e_true = [EFF[k][t] for k in range(E.NG)]
        comps.append(E.composite_tick(e_true, s, prev))
        prev, prev_deliv = s, [e_true[k] * s[k] for k in range(E.NG)]
    return statistics.fmean(comps)


def worst_env_prompt(ctrl_fn, seeds=E.DEV, **kw):
    return min(eval_prompt_controller(ctrl_fn, s, **kw) for s in seeds)
