"""Faithful offline replay helpers copied from the verified parameter-edit experiment."""
import ast
import copy
import csv
import statistics as st
import types

def load_source(src):
    mod = types.ModuleType('controller_under_test')
    exec(compile(src, '<controller>', 'exec'), mod.__dict__)
    return mod.controller

def numeric_edits_only(original, edited):
    """Reject any executable AST change other than numerical constants."""
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) in (int, float):
                return ast.copy_location(ast.Constant(value=0), node)
            return node
    a, b = ast.parse(original), ast.parse(edited)
    av = [n.value for n in ast.walk(a) if isinstance(n, ast.Constant) and type(n.value) in (int, float)]
    bv = [n.value for n in ast.walk(b) if isinstance(n, ast.Constant) and type(n.value) in (int, float)]
    assert ast.dump(Normalize().visit(copy.deepcopy(a))) == ast.dump(Normalize().visit(copy.deepcopy(b)))
    assert len(av) == len(bv)
    changes = [{'before': x, 'after': y} for x, y in zip(av, bv) if x != y]
    assert 1 <= len(changes) <= 2, changes
    return changes

def replay(src, seed, E, H):
    fn = load_source(src)
    hist, state, sla = H._Slices(E.NG), H._State(), H._Sla()
    cq, eff = E.cqi_series(seed), E.eff_series(seed)
    prev, delivery = None, [0.0]*E.NG
    trace = []
    for t in range(E.TICKS):
        ti = max(0, t-1)
        for k in range(E.NG):
            hist.slices[k].push(H._Sample(cq[k][ti], prev[k] if prev else 0, delivery[k]))
        raw = list(fn(hist, state, sla, None))
        shares = E.feasible(raw)
        rates = [eff[k][t] for k in range(E.NG)]
        delivery = E.deliver(rates, shares)
        attained = [delivery[k] >= E.T[k]-E.OK_EPS for k in range(E.NG)]
        guar = sum(E.W[k] if attained[k] else -.5*E.W[k] for k in range(E.NG))
        churn = 0 if prev is None else sum(a != b for a, b in zip(shares+[E.be_share(shares)], prev+[E.be_share(prev)]))
        trace.append({'tick': t, 'raw': raw, 'shares': shares, 'cqi': [cq[k][t] for k in range(E.NG)],
                      'delivered': delivery, 'attained': attained,
                      'fitness_tick': E.composite_tick(rates, shares, prev),
                      'guarantee_reward': guar, 'be_mbps': E.E_BE*E.be_share(shares), 'churn_count': churn})
        prev = shares
    counters = dict(release_events=0, release_success=0, restore_events=0, restore_success=0)
    for k in range(E.NG):
        feasible = [eff[k][t]*E.GCAP >= E.T[k]-E.OK_EPS for t in range(E.TICKS)]
        for t in range(E.TICKS):
            was = feasible[t-1] if t else True
            window = range(t, min(t+8, E.TICKS))
            if was and not feasible[t]:
                counters['release_events'] += 1
                counters['release_success'] += any(trace[u]['shares'][k] == 0 for u in window)
            if t > 0 and not was and feasible[t]:
                counters['restore_events'] += 1
                counters['restore_success'] += any(trace[u]['attained'][k] for u in window)
    result = {key: st.fmean(row[key] for row in trace)
              for key in ('fitness_tick', 'guarantee_reward', 'be_mbps', 'churn_count')}
    result['episode_reward'] = result.pop('fitness_tick')
    result['attainment_pct'] = 100*st.fmean(sum(row['attained'])/E.NG for row in trace)
    result.update(counters)
    # Independent check against the unchanged bundled scalar evaluator.
    assert abs(result['episode_reward'] - H.eval_prompt_controller(load_source(src), seed)) < 1e-12
    return result, trace

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
