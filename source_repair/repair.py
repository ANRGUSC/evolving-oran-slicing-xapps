"""Apply the 11.8 -> 2.19 source edit and evaluate Fig. 7 on 16 traces."""
import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import statistics as st
import subprocess
import sys

from evaluation_helpers import numeric_edits_only, replay, write_csv

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT.parent/'four_slice'/'engine'
RUN = 'evolution_from_prompting_r3'
INPUTS = {'controller.py': ROOT/'inputs'/'controller.py',
          'simulator.py': ENGINE/'simulator.py',
          'scorer.py': ENGINE/'scorer.py'}
OLD = 'rate_k = max(0.2, (cqi - 5.0) / 10.0 * 11.8 + 0.2)'
NEW = 'rate_k = max(0.2, (cqi - 5.0) / 10.0 * 2.19 + 0.2)'


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def summarize(name, per):
    summary = {'controller': RUN, 'variant': name,
               'fitness': min(x['episode_reward'] for x in per),
               'mean16': st.fmean(x['episode_reward'] for x in per)}
    for key in ['attainment_pct', 'be_mbps', 'guarantee_reward', 'churn_count']:
        summary[key] = st.fmean(x[key] for x in per)
    for kind in ['release', 'restore']:
        total = sum(x[kind+'_events'] for x in per)
        summary[kind+'_events'] = total
        summary[kind+'_success'] = sum(x[kind+'_success'] for x in per)
        summary[kind+'_pct'] = 100*summary[kind+'_success']/total if total else None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results')
    parser.add_argument('--skip-figure', action='store_true', help='Evaluate using only Python stdlib.')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    E = import_path('simulator', INPUTS['simulator.py'])
    H = import_path('scorer', INPUTS['scorer.py'])
    if list(E.DEV) != list(range(16)) or E.TICKS != 360:
        raise ValueError('Expected development seeds 0..15 and 360 ticks.')

    original = INPUTS['controller.py'].read_text()
    if original.count(OLD) != 1:
        raise ValueError('Expected exactly one occurrence of the coefficient expression.')
    edited = original.replace(OLD, NEW, 1)
    edited = edited.replace(
        '# CQI-based estimation: CQI 5..15 maps to 0.2..12 Mbps/share',
        '# CQI-based fallback: CQI 5..15 maps to 0.2..2.39 Mbps/share', 1)
    changes = numeric_edits_only(original, edited)
    if changes != [{'before': 11.8, 'after': 2.19}]:
        raise ValueError(f'Unexpected executable change: {changes}')
    sources = out/'sources'
    sources.mkdir(exist_ok=True)
    (sources/'controller_original.py').write_text(original)
    repaired_source = '# Repaired controller: coefficient 11.8 -> 2.19.\n' + edited[edited.index('def controller('):]
    (sources/'controller_rate_fix.py').write_text(repaired_source)
    (out/'one_parameter.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), edited.splitlines(True),
        fromfile='controller_original.py', tofile='controller_rate_fix.py')))

    per_seed, summaries, by_variant = [], [], {}
    for name, source in [('original', original), ('edited', edited)]:
        per = []
        for seed in E.DEV:
            values, _ = replay(source, seed, E, H)
            per.append(values)
            per_seed.append({'variant': name, 'dev_seed': seed, **values})
        by_variant[name] = per
        summary = summarize(name, per)
        summaries.append(summary)
    pairs = []
    for seed, a, b in zip(E.DEV, by_variant['original'], by_variant['edited']):
        pairs.append({'dev_seed': seed, 'original': a['episode_reward'],
                      'edited': b['episode_reward'],
                      'difference': b['episode_reward']-a['episode_reward']})
    improved = sum(x['difference'] > 1e-12 for x in pairs)
    worsened = sum(x['difference'] < -1e-12 for x in pairs)
    comparison = {'improved_traces': improved, 'worsened_traces': worsened,
                  'tied_traces': len(pairs)-improved-worsened,
                  'fitness_change': summaries[1]['fitness']-summaries[0]['fitness'],
                  'mean_episode_change': st.fmean(x['difference'] for x in pairs),
                  'min_episode_change': min(x['difference'] for x in pairs),
                  'max_episode_change': max(x['difference'] for x in pairs)}
    cqi = [5+i/20 for i in range(201)]
    plot_data = {
        'controller': RUN, 'original': summaries[0], 'edited': summaries[1], 'pairs': pairs,
        'calibration': {'original_coefficient': 11.8, 'edited_coefficient': 2.19},
        'curves': [{'cqi': c, 'original': max(.2, .2+11.8*(c-5)/10),
                    'edited': max(.2, .2+2.19*(c-5)/10), 'nominal': E.e_of_cqi(c)} for c in cqi],
        'comparison': comparison,
    }
    (out/'plot_data.json').write_text(json.dumps(plot_data, indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    (out/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    write_csv(out/'summary.csv', summaries)
    write_csv(out/'per_seed.csv', per_seed)
    write_csv(out/'paired_scores.csv', pairs)
    if not args.skip_figure:
        subprocess.run([sys.executable, str(ROOT/'plot_fig7.py'),
                        '--data', str(out/'plot_data.json'), '--out', str(out)], check=True)
    print('Source-repair results written.')


if __name__ == '__main__':
    main()
