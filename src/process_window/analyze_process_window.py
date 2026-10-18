"""Frozen-checkpoint, Train-only process-window discovery and later-date audit.

Run from Final_Welding: python src/process_window/analyze_process_window.py
No checkpoint, existing policy, or source data is changed.
"""
from pathlib import Path
import sys
import json
import itertools
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from src.predict import load_model
from src.config import FEATURES

SENSORS = FEATURES[2:]
SEEDS = [17, 42, 97]
OUT = ROOT / 'results/tables/process_window'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / 'data/processed/weld_inputs.csv')
    train = frame.loc[frame.split.eq('train')].copy()
    models = [load_model('DNN_GAMI2', seed) for seed in SEEDS]
    cuts = np.array([np.quantile(m.score(train[FEATURES]), .5) for m in models])
    success_cuts = np.array([m.policy['success_below'] for m in models])
    # These are sensitivity scenarios, not verified machine/control tolerances.
    steps = np.array([.01, .01, .001, 1.])
    lo, hi = train[SENSORS].min().to_numpy(), train[SENSORS].max().to_numpy()
    med = train[SENSORS].median().to_numpy()
    scale = (train[SENSORS].quantile(.75) - train[SENSORS].quantile(.25)).to_numpy()
    scale = np.where(scale > 1e-12, scale, train[SENSORS].std().to_numpy())
    unique = train[SENSORS].drop_duplicates().to_numpy()
    nn = NearestNeighbors(n_neighbors=6).fit((unique-med)/scale)
    # Fifth OTHER distinct neighbor defines local observed support.
    support_cut = float(np.quantile(nn.kneighbors((unique-med)/scale)[0][:, -1], .95))

    def supported(x):
        d = nn.kneighbors((x-med)/scale, n_neighbors=5)[0][:, -1]
        return ((x >= lo-1e-12) & (x <= hi+1e-12)).all(1) & (d <= support_cut+1e-12)

    def scores(x):
        f = pd.DataFrame(x, columns=SENSORS)
        for i, col in enumerate(FEATURES[:2]):
            f[col] = float(models[0].fixed_values[i])
        return np.column_stack([m.score(f[FEATURES]) for m in models])

    def stable(x):
        base = scores(x)
        base_low = (base <= cuts).all(1) & (base < success_cuts).all(1)
        worst = base.copy()
        perturb_support = np.ones(len(x), dtype=bool)
        # Axis perturbations keep the derived resistance recalculated from V/I.
        for j in range(4):
            for sign in [-1, 1]:
                z = x.copy(); z[:, j] += sign*steps[j]
                perturb_support &= supported(z)
                worst = np.maximum(worst, scores(z))
        robust = (worst <= cuts).all(1) & (worst < success_cuts).all(1)
        # Nominal inputs require joint support. Perturbation support is reported
        # separately: unsupported perturbations cannot be physical validation.
        return base, worst, base_low, robust & supported(x), perturb_support

    x = frame[SENSORS].to_numpy()
    base, worst, low, robust, perturb_support = stable(x)
    rows = frame.copy()
    for i, seed in enumerate(SEEDS):
        rows[f'score_seed{seed}'] = base[:, i]
        rows[f'worst_axis_score_seed{seed}'] = worst[:, i]
    rows['joint_train_support'] = supported(x)
    rows['all_seeds_low'] = low
    rows['robust_low_candidate'] = robust
    rows['all_axis_perturbations_supported'] = perturb_support
    rows.to_csv(OUT/'row_diagnostics.csv', index=False, encoding='utf-8-sig')

    train_mask = frame.split.eq('train').to_numpy()
    # Conservative target: low scores AND all perturbations within support.
    target = robust & perturb_support
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=200, random_state=20261007)
    tree.fit(x[train_mask], target[train_mask])
    candidates = []

    def walk(node, lower, upper, lower_open):
        t = tree.tree_
        if t.children_left[node] != t.children_right[node]:
            j, v = t.feature[node], float(t.threshold[node])
            u = upper.copy(); u[j] = min(u[j], v)
            walk(t.children_left[node], lower.copy(), u, lower_open.copy())
            l = lower.copy(); l[j] = max(l[j], v)
            op = lower_open.copy(); op[j] = True
            walk(t.children_right[node], l, upper.copy(), op)
            return
        mask = (tree.apply(x[train_mask]) == node)
        selected = train.loc[mask]
        n = len(selected)
        frac = float(target[train_mask][mask].mean())
        dates = selected.groupby('date').size()
        if frac < .95 or len(dates) < 3 or (dates >= 20).sum() < 3:
            return
        # The leaf's actual observed envelope is easier to interpret than
        # unobserved tree cut edges; evaluate this envelope directly below.
        lower = selected[SENSORS].min().to_numpy()
        upper = selected[SENSORS].max().to_numpy()
        candidates.append({'leaf': int(node), 'lower': lower.tolist(), 'upper': upper.tolist(),
                           'train_leaf_rows': n, 'train_leaf_stable_fraction': frac,
                           'train_leaf_dates': dates.to_dict()})

    walk(0, lo.copy(), hi.copy(), np.zeros(4, dtype=bool))
    # Freeze the Train-only discovery rules before later-date reporting.
    protocol = {'selected_checkpoint_seed': 17, 'sensitivity_seeds': SEEDS,
                'low_score_definition': 'at/below each seed Train median AND below its frozen success threshold',
                'score_median_cuts': cuts.tolist(), 'success_cuts': success_cuts.tolist(),
                'support_definition': 'within Train bounds and distance to 5 distinct Train neighbors <= Train unique leave-self-out 95th percentile fifth-neighbor distance, IQR scaling',
                'support_cut': support_cut, 'sensor_order': SENSORS,
                'axis_perturbation_steps': steps.tolist(),
                'perturbations_are_verified_control_tolerances': False,
                'discovery_split': 'train only; checkpoint itself previously selected on inner Train',
                'tree': {'max_depth': 5, 'min_samples_leaf': 200, 'min_stable_fraction': .95, 'min_dates_with_20_rows': 3},
                'later_dates_role': 'audit only; no range or threshold reselection',
                'causal_or_inspection_validation': False,
                'scope': {'machine': sorted(frame.machine.unique()), 'item': sorted(frame.item.unique()), 'thickness_mm': [float(v) for v in models[0].fixed_values[:2]]},
                'data_sha256': digest(ROOT/'data/processed/weld_inputs.csv'),
                'checkpoint_sha256': {str(s): digest(ROOT/f'results/models/DNN_GAMI2/seeds/seed{s}/dnn_gami2_weights.pt') for s in SEEDS},
                'candidate_count': len(candidates)}
    (OUT/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    audit = []
    for k, c in enumerate(candidates, 1):
        c['candidate'] = f'W{k}'
        a, b = np.array(c['lower']), np.array(c['upper'])
        inside = ((x >= a-1e-12) & (x <= b+1e-12)).all(1)
        rows[c['candidate']] = inside
        for split in ['train', 'validation', 'calibration', 'test']:
            mask = frame.split.eq(split).to_numpy() & inside
            n = int(mask.sum())
            audit.append({'candidate': c['candidate'], 'split': split, 'rows': n,
                          'dates': int(frame.loc[mask, 'date'].nunique()),
                          'consensus_low_fraction': float(low[mask].mean()) if n else None,
                          'supported_fraction': float(supported(x[mask]).mean()) if n else None,
                          'robust_supported_fraction': float(target[mask].mean()) if n else None,
                          'selected_score_median': float(np.median(base[mask, 0])) if n else None})
        # Deterministic 5^4 grid; unsupported combinations remain excluded.
        grid = np.array(list(itertools.product(*[np.linspace(aa, bb, 5) for aa, bb in zip(a, b)])))
        gb, gw, gl, gr, gp = stable(grid)
        gs = supported(grid)
        c.update(grid_points=len(grid), grid_supported_points=int(gs.sum()),
                 grid_supported_fraction=float(gs.mean()),
                 grid_consensus_low_fraction_within_support=float(gl[gs].mean()) if gs.any() else None,
                 grid_robust_supported_fraction=float((gr & gp).mean()),
                 joint_support_gate_required=True)
        gd = pd.DataFrame(grid, columns=SENSORS)
        gd['supported'] = gs; gd['all_seeds_low'] = gl
        gd['robust_and_all_perturbations_supported'] = gr & gp
        gd.to_csv(OUT/f'{c["candidate"]}_joint_grid.csv', index=False, encoding='utf-8-sig')
    rows.to_csv(OUT/'row_diagnostics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(audit).to_csv(OUT/'candidate_audit.csv', index=False, encoding='utf-8-sig')
    (OUT/'candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'rows': len(frame), 'train_rows': len(train),
               'by_split': rows.groupby('split')[['all_seeds_low', 'robust_low_candidate', 'all_axis_perturbations_supported']].mean().to_dict(),
               'candidates': candidates,
               'train_score_quantiles': {str(seed): np.quantile(base[train_mask, i], [0,.25,.5,.75,.95,1]).tolist() for i,seed in enumerate(SEEDS)}}
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    # Train-only refinement: discrete time setpoints and nested central sensor
    # intervals. No later-date observations enter these boundaries or ranking.
    refined = []
    search_count = 0
    for c in candidates:
        ca, cb = np.array(c['lower']), np.array(c['upper'])
        cm = train_mask & ((x >= ca-1e-12) & (x <= cb+1e-12)).all(1) & target
        for duration in np.unique(x[cm, 3]):
            local = x[cm & np.isclose(x[:, 3], duration)]
            if len(local) < 100:
                continue
            for q in [.1, .2, .3, .4]:
                a, b = np.quantile(local, [q, 1-q], axis=0)
                a[3] = b[3] = duration
                mask = train_mask & ((x >= a-1e-12) & (x <= b+1e-12)).all(1)
                day_counts = frame.loc[mask].groupby('date').size()
                if mask.sum() < 100 or (day_counts >= 20).sum() < 3:
                    continue
                search_count += 1
                grid = np.array(list(itertools.product(*[np.unique(np.linspace(aa, bb, 5)) for aa,bb in zip(a,b)])))
                _, _, gl, gr, gp = stable(grid)
                gs = supported(grid)
                pass_fraction = float((gr & gp).mean())
                if gs.mean() < .95 or pass_fraction < .95 or target[mask].mean() < .95:
                    continue
                refined.append({'lower': a.tolist(), 'upper': b.tolist(), 'train_rows': int(mask.sum()),
                                'train_dates': day_counts.to_dict(), 'train_stable_fraction': float(target[mask].mean()),
                                'grid_points': len(grid), 'grid_supported_fraction': float(gs.mean()),
                                'grid_robust_supported_fraction': pass_fraction, 'central_quantile': q,
                                'parent': c['candidate']})
    refined.sort(key=lambda c: (-c['train_rows'], -c['grid_robust_supported_fraction']))
    # Keep the largest admitted range; other nested ranges remain in the log.
    for i, c in enumerate(refined, 1):
        c['candidate'] = f'C{i}'
    selected_core = refined[:1]
    core_audit = []
    for c in selected_core:
        a, b = np.array(c['lower']), np.array(c['upper'])
        inside = ((x >= a-1e-12) & (x <= b+1e-12)).all(1)
        rows[c['candidate']] = inside
        for split in ['train', 'validation', 'calibration', 'test']:
            mask = frame.split.eq(split).to_numpy() & inside
            outside = frame.split.eq(split).to_numpy() & ~inside
            n = int(mask.sum())
            core_audit.append({'candidate': c['candidate'], 'split': split, 'rows': n,
                               'dates': int(frame.loc[mask, 'date'].nunique()),
                               'consensus_low_fraction': float(low[mask].mean()) if n else None,
                               'robust_supported_fraction': float(target[mask].mean()) if n else None,
                               'score_median_inside': float(np.median(base[mask,0])) if n else None,
                               'score_median_outside': float(np.median(base[outside,0])) if outside.any() else None})
    (OUT/'core_candidates.json').write_text(json.dumps({'train_search_count': search_count, 'all_admitted': refined, 'selected': selected_core}, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(core_audit).to_csv(OUT/'core_audit.csv', index=False, encoding='utf-8-sig')
    rows.to_csv(OUT/'row_diagnostics.csv', index=False, encoding='utf-8-sig')
    print('CORE', json.dumps({'candidates': refined, 'audit': core_audit}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
