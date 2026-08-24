from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from impls.diagnostics.checkpoints import actor_run_dir, actor_sources

from common import protocol_from_arg, write_json


def read_csv(path):
    with Path(path).open(newline='') as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else ['empty']
    with Path(path).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rank(values):
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def spearman(x, y):
    if len(x) < 2:
        return None
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def endpoint_success(protocol, actor_name, seed):
    source = actor_sources(protocol)[actor_name]
    run_dir = actor_run_dir(source, run_root=protocol['run_root'], environment=protocol['environment'], seed=seed)
    path = run_dir / 'eval.csv'
    if not path.is_file():
        return None
    rows = read_csv(path)
    candidates = [row for row in rows if int(float(row.get('step', -1))) == 1_000_000]
    if not candidates or 'evaluation/overall_success' not in candidates[-1]:
        return None
    return float(candidates[-1]['evaluation/overall_success'])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.yaml')))
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    protocol = protocol_from_arg(args.protocol)
    root, output = Path(args.root), Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite aggregate output: {output}')
    output.mkdir(parents=True)
    summary_rows, pairwise_rows = [], []
    for summary_path in sorted(root.rglob('summary.csv')):
        eval_dir = summary_path.parent
        bank = eval_dir.name
        seed = next((part for part in eval_dir.parts if part.startswith('seed_')), 'unknown')
        for row in read_csv(summary_path):
            summary_rows.append({'seed': seed, 'bank': bank, **row})
        pairwise_path = eval_dir / 'pairwise_contrasts.json'
        if pairwise_path.is_file():
            with pairwise_path.open() as file:
                pairwise_rows.extend({'seed': seed, 'bank': bank, **row} for row in json.load(file))
    write_csv(output / 'aggregate_summary.csv', summary_rows)
    write_csv(output / 'aggregate_pairwise.csv', pairwise_rows)
    grouped = {}
    for row in pairwise_rows:
        key = (row['bank'], row['left_actor'], row['right_actor'])
        grouped.setdefault(key, []).append(row)
    means = []
    for (bank, left, right), rows in sorted(grouped.items()):
        fields = ('action_l2_mean', 'action_l2_squared_mean', 'q_delta_right_minus_left',
                  'q_delta_normalized_by_left_abs_mean', 'q_right_win_rate', 'q_tie_rate')
        stats = {}
        for field in fields:
            values = np.asarray([float(row[field]) for row in rows], dtype=float)
            stats[f'{field}_mean'] = float(np.mean(values))
            stats[f'{field}_median'] = float(np.median(values))
            stats[f'{field}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        means.append({
            'bank': bank, 'left_actor': left, 'right_actor': right, 'seed_count': len(rows),
            **stats,
        })
    write_csv(output / 'pairwise_means.csv', means)
    lookup = {(row['bank'], row['left_actor'], row['right_actor']): row['q_delta_right_minus_left_mean'] for row in means}
    gap_rows = []
    for left, right in [(row['left_actor'], row['right_actor']) for row in means]:
        if all((bank, left, right) in lookup for bank in ('B_T', 'B_DE', 'B_R')):
            t, de, r = (lookup[bank, left, right] for bank in ('B_T', 'B_DE', 'B_R'))
            gap_rows.append({'left_actor': left, 'right_actor': right, 'delta_q_T': t,
                             'delta_q_DE': de, 'delta_q_R': r, 'G_goal': de - t, 'G_state': r - de})
    unique_gap = {(row['left_actor'], row['right_actor']): row for row in gap_rows}
    write_csv(output / 'gap_decomposition.csv', list(unique_gap.values()))
    validity_rows = []
    for row in pairwise_rows:
        if not row['seed'].startswith('seed_') or row['bank'] not in {'B_T', 'B_DE', 'B_R'}:
            continue
        seed = int(row['seed'].split('_')[1])
        left_j, right_j = endpoint_success(protocol, row['left_actor'], seed), endpoint_success(protocol, row['right_actor'], seed)
        if left_j is not None and right_j is not None:
            dq = float(row['q_delta_right_minus_left'])
            dj = right_j - left_j
            validity_rows.append({'seed': seed, 'bank': row['bank'], 'left_actor': row['left_actor'],
                                  'right_actor': row['right_actor'], 'delta_metric': dq,
                                  'delta_success': dj, 'sign_agreement': int(np.sign(dq) == np.sign(dj))})
    write_csv(output / 'diagnostic_validity_seed_level.csv', validity_rows)
    validity_summary = []
    for key in sorted({(r['bank'], r['left_actor'], r['right_actor']) for r in validity_rows}):
        rows = [r for r in validity_rows if (r['bank'], r['left_actor'], r['right_actor']) == key]
        validity_summary.append({
            'bank': key[0], 'left_actor': key[1], 'right_actor': key[2], 'seed_count': len(rows),
            'sign_agreement_rate': float(np.mean([r['sign_agreement'] for r in rows])),
            'spearman_rank_correlation': spearman([r['delta_metric'] for r in rows], [r['delta_success'] for r in rows]),
            'interpretation': 'descriptive seed-block validity; no significance claim',
        })
    write_csv(output / 'diagnostic_validity_summary.csv', validity_summary)
    support_rows = []
    for support_path in sorted(root.rglob('support_proxy.npy')):
        eval_dir = support_path.parent
        if not eval_dir.name.startswith('B_R'):
            continue
        support = np.load(support_path)
        names = list(protocol['primary_actor_names'])
        samples = {name: np.load(eval_dir / name / 'metrics_raw.npz') for name in names}
        order = np.argsort(support, kind='stable')
        for quartile, indices in enumerate(np.array_split(order, 4), 1):
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    left_q, right_q = samples[left]['q_min'], samples[right]['q_min']
                    left_u = np.abs(samples[left]['q1'] - samples[left]['q2'])
                    right_u = np.abs(samples[right]['q1'] - samples[right]['q2'])
                    support_rows.append({'bank': eval_dir.name, 'support_quartile': quartile,
                                         'left_actor': left, 'right_actor': right, 'sample_count': len(indices),
                                         'q_gap_right_minus_left': float(np.mean(right_q[indices] - left_q[indices])),
                                         'left_disagreement_mean': float(np.mean(left_u[indices])),
                                         'right_disagreement_mean': float(np.mean(right_u[indices]))})
    write_csv(output / 'support_quartile_analysis.csv', support_rows)
    write_json(output / 'aggregate_manifest.json', {
        'root': str(root.resolve()), 'summary_row_count': len(summary_rows),
        'pairwise_row_count': len(pairwise_rows), 'gap_decomposition_rows': len(unique_gap),
        'validity_rows': len(validity_rows), 'support_quartile_rows': len(support_rows),
        'seed_is_model_statistical_block': True,
        'q_normalization': 'same-critic q gap divided by left actor mean absolute q plus epsilon',
        'uncertainty_note': 'sample-level bootstrap, if added later, is descriptive conditional uncertainty only',
    })
    print(f'Aggregate written: {output}')


if __name__ == '__main__':
    main()
