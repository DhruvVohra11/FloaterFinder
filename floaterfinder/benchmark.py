from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from . import core
from .data import load_dataset, load_config, sha256, write_json
from .detection import CONVENTIONS, checked_calibration, save_tiles

DETECTORS = ('organ_only', 'visual_only', 'combined')


def source_hash():
    root = Path(__file__).parent
    return {name: sha256(root / name) for name in ('core.py', 'data.py', 'benchmark.py', 'detection.py')}


def calibrate(dataset_path, config_path, output):
    cfg, data = load_config(config_path), load_dataset(dataset_path)
    if Path(output).exists():
        raise FileExistsError('Use a new calibration filename')
    result = {'dataset_fingerprint': data['fingerprint'], 'config': cfg,
              'source_hashes': source_hash(), 'models': {},
              'rule': 'Strictly greater than pooled tile-score quantile AFTER smoothing; no guarantee on new slides',
              'created_at': datetime.now(timezone.utc).isoformat()}
    for name, model in data['models'].items():
        raw = defaultdict(list)
        slides = [s for s in model['slides'].values() if s['meta']['split'] == 'calibration']
        if not slides:
            raise ValueError('Need calibration slides')
        missing = total = 0
        for slide in slides:
            own = model['labels'].index(slide['meta']['organ'])
            _, signals, valid = core.foreignness(slide['features'], slide['xy_um'], model['prototypes'], own, cfg)
            missing += int((~valid).sum())
            total += len(valid)
            for key, values in signals.items():
                raw[key].append(values)
        thresholds, observed = {}, {}
        for key in DETECTORS:
            values = np.concatenate(raw[key])
            thresholds[key] = float(np.quantile(values, cfg['calibration_quantile'], method='higher'))
            observed[key] = float((values > thresholds[key]).mean())
        result['models'][name] = {'thresholds': thresholds, 'calibration_flagged_tile_fraction': observed,
            'calibration_slides': len(slides), 'q2_unavailable_fraction': missing / total}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def pairs_for(slides, split, seed):
    """Sample unordered source pairs, then emit both directions exactly once."""
    pool = sorted(sid for sid, s in slides.items() if s['meta']['split'] == split)
    rng, selected = np.random.default_rng(seed), set()
    for host in pool:
        h = slides[host]['meta']
        for regime in ('same_organ', 'different_organ'):
            candidates = [sid for sid in pool if slides[sid]['meta']['patient_id'] != h['patient_id']
                and ((slides[sid]['meta']['organ'] == h['organ']) == (regime == 'same_organ'))]
            if candidates:
                donor = candidates[int(rng.integers(len(candidates)))]
                a, b = sorted((host, donor))
                selected.add((regime, a, b))
    pairs = []
    for regime, a, b in sorted(selected):
        pair_id = hashlib.sha256(json.dumps([split, a, b]).encode()).hexdigest()[:16]
        for host, donor in ((a, b), (b, a)):
            pairs.append({'pair_id': pair_id, 'host': host, 'donor': donor,
                          'regime': regime, 'seed': int(rng.integers(2**31))})
    return pairs


def reciprocal_bags(slides, directions, dose):
    """Build both bags from originals, or fail the entire pair at this dose."""
    a, b = directions
    if a['host'] != b['donor'] or a['donor'] != b['host']:
        raise ValueError('Expected reciprocal directions')
    return [core.inject(slides[p['host']], slides[p['donor']], dose, p['seed'])
            for p in directions]


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['model'], row['regime'], row['pooling'], row['method'], row['dose'])].append(row)
    result = []
    for (model, regime, pooling, method, dose), cases in groups.items():
        correct = [r for r in cases if r['clean_prediction'] == r['true_class']]
        corrupted = [r for r in correct if r['contaminated_prediction'] != r['true_class']]
        tp = sum(r['true_positive_tiles'] for r in cases)
        fp = sum(r['false_positive_tiles'] for r in cases)
        foreign = sum(r['foreign_tiles'] for r in cases)
        ranking = [r for r in cases if r.get('auroc') is not None]
        result.append({'model': model, 'regime': regime, 'pooling': pooling, 'method': method,
            'dose': dose, 'cases': len(cases), 'initially_correct_cases': len(correct),
            'corrupted_correct_cases': len(corrupted),
            'prediction_change_rate': float(np.mean([r['prediction'] != r['clean_prediction'] for r in cases])),
            'accuracy': float(np.mean([r['prediction'] == r['true_class'] for r in cases])),
            'error_rate_initially_correct': float(np.mean([r['prediction'] != r['true_class'] for r in correct])) if correct else None,
            'recovery_rate': float(np.mean([r['prediction'] == r['true_class'] for r in corrupted])) if corrupted else None,
            'detection_precision': tp / (tp+fp) if tp+fp else None,
            'detection_recall': tp / foreign if foreign else None,
            'detection_f1': 2*tp / (tp+fp+foreign) if tp+fp+foreign else None,
            'macro_auroc': float(np.mean([r['auroc'] for r in ranking])) if ranking else None,
            'macro_average_precision': float(np.mean([r['average_precision'] for r in ranking])) if ranking else None,
            'ranking_cases': len(ranking),
            'actual_dose_min': min(r.get('actual_dose', dose) for r in cases),
            'actual_dose_max': max(r.get('actual_dose', dose) for r in cases),
            'abstention_rate': float(np.mean([r['prediction'] is None for r in cases])),
            'q2_unavailable_fraction': float(np.mean([r.get('q2_unavailable_fraction', 0) for r in cases])),
            'host_removal_rate': float(np.mean([r['host_removal_rate'] for r in cases]))})
    return result


def evaluate(dataset_path, calibration_path, output, split='test'):
    if split not in ('development', 'test'):
        raise ValueError('Evaluate development or test, never calibration')
    data, cal = checked_calibration(dataset_path, calibration_path)
    cfg = cal['config']
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'protocol.json', {'calibration_sha256': sha256(calibration_path),
        'dataset_fingerprint': data['fingerprint'], 'source_hashes': source_hash(), 'config': cfg,
        'split': split, 'opened_at': datetime.now(timezone.utc).isoformat(), 'conventions': CONVENTIONS,
        'endpoint': 'Organ classification, not disease diagnosis',
        'primary': 'Combined versus no_filter correct-to-incorrect rate at each dose, with host-removal harm reported',
        'note': 'A new directory is not a fresh holdout. Do not retune after inspecting test results.'})
    begin = time.perf_counter()
    first = next(iter(data['models'].values()))
    pairs = pairs_for(first['slides'], split, cfg['seed'])
    if not pairs:
        raise ValueError('No host/donor pairs in requested split')
    write_json(out / 'pairs.json', pairs)
    example_pair_ids = set()
    for regime in sorted({p['regime'] for p in pairs}):
        example_pair_ids.update(list(dict.fromkeys(p['pair_id'] for p in pairs
                                                   if p['regime'] == regime))[:3])
    rows, controls, examples, skipped, mixtures, artifacts = [], [], [], [], [], []
    sanity = {}
    for name, model in data['models'].items():
        thresholds = cal['models'][name]['thresholds']
        split_slides = [s for s in model['slides'].values() if s['meta']['split'] == split]
        clean_cache = {}
        for s in split_slides:
            sid = s['meta']['slide_id']
            own = model['labels'].index(s['meta']['organ'])
            scores, signals, valid = core.foreignness(s['features'], s['xy_um'], model['prototypes'], own, cfg)
            clean_cache[sid] = scores, signals, valid
            artifacts.append(save_tiles(out, {'model': name, 'slide': sid, 'kind': 'original'},
                s['tile_ids'], [sid]*len(scores), s['xy_um'], scores, signals, valid,
                thresholds, model['labels']))
        for pooling in ('mean', 'topk'):
            confusion = np.zeros((len(model['labels']), len(model['labels'])), dtype=int)
            for s in split_slides:
                true = model['labels'].index(s['meta']['organ'])
                scores, signals, valid = clean_cache[s['meta']['slide_id']]
                pred = core.predict(scores, pooling, cfg['top_k'])
                confusion[true, pred] += 1
                for method in DETECTORS:
                    flags = signals[method] > thresholds[method]
                    filtered = core.predict(scores[~flags], pooling, cfg['top_k'])
                    controls.append({'model': name, 'slide': s['meta']['slide_id'], 'pooling': pooling,
                        'method': method, 'flagged_fraction': float(flags.mean()), 'any_flag': bool(flags.any()),
                        'clean_correct': pred == true, 'filtered_correct': filtered == true,
                        'new_error': pred == true and filtered != true,
                        'q2_unavailable_fraction': float((~valid).mean())})
            recalls = [float(confusion[i, i]/confusion[i].sum()) if confusion[i].sum() else None
                       for i in range(len(model['labels']))]
            sanity[f'{name}/{pooling}'] = {'labels': model['labels'], 'confusion_matrix': confusion.tolist(),
                'per_class_recall': recalls, 'balanced_accuracy': float(np.mean([r for r in recalls if r is not None])),
                'all_classes_present': all(r is not None for r in recalls)}
        # Validate both directions before evaluating either. If one donor cannot
        # supply the fragment, neither direction enters that pair/dose result.
        failed = set()
        for start in range(0, len(pairs), 2):
            directions = pairs[start:start+2]
            for dose in cfg['doses']:
                try:
                    reciprocal_bags(model['slides'], directions, dose)
                except ValueError as exc:
                    failed.add((directions[0]['pair_id'], dose))
                    for direction in directions:
                        skipped.append({'model': name, 'pair': direction, 'dose': dose,
                                        'reason': f'Reciprocal pair skipped: {exc}'})
        for pair_index, pair in enumerate(pairs):
            host, donor = model['slides'][pair['host']], model['slides'][pair['donor']]
            own = model['labels'].index(host['meta']['organ'])
            clean_scores = clean_cache[pair['host']][0]
            for dose in cfg['doses']:
                if (pair['pair_id'], dose) in failed:
                    continue
                bag = core.inject(host, donor, dose, pair['seed'])
                mixtures.append({'model': name, **pair, 'requested_dose': dose,
                    'actual_dose': bag['actual_fraction'],
                    'host_tile_count': len(host['features']),
                    'donor_tile_ids': donor['tile_ids'][bag['donor_indices']].tolist(),
                    'inserted_xy_um': bag['xy_um'][bag['is_foreign']].tolist()})
                if dose == 0:
                    scores, signals, valid = clean_cache[pair['host']]
                else:
                    scores, signals, valid = core.foreignness(bag['features'], bag['xy_um'], model['prototypes'], own, cfg)
                truth = bag['is_foreign']
                artifacts.append(save_tiles(out,
                    {'model': name, 'kind': 'mixture', **pair, 'requested_dose': dose,
                     'actual_dose': bag['actual_fraction']},
                    np.concatenate([host['tile_ids'], donor['tile_ids'][bag['donor_indices']]]),
                    [pair['host']]*len(host['features']) + [pair['donor']]*int(truth.sum()),
                    bag['xy_um'], scores, signals, valid, thresholds, model['labels'], truth))
                ranks = {method: core.ranking_metrics(values, truth) for method, values in signals.items()}
                masks = {method: signals[method] > thresholds[method] for method in DETECTORS}
                masks['no_filter'] = np.zeros(len(truth), dtype=bool)
                masks['oracle'] = truth.copy()
                random_mask = np.zeros(len(truth), dtype=bool)
                rng = np.random.default_rng(pair['seed'] + int(dose * 1_000_000))
                random_mask[rng.choice(len(truth), int(masks['combined'].sum()), replace=False)] = True
                masks['random_matched'] = random_mask
                for pooling in ('mean', 'topk'):
                    clean_pred = core.predict(clean_scores, pooling, cfg['top_k'])
                    contaminated = core.predict(scores, pooling, cfg['top_k'])
                    for method, flags in masks.items():
                        rows.append({'model': name, 'pair_id': pair['pair_id'],
                            'host': pair['host'], 'donor': pair['donor'],
                            'host_patient': host['meta']['patient_id'], 'donor_patient': donor['meta']['patient_id'],
                            'regime': pair['regime'], 'pooling': pooling, 'method': method,
                            'dose': dose, 'actual_dose': bag['actual_fraction'], 'true_class': own,
                            'clean_prediction': clean_pred, 'contaminated_prediction': contaminated,
                            'prediction': core.predict(scores[~flags], pooling, cfg['top_k']),
                            'host_tiles': len(host['features']), 'foreign_tiles': int(truth.sum()),
                            'q2_unavailable_fraction': float((~valid).mean()),
                            **ranks.get(method, {'auroc': None, 'average_precision': None}),
                            **core.detection_metrics(flags, truth)})
                # One direction per unordered pair at each nonzero dose, capped
                # for portable HTML. Complete per-tile arrays are always exported.
                if dose > 0 and pair_index % 2 == 0 and pair['pair_id'] in example_pair_ids:
                    examples.append({'model': name, 'host': pair['host'], 'donor': pair['donor'],
                        'regime': pair['regime'], 'pair_id': pair['pair_id'], 'requested_dose': dose,
                        'dose': bag['actual_fraction'], 'xy_um': bag['xy_um'].tolist(),
                        'tile_size_um': host['meta']['tile_size_um'], 'scores': signals['combined'].tolist(),
                        'flags': {m: masks[m].tolist() for m in DETECTORS},
                        'tile_ids': np.concatenate([host['tile_ids'], donor['tile_ids'][bag['donor_indices']]]).tolist(),
                        'signal_scores': {m: v.tolist() for m, v in signals.items()},
                        'flagged': masks['combined'].tolist(), 'inserted': truth.tolist(),
                        'note': 'Coordinate schematic; not a tissue image or clinical annotation'})
    if not rows:
        raise ValueError('All simulation cases failed; inspect cohort/fragment sizes')
    with (out / 'cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'split': split, 'runtime_seconds': time.perf_counter()-begin,
        'source_patients': len({s['meta']['patient_id'] for s in first['slides'].values()
                                if s['meta']['split'] == split}),
        'unordered_source_pairs': len(pairs) // 2, 'directed_recipes_per_dose': len(pairs),
        'successful_mixtures_by_model': {name: sum(m['model'] == name and m['requested_dose'] > 0
                                                  for m in mixtures) for name in data['models']},
        'data_provenance': sorted({s['meta']['provenance'] for m in data['models'].values() for s in m['slides'].values()}),
        'sanity': sanity, 'curves': aggregate(rows), 'unmodified_controls': controls,
        'skipped_cases': skipped, 'case_rows': len(rows),
        'limitations': ['Synthetic insertions do not validate real floater detection.',
            'Organ classification is not cancer diagnosis; same-organ donors cannot change organ ground truth.',
            'Unmodified controls are not proven floater-free.',
            'Repeated donor-host mixtures are correlated; no patient confidence intervals computed.',
            'Within-slide normalization and fixed-radius exclusion can mask large or repeated contaminants.',
            'CPU exact neighbor search is quadratic in tile count; blocking reduces memory, not total work.']}
    write_json(out / 'summary.json', summary)
    with (out / 'curves.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary['curves'][0]))
        writer.writeheader()
        writer.writerows(summary['curves'])
    write_json(out / 'mixtures.json', mixtures)
    write_json(out / 'examples.json', examples)
    write_json(out / 'tile_artifacts.json', artifacts)
    return summary
