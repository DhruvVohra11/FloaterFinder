"""Embedding intake and the complete offline analysis workflow."""
from pathlib import Path
import json

from . import benchmark, data, report
from .detection import CONVENTIONS


def inspect_dataset(dataset_path):
    return describe_dataset(data.load_dataset(dataset_path))


def describe_dataset(dataset):
    result = {'schema_version': 1, 'dataset_fingerprint': dataset['fingerprint'],
              'conventions': CONVENTIONS, 'models': {}, 'warnings': []}
    for name, model in dataset['models'].items():
        splits = {}
        for split in sorted(data.SPLITS):
            slides = [s for s in model['slides'].values() if s['meta']['split'] == split]
            counts = {label: sum(s['meta']['organ'] == label for s in slides) for label in model['labels']}
            splits[split] = {'slides': len(slides),
                'patients': len({s['meta']['patient_id'] for s in slides}),
                'tiles': sum(len(s['features']) for s in slides), 'organ_slide_counts': counts}
            if not slides:
                result['warnings'].append(f'{name}: no {split} slides')
            elif not all(counts.values()):
                result['warnings'].append(f'{name}: some text-bank organs absent in {split}')
        result['models'][name] = {'dimensions': model['prototypes'].shape[1],
                                 'labels': model['labels'], 'splits': splits}
    return result


def preflight(dataset_path, config_path, split='development'):
    """Check the exact reciprocal recipe geometry, without running a detector."""
    return plan_geometry(data.load_dataset(dataset_path), data.load_config(config_path), split)


def plan_geometry(dataset, cfg, split):
    if split not in ('development', 'test'):
        raise ValueError('Preflight development or test, never calibration')
    audit = describe_dataset(dataset)
    first = next(iter(dataset['models'].values()))
    pairs = benchmark.pairs_for(first['slides'], split, cfg['seed'])
    errors, warnings, scenarios = [], list(audit['warnings']), []
    for name, model in audit['models'].items():
        if not model['splits']['calibration']['slides']:
            errors.append(f'{name}: no calibration slides')
        if model['splits'][split]['patients'] < 2:
            errors.append(f'{name}: fewer than two patients in {split}')
    if not pairs:
        errors.append(f'No eligible source pairs in {split}')
    for regime in ('same_organ', 'different_organ'):
        if not any(p['regime'] == regime for p in pairs):
            warnings.append(f'No {regime} pairs in {split}')
    # Cohorts and coordinates match across models by load_dataset validation.
    # Geometry therefore needs checking once, independent of embedding scores.
    for start in range(0, len(pairs), 2):
        directions = pairs[start:start+2]
        for dose in cfg['doses']:
            scenario = {'pair_id': directions[0]['pair_id'], 'regime': directions[0]['regime'],
                        'requested_dose': dose}
            try:
                bags = benchmark.reciprocal_bags(first['slides'], directions, dose)
                scenario.update(status='ready', directions=[{
                    'host': p['host'], 'donor': p['donor'],
                    'host_tiles': len(first['slides'][p['host']]['features']),
                    'donor_tiles': int(bag['is_foreign'].sum()), 'actual_dose': bag['actual_fraction']}
                    for p, bag in zip(directions, bags)])
            except ValueError as exc:
                scenario.update(status='skipped', reason=str(exc))
            scenarios.append(scenario)
    ready = [s for s in scenarios if s['status'] == 'ready' and s['requested_dose'] > 0]
    skipped = [s for s in scenarios if s['status'] == 'skipped']
    if not ready:
        errors.append('No feasible positive-dose reciprocal mixtures; fix geometry, cohort or dose settings')
    if skipped:
        warnings.append(f'{len(skipped)} reciprocal pair/dose scenarios will be skipped (both directions)')
    return {'status': 'blocked' if errors else 'ready', 'split': split, 'intake': audit,
            'config': cfg, 'conventions': CONVENTIONS, 'errors': errors, 'warnings': warnings,
            'unordered_pairs': len(pairs)//2, 'positive_directed_mixtures_per_model': 2*len(ready),
            'scenarios': scenarios, 'note': 'Geometry and integrity check only; no detector or classification results computed.'}


def run(dataset_path, output, config_path=None, calibration_path=None, split='development'):
    if split not in ('development', 'test'):
        raise ValueError('Use development or test split')
    if split == 'test' and calibration_path is None:
        raise ValueError('Test runs require --calibration from a frozen development workflow')
    if calibration_path is not None and config_path is not None:
        raise ValueError('Use either a config for new calibration or an existing calibration')
    print('Validating embedding cache and split coverage…', flush=True)
    audit = inspect_dataset(dataset_path)
    for name, model in audit['models'].items():
        if model['splits'][split]['patients'] < 2:
            raise ValueError(f'{name}: need at least two patients in {split}')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    data.write_json(out / 'intake.json', audit)
    try:
        if calibration_path is None:
            print('Checking reciprocal fragment feasibility…', flush=True)
            plan = preflight(dataset_path, config_path, split)
            data.write_json(out / 'preflight.json', plan)
            if plan['status'] != 'ready':
                raise ValueError('; '.join(plan['errors']))
            calibration_path = out / 'calibration.json'
            print('Calibrating thresholds on unmodified calibration slides…', flush=True)
            benchmark.calibrate(dataset_path, config_path, calibration_path)
        else:
            # Copy the frozen artifact so the run is self-contained.
            calibration = json.loads(Path(calibration_path).read_text())
            calibration_path = out / 'calibration.json'
            data.write_json(calibration_path, calibration)
        print(f'Evaluating reciprocal mixtures and original-slide controls ({split})…', flush=True)
        summary = benchmark.evaluate(dataset_path, calibration_path, out / split, split)
        print('Rendering report and exporting tables and tile scores…', flush=True)
        report.render(out / split, out / split / 'report.html')
        data.write_json(out / 'run.json', {'status': 'complete', 'split': split,
            'dataset_fingerprint': audit['dataset_fingerprint'], 'report': f'{split}/report.html',
            'case_rows': summary['case_rows'], 'skipped_scenarios': len(summary['skipped_cases'])})
    except Exception as exc:
        data.write_json(out / 'run.json', {'status': 'failed', 'error': str(exc), 'split': split})
        raise
    return out / split / 'report.html'
