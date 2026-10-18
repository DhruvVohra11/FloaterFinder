"""Calibrated candidate detection and portable per-tile outputs."""
from pathlib import Path
import hashlib
import json
import csv
import numpy as np

from . import core
from .data import load_dataset, write_json


CONVENTIONS = {
    'version': 1,
    'dose_basis': 'donor_tiles / (host_tiles + donor_tiles)',
    'donor_count': 'ceil(requested_fraction * host_tiles / (1 - requested_fraction))',
    'host_relative_conversion': 'final_fraction = host_relative_ratio / (1 + host_relative_ratio)',
    'pooling': 'mean cosine or per-class top-K mean cosine; K=min(config.top_k, remaining tiles)',
    'ties': 'first class in text-bank label order',
    'empty_bag': 'abstain; counts as incorrect',
    'threshold': 'strictly greater than calibration quantile after smoothing',
    'reciprocity': 'both directions from originals; skip both if either fails',
    'ranking': 'per-bag tie-aware AUROC and average precision; macro average over valid bags',
}


def checked_calibration(dataset_path, calibration_path):
    from .benchmark import source_hash
    data = load_dataset(dataset_path)
    cal = json.loads(Path(calibration_path).read_text())
    if data['fingerprint'] != cal['dataset_fingerprint']:
        raise ValueError('Dataset changed since calibration')
    if source_hash() != cal['source_hashes']:
        raise ValueError('Core implementation changed; recalibration required')
    return data, cal


def save_tiles(out, identity, tile_ids, source_slides, xy, scores, signals, valid,
               thresholds, labels, truth=None):
    """Names use a digest, never untrusted source identifiers as paths."""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = Path(out) / 'tiles'
    folder.mkdir(exist_ok=True)
    arrays = {'tile_ids': np.asarray(tile_ids), 'source_slide_ids': np.asarray(source_slides),
              'xy_um': xy, 'class_scores': scores, 'labels': np.asarray(labels),
              'q2_available': valid}
    for method, values in signals.items():
        arrays[method + '_score'] = values
        arrays[method + '_flagged'] = values > thresholds[method]
    if truth is not None:
        arrays['is_inserted'] = truth
    np.savez_compressed(folder / (key + '.npz'), **arrays)
    return {**identity, 'path': f'tiles/{key}.npz', 'tiles': len(tile_ids)}


def detect(dataset_path, calibration_path, output, split='development'):
    """Score original slides without insertion labels or donor references."""
    if split not in ('development', 'test'):
        raise ValueError('Detect development or test slides, never calibration')
    data, cal = checked_calibration(dataset_path, calibration_path)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    index, summary = [], []
    with (out / 'candidates.csv').open('w', newline='') as file:
        columns = ['model', 'slide', 'patient', 'tile_id', 'x_um', 'y_um', 'q2_available']
        for method in ('organ_only', 'visual_only', 'combined'):
            columns.extend([method + '_score', method + '_flagged'])
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for name, model in data['models'].items():
            thresholds = cal['models'][name]['thresholds']
            for sid, slide in sorted(model['slides'].items()):
                meta = slide['meta']
                if meta['split'] != split:
                    continue
                scores, signals, valid = core.foreignness(slide['features'], slide['xy_um'],
                    model['prototypes'], model['labels'].index(meta['organ']), cal['config'])
                index.append(save_tiles(out, {'model': name, 'slide': sid, 'kind': 'original'},
                    slide['tile_ids'], [sid] * len(scores), slide['xy_um'], scores, signals,
                    valid, thresholds, model['labels']))
                for i, tid in enumerate(slide['tile_ids']):
                    row = dict(model=name, slide=sid, patient=meta['patient_id'], tile_id=tid,
                               x_um=slide['xy_um'][i, 0], y_um=slide['xy_um'][i, 1], q2_available=bool(valid[i]))
                    for method, values in signals.items():
                        row[method + '_score'] = float(values[i])
                        row[method + '_flagged'] = bool(values[i] > thresholds[method])
                    writer.writerow(row)
                summary.append({'model': name, 'slide': sid, 'tiles': len(scores),
                    'q2_unavailable_fraction': float((~valid).mean()),
                    'flagged': {m: int((v > thresholds[m]).sum()) for m, v in signals.items()}})
    if not index:
        raise ValueError('No slides in requested split')
    write_json(out / 'tile_artifacts.json', index)
    write_json(out / 'detection.json', {'split': split, 'conventions': CONVENTIONS,
        'dataset_fingerprint': data['fingerprint'], 'calibration': cal, 'slides': summary,
        'interpretation': 'Candidates for review; unknown real contamination status, no precision/recall claimed.'})
    return out / 'candidates.csv'
