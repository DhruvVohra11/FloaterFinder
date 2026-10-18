from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from importlib.resources import files
import numpy as np
from .core import unit

SPLITS = {'development', 'calibration', 'test'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def npz(path):
    with np.load(path, allow_pickle=False) as obj:
        return {k: obj[k] for k in obj.files}


def string_scalar(value, name):
    value = np.asarray(value)
    if value.shape != () or value.dtype.kind != 'U' or not str(value).strip():
        raise ValueError(f'{name} must be a nonempty scalar string')
    return str(value)


def string_vector(value, name):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind != 'U' or not all(v.strip() for v in value.tolist()):
        raise ValueError(f'{name} must be a vector of nonempty strings')
    return value.tolist()


def load_dataset(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('models'):
        raise ValueError('Expected schema_version=1 and nonempty models')
    patients, tile_owners, canonical, models, hashes, image_owners = {}, {}, {}, {}, {}, {}
    hashes[str(path)] = sha256(path)
    expected_slides = None
    for entry in manifest['models']:
        model_id = entry['model_id']
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError('model_id must be a nonempty string')
        if model_id in models:
            raise ValueError('Duplicate model entry')
        bank_path = (path.parent / entry['text_bank']).resolve()
        bank = npz(bank_path)
        hashes[str(bank_path)] = sha256(bank_path)
        labels = string_vector(bank['labels'], 'labels')
        if len(labels) < 2 or len(set(labels)) != len(labels) or not all(labels):
            raise ValueError('Need unique nonempty organ labels')
        proto = unit(bank['prototypes'])
        if proto.ndim != 2 or len(proto) != len(labels):
            raise ValueError('Text bank shape mismatch')
        if string_scalar(bank['model_id'], 'model_id') != model_id:
            raise ValueError('Text bank model mismatch')
        fingerprint = string_scalar(bank['encoder_fingerprint'], 'encoder_fingerprint')
        slides = {}
        for rel in entry['slides']:
            file = (path.parent / rel).resolve()
            slide = npz(file)
            hashes[str(file)] = sha256(file)
            meta = json.loads(string_scalar(slide['metadata'], 'metadata'))
            required = {'slide_id', 'patient_id', 'organ', 'split', 'model_id',
                        'encoder_fingerprint', 'tile_size_um', 'stride_um', 'provenance'}
            if not required.issubset(meta):
                raise ValueError('Missing slide metadata')
            for key in required - {'tile_size_um', 'stride_um'}:
                if not isinstance(meta[key], str) or not meta[key].strip():
                    raise ValueError(f'{key} must be a nonempty string')
            sid, patient = meta['slide_id'], meta['patient_id']
            if not sid or not patient or sid in slides or meta['split'] not in SPLITS:
                raise ValueError('Invalid slide/patient/split identity')
            if meta['organ'] not in labels or meta['model_id'] != model_id:
                raise ValueError('Slide organ/model mismatch')
            if meta['encoder_fingerprint'] != fingerprint:
                raise ValueError('Image/text encoder fingerprints do not match')
            for key in ('tile_size_um', 'stride_um'):
                if not np.isfinite(meta[key]) or meta[key] <= 0:
                    raise ValueError('Invalid physical scale')
            x, xy, ids = slide['features'], slide['xy_um'], slide['tile_ids']
            string_vector(ids, 'tile_ids')
            if x.ndim != 2 or len(x) < 2 or x.shape[1] != proto.shape[1]:
                raise ValueError('Feature dimensions mismatch or insufficient tiles')
            if xy.shape != (len(x), 2) or not np.isfinite(xy).all():
                raise ValueError('Coordinates must be finite [N,2] microns')
            if len(np.unique(xy, axis=0)) != len(x):
                raise ValueError('Duplicate tile coordinates')
            if ids.shape != (len(x),) or len(set(ids.tolist())) != len(x) or not all(ids):
                raise ValueError('Invalid tile IDs')
            if patient in patients and patients[patient] != meta['split']:
                raise ValueError('Patient crosses data splits')
            patients[patient] = meta['split']
            for tid in ids.tolist():
                if tid in tile_owners and tile_owners[tid] != sid:
                    raise ValueError('Tile ID reused across slides')
                tile_owners[tid] = sid
            if 'image_sha256' in slide:
                image_hashes = slide['image_sha256']
                if image_hashes.shape != (len(x),) or not all(image_hashes):
                    raise ValueError('Invalid source-image hashes')
                for sha, tid in zip(image_hashes.tolist(), ids.tolist()):
                    if sha in image_owners and image_owners[sha] != (sid, tid):
                        raise ValueError('Duplicate source image used for different tiles')
                    image_owners[sha] = (sid, tid)
            identity = {k: meta[k] for k in ('patient_id', 'organ', 'split', 'tile_size_um', 'stride_um')}
            if sid in canonical:
                old = canonical[sid]
                if old[0] != identity or not np.array_equal(old[1], xy) or not np.array_equal(old[2], ids):
                    raise ValueError('Model caches must cover identical slides/tiles/coordinates')
            canonical[sid] = (identity, xy, ids)
            slides[sid] = {'features': unit(x), 'xy_um': xy.astype(float),
                           'tile_ids': ids, 'meta': meta,
                           'image_sha256': slide.get('image_sha256', np.array([], dtype=str))}
        if not slides:
            raise ValueError('No slides')
        if expected_slides is not None and set(slides) != expected_slides:
            raise ValueError('Models must use identical slide cohorts')
        expected_slides = set(slides)
        models[model_id] = {'labels': labels, 'prototypes': proto, 'slides': slides,
            'embedding_contract': {'encoder_fingerprint': fingerprint, 'labels': labels,
                'dimensions': proto.shape[1],
                'normalized_prototypes_sha256': hashlib.sha256(proto.tobytes()).hexdigest()}}
    # The same relative-path bundle can move between machines without changing
    # identity; changing any manifest/cache bytes still invalidates calibration.
    relative_hashes = {os.path.relpath(name, path.parent): value for name, value in hashes.items()}
    fingerprint = hashlib.sha256(json.dumps(relative_hashes, sort_keys=True).encode()).hexdigest()
    return {'models': models, 'fingerprint': fingerprint, 'files': hashes}


def load_config(path=None):
    source = files('floaterfinder').joinpath('default_config.json') if path is None else Path(path)
    cfg = json.loads(source.read_text())
    required = {'neighbors', 'exclusion_radius_um', 'smoothing_radius_um', 'visual_weight',
                'chunk_size', 'top_k', 'calibration_quantile', 'doses', 'seed'}
    if set(cfg) != required:
        raise ValueError(f'Expected config keys {sorted(required)}')
    for k in ('neighbors', 'chunk_size', 'top_k'):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if type(cfg['seed']) is not int or cfg['seed'] < 0:
        raise ValueError('seed must be nonnegative integer')
    for k in ('exclusion_radius_um', 'smoothing_radius_um', 'visual_weight'):
        if not np.isfinite(cfg[k]) or cfg[k] < 0:
            raise ValueError(f'Invalid {k}')
    if not 0 < cfg['calibration_quantile'] < 1:
        raise ValueError('Quantile must be between 0 and 1')
    if not cfg['doses'] or any(not 0 <= d < 1 for d in cfg['doses']):
        raise ValueError('Invalid doses')
    if cfg['doses'] != sorted(set(cfg['doses'])) or cfg['doses'][0] != 0:
        raise ValueError('Doses must be increasing and include zero')
    return cfg


def demo_data(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(82)
    labels = ['prostate', 'colon', 'lung']
    models = []
    grid = np.stack(np.meshgrid(np.arange(12), np.arange(12)), axis=-1).reshape(-1, 2) * 112.
    # One shared synthetic tissue fixture transformed into two distinct mock spaces.
    latent = {}
    for split in sorted(SPLITS):
        for organ, label in enumerate(labels):
            for patient in range(2):
                sid = f'{split}-{label}-{patient}'
                x = rng.normal(0, .12, (len(grid), 12))
                x[:, organ] += 1.0
                x[:, 4 + patient] += .15
                # A native unusual region: a detector must not assume unusual=foreign.
                x[:4, 8] += 1.2
                latent[sid] = (x, split, label)
    for index in range(2):
        model_id = f'synthetic-encoder-{index+1}'
        folder = root / model_id
        folder.mkdir()
        rotation, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        proto = np.eye(12)[:3] @ rotation
        np.savez_compressed(folder / 'text.npz', labels=np.array(labels), prototypes=unit(proto),
                            model_id=model_id, encoder_fingerprint=f'fixture-{index}')
        files = []
        for sid, (x, split, label) in latent.items():
            meta = {'slide_id': sid, 'patient_id': f'patient-{sid}', 'organ': label,
                    'split': split, 'model_id': model_id, 'encoder_fingerprint': f'fixture-{index}',
                    'tile_size_um': 112., 'stride_um': 112.,
                    'provenance': 'SYNTHETIC VECTORS AND COORDINATES — NO PATHOLOGY MODEL OR REAL TISSUE'}
            np.savez_compressed(folder / f'{sid}.npz', features=unit(x @ rotation), xy_um=grid,
                tile_ids=np.array([f'{sid}-tile-{i}' for i in range(len(grid))]), metadata=json.dumps(meta))
            files.append(f'{model_id}/{sid}.npz')
        models.append({'model_id': model_id, 'text_bank': f'{model_id}/text.npz', 'slides': files})
    write_json(root / 'dataset.json', {'schema_version': 1, 'models': models})
    return root / 'dataset.json'
