"""Optional organizer API client. Importing this module never contacts a service."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import time
import numpy as np
from .core import unit
from .data import sha256, write_json


class ModelService:
    def __init__(self, endpoint, key, allow_inference=False, session=None):
        if session is None:
            import requests
            session = requests.Session()
        self.session, self.endpoint = session, endpoint.rstrip('/')
        self._key, self.allow_inference = key, allow_inference

    def embed(self, paths):
        if not self.allow_inference:
            raise RuntimeError('Remote inference is disabled: /embed uses server GPU compute')
        if not self._key:
            raise ValueError('Set the selected API key environment variable')
        if not 1 <= len(paths) <= 32:
            raise ValueError('Use batches of 1–32 images')
        payload = {'images': [base64.b64encode(Path(p).read_bytes()).decode('ascii') for p in paths]}
        response = self.session.post(self.endpoint + '/embed', json=payload,
            headers={'Authorization': 'Bearer ' + self._key}, timeout=(10, 180), allow_redirects=False)
        if response.status_code != 200:
            # Do not print request headers, credentials, or arbitrary server bodies.
            raise RuntimeError(f'Embedding service HTTP status {response.status_code}')
        values = np.asarray(response.json()['embeddings'], dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(paths):
            raise ValueError('Unexpected embedding response shape')
        return unit(values)


def export_manifest(args):
    if not args.allow_remote_inference:
        raise RuntimeError('Explicit --allow-remote-inference is required; this command uses remote GPUs')
    if args.model in ('conch', 'musk') and not args.hf_access_approved:
        raise ValueError('Organizer requires your own approved Hugging Face access for CONCH/MUSK')
    services = json.loads(Path(args.services).read_text())
    cfg = services[args.model]
    path = Path(args.tiles).resolve()
    manifest = json.loads(path.read_text())
    entries = manifest['slides']
    n = sum(len(s['tiles']) for s in entries)
    if n < 1 or n > 2000:
        raise ValueError('This shared-service pilot accepts 1–2000 tiles; use your own job for whole datasets')
    if not 1 <= args.batch_size <= 32:
        raise ValueError('Batch size must be 1–32')
    for slide in entries:
        if not slide['tiles']:
            raise ValueError('Empty slide')
        for tile in slide['tiles']:
            if not (path.parent / tile['path']).is_file():
                raise ValueError('A tile file is missing; no inference requests sent')
    key = os.environ.get(cfg['key_env'], '')
    if not key:
        raise ValueError(f'Set {cfg["key_env"]} in the environment; never put it in source control')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    client = ModelService(cfg['endpoint'], key, allow_inference=True)
    start, files = time.perf_counter(), []
    for index, slide in enumerate(entries):
        tiles = slide['tiles']
        batches = []
        for begin in range(0, len(tiles), args.batch_size):
            paths = [path.parent / t['path'] for t in tiles[begin:begin+args.batch_size]]
            batches.append(client.embed(paths))
        meta = {k: slide[k] for k in ('slide_id', 'patient_id', 'organ', 'split', 'tile_size_um', 'stride_um')}
        meta.update(model_id=args.model, encoder_fingerprint=args.encoder_fingerprint,
                    provenance='Real tile images via organizer /embed; identity of weights must be verified with organizer')
        file = out / f'slide-{index:04d}.npz'
        np.savez_compressed(file, features=np.concatenate(batches),
            xy_um=np.array([[t['x_um'], t['y_um']] for t in tiles]),
            tile_ids=np.array([t['tile_id'] for t in tiles]), metadata=json.dumps(meta),
            image_sha256=np.array([sha256(path.parent / t['path']) for t in tiles]))
        files.append(file.name)
    elapsed = time.perf_counter() - start
    write_json(out / 'export.json', {'model_id': args.model, 'slides': files,
        'tiles': n, 'wall_seconds': elapsed, 'end_to_end_tiles_per_second': n / elapsed,
        'encoder_fingerprint': args.encoder_fingerprint,
        'note': 'Includes network/queue time. No server GPU memory telemetry is available. Matching text bank still required.'})
    return out / 'export.json'


def make_text_bank(input_path, output):
    if Path(output).exists():
        raise FileExistsError('Text bank already exists')
    with np.load(input_path, allow_pickle=False) as values:
        prompts = values['prompt_features']
        if prompts.ndim != 3 or prompts.shape[0] != len(values['labels']) or prompts.shape[1] < 1:
            raise ValueError('Expected prompt_features [classes, paraphrases, dimensions]')
        prototypes = unit(unit(prompts).mean(axis=1))
        np.savez_compressed(output, prototypes=prototypes, labels=values['labels'],
            model_id=values['model_id'], encoder_fingerprint=values['encoder_fingerprint'],
            prompt_source_sha256=sha256(input_path))
