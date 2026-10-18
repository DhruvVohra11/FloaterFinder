"""Write and assemble embedding caches without model inference or invented metadata."""
import json
from pathlib import Path
import numpy as np

from . import data


class CacheWriter:
    """Streaming exporter for upstream encoders. Publish a manifest only after validation.

    Use as a context manager; successful exit validates and finishes the cache.
    Failed exports retain their files plus a failed receipt, without dataset.json.
    """

    def __init__(self, output):
        self.root = Path(output)
        self.root.mkdir(parents=True, exist_ok=False)
        self.models = {}
        self.closed = False
        data.write_json(self.root / 'export-status.json', {'status': 'writing'})

    def _open(self):
        if self.closed:
            raise ValueError('CacheWriter is already closed')

    def add_model(self, model_id, encoder_fingerprint, labels, prototypes):
        self._open()
        data.string_scalar(model_id, 'model_id')
        data.string_scalar(encoder_fingerprint, 'encoder_fingerprint')
        data.string_vector(labels, 'labels')
        if model_id in self.models:
            raise ValueError(f'Duplicate text bank for model {model_id}')
        folder = f'model-{len(self.models):03d}'
        (self.root / folder).mkdir()
        bank = f'{folder}/text.npz'
        np.savez_compressed(self.root / bank, model_id=model_id,
            encoder_fingerprint=encoder_fingerprint, labels=labels, prototypes=prototypes)
        self.models[model_id] = {'model_id': model_id, 'text_bank': bank, 'slides': []}

    def add_slide(self, features, xy_um, tile_ids, metadata, image_sha256=None):
        self._open()
        model_id = metadata.get('model_id')
        if model_id not in self.models:
            raise ValueError('Register the matching text bank with add_model first')
        data.string_vector(tile_ids, 'tile_ids')
        model = self.models[model_id]
        filename = str(Path(model['text_bank']).parent / f'slide-{len(model["slides"]):06d}.npz')
        arrays = dict(features=features, xy_um=xy_um, tile_ids=tile_ids,
                      metadata=json.dumps(metadata, allow_nan=False))
        if image_sha256 is not None:
            arrays['image_sha256'] = image_sha256
        np.savez_compressed(self.root / filename, **arrays)
        model['slides'].append(filename)

    def finish(self):
        self._open()
        candidate = self.root / '.dataset.pending.json'
        try:
            data.write_json(candidate, {'schema_version': 1, 'models': list(self.models.values())})
            data.load_dataset(candidate)
            candidate.rename(self.root / 'dataset.json')
            data.write_json(self.root / 'export-status.json', {'status': 'complete',
                'models': len(self.models), 'slides_by_model': {m: len(v['slides']) for m, v in self.models.items()}})
            self.closed = True
        except Exception as exc:
            self._fail(exc)
            raise
        return self.root / 'dataset.json'

    def _fail(self, exc):
        self.closed = True
        (self.root / '.dataset.pending.json').unlink(missing_ok=True)
        data.write_json(self.root / 'export-status.json', {'status': 'failed', 'error': str(exc)})

    def __enter__(self):
        return self

    def __exit__(self, kind, exc, traceback):
        if exc is not None:
            if not self.closed:
                self._fail(exc)
        elif not self.closed:
            self.finish()
        return False


def assemble(input_dir, output):
    """Combine a dedicated drop of contract-format NPZ files into one portable cache.

    Text banks are identified by prototypes/labels; slides by features/metadata.
    Unknown NPZ files are rejected, so detection artifacts cannot masquerade as
    source embeddings. Only one text bank per model is allowed.
    """
    source, target = Path(input_dir).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError('Input must be an embedding-drop directory')
    if target == source or source in target.parents:
        raise ValueError('Output must be outside the input drop directory')
    banks, slides = [], []
    for path in sorted(source.rglob('*.npz')):
        try:
            with np.load(path, allow_pickle=False) as arrays:
                keys = set(arrays.files)
            if {'prototypes', 'labels', 'model_id', 'encoder_fingerprint'} <= keys:
                banks.append(path)
            elif {'features', 'xy_um', 'tile_ids', 'metadata'} <= keys:
                slides.append(path)
            else:
                raise ValueError('Expected a slide cache or completed text bank')
        except Exception as exc:
            raise ValueError(f'{path.relative_to(source)}: {exc}') from exc
    if not banks or not slides:
        raise ValueError('Need at least one text bank and one slide NPZ')
    with CacheWriter(target) as writer:
        for path in banks:
            values = data.npz(path)
            writer.add_model(data.string_scalar(values['model_id'], 'model_id'),
                data.string_scalar(values['encoder_fingerprint'], 'encoder_fingerprint'),
                values['labels'], values['prototypes'])
        for path in slides:
            values = data.npz(path)
            metadata = json.loads(data.string_scalar(values['metadata'], 'metadata'))
            writer.add_slide(values['features'], values['xy_um'], values['tile_ids'],
                             metadata, values.get('image_sha256'))
    data.write_json(target / 'assembly.json', {'input': str(source),
        'source_sha256': {str(p.relative_to(source)): data.sha256(p) for p in banks + slides},
        'note': 'No embeddings generated; no metadata inferred. NPZ arrays repackaged after validation.'})
    return target / 'dataset.json'
