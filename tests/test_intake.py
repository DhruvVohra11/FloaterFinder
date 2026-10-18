import json
from pathlib import Path

import numpy as np
import pytest

from floaterfinder import benchmark, data, workflow
from floaterfinder.intake import CacheWriter, assemble

CONFIG = Path(__file__).parents[1] / 'configs/cpu_pilot.json'


@pytest.fixture
def cache(tmp_path):
    return data.demo_data(tmp_path / 'source')


def test_assemble_reconstructs_models_without_changing_vectors(cache, tmp_path):
    path = assemble(cache.parent, tmp_path / 'assembled')
    original, packaged = data.load_dataset(cache), data.load_dataset(path)
    assert set(original['models']) == set(packaged['models'])
    for name, model in original['models'].items():
        other = packaged['models'][name]
        np.testing.assert_array_equal(model['prototypes'], other['prototypes'])
        for sid, slide in model['slides'].items():
            for key in ('features', 'xy_um', 'tile_ids'):
                np.testing.assert_array_equal(slide[key], other['slides'][sid][key])
            assert slide['meta'] == other['slides'][sid]['meta']
    manifest = json.loads(path.read_text())
    assert all(not Path(p).is_absolute() for m in manifest['models'] for p in m['slides'])
    assert json.loads((path.parent / 'export-status.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        assemble(cache.parent, path.parent)


def test_unknown_npz_rejected_without_publishing(cache, tmp_path):
    np.savez(cache.parent / 'detector-output.npz', combined_score=[.5])
    with pytest.raises(ValueError, match='detector-output.npz'):
        assemble(cache.parent, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_duplicate_bank_and_nested_output_rejected(cache, tmp_path):
    with pytest.raises(ValueError, match='outside'):
        assemble(cache.parent, cache.parent / 'assembled')
    bank = cache.parent / 'synthetic-encoder-1/text.npz'
    (bank.parent / 'duplicate-text.npz').write_bytes(bank.read_bytes())
    with pytest.raises(ValueError, match='Duplicate text bank'):
        assemble(cache.parent, tmp_path / 'output')
    assert not (tmp_path / 'output/dataset.json').exists()
    assert json.loads((tmp_path / 'output/export-status.json').read_text())['status'] == 'failed'


def test_writer_never_publishes_partial_or_leaky_cache(tmp_path):
    out = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='Patient crosses'):
        with CacheWriter(out) as writer:
            writer.add_model('model/unsafe-path', 'checkpoint', ['a', 'b'], np.eye(2))
            for split in ('development', 'test'):
                writer.add_slide(np.eye(2), [[0, 0], [100, 0]], [f'{split}/0', f'{split}/1'],
                    {'slide_id': split, 'patient_id': 'same-patient', 'split': split,
                     'organ': 'a', 'model_id': 'model/unsafe-path', 'encoder_fingerprint': 'checkpoint',
                     'tile_size_um': 100, 'stride_um': 100, 'provenance': 'SYNTHETIC test'})
    assert not (out / 'dataset.json').exists()
    assert not (out / '.dataset.pending.json').exists()
    assert (out / 'model-000/text.npz').exists()


@pytest.mark.parametrize('field,value,message', [
    ('provenance', '', 'provenance'), ('patient_id', 123, 'patient_id'),
    ('encoder_fingerprint', '', 'encoder_fingerprint'),
])
def test_invalid_metadata_rejected(cache, field, value, message):
    path = cache.parent / 'synthetic-encoder-1/development-prostate-0.npz'
    values = data.npz(path)
    meta = json.loads(str(values['metadata']))
    meta[field] = value
    values['metadata'] = json.dumps(meta)
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match=message):
        data.load_dataset(cache)


def test_numeric_tile_ids_rejected(cache):
    path = cache.parent / 'synthetic-encoder-1/development-prostate-0.npz'
    values = data.npz(path)
    values['tile_ids'] = np.arange(1, len(values['features'])+1)
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match='tile_ids'):
        data.load_dataset(cache)


def test_preflight_matches_real_evaluation_skip_plan(cache, tmp_path):
    # A disconnected but valid coordinate grid cannot supply the larger doses.
    for model in ('synthetic-encoder-1', 'synthetic-encoder-2'):
        path = cache.parent / model / 'development-prostate-0.npz'
        values = data.npz(path)
        values['xy_um'] = np.column_stack([np.arange(144)*112*10, np.zeros(144)])
        np.savez_compressed(path, **values)
    plan = workflow.preflight(cache, CONFIG)
    assert plan['status'] == 'ready'
    assert any(s['status'] == 'skipped' for s in plan['scenarios'])
    cal = tmp_path / 'calibration.json'
    benchmark.calibrate(cache, CONFIG, cal)
    summary = benchmark.evaluate(cache, cal, tmp_path / 'result', 'development')
    actual = {(s['pair']['pair_id'], s['dose']) for s in summary['skipped_cases']}
    expected = {(s['pair_id'], s['requested_dose']) for s in plan['scenarios'] if s['status'] == 'skipped'}
    assert expected == actual


def test_preflight_blocks_all_positive_doses_before_calibration(cache, tmp_path):
    for model in ('synthetic-encoder-1', 'synthetic-encoder-2'):
        for path in (cache.parent / model).glob('development-*.npz'):
            values = data.npz(path)
            values['xy_um'][0, 0] += 13  # off the recorded stride grid
            np.savez_compressed(path, **values)
    plan = workflow.preflight(cache, CONFIG)
    assert plan['status'] == 'blocked'
    assert plan['positive_directed_mixtures_per_model'] == 0
    with pytest.raises(ValueError, match='No feasible'):
        workflow.run(cache, tmp_path / 'run', CONFIG)
    assert not (tmp_path / 'run/calibration.json').exists()
    assert json.loads((tmp_path / 'run/run.json').read_text())['status'] == 'failed'
