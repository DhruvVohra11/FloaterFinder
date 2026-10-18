import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from floaterfinder import benchmark, core, data, detection, workflow, report

CONFIG = Path(__file__).parents[1] / 'configs/cpu_pilot.json'


def test_packaged_defaults_match_example_and_are_independent():
    assert data.load_config() == data.load_config(CONFIG)
    first = data.load_config()
    first['doses'].append(.9)
    assert .9 not in data.load_config()['doses']


def test_normalization_extreme_finite_scales_and_invalid_types():
    for scale in (1e-300, 1e300):
        np.testing.assert_allclose(core.unit([[scale, scale]]), [[2**-.5, 2**-.5]])
    for x in ([[0, 0]], [[complex(1, 2), 1]], [['1', '2']], np.empty((1, 0))):
        with pytest.raises(ValueError):
            core.unit(x)


@pytest.mark.parametrize('chunk', [7, 128, 2000])
def test_knn_crosses_reference_blocks_and_matches_dense_oracle(chunk):
    rng = np.random.default_rng(99)
    x = rng.normal(size=(1100, 12))
    xy = rng.uniform(0, 1000, (len(x), 2))
    actual, valid = core.far_knn(x, xy, 100, 7, chunk)
    unit = core.unit(x)
    scores = unit @ unit.T
    eligible = np.linalg.norm(xy[:, None]-xy[None, :], axis=2) > 100
    scores[~eligible] = -np.inf
    expected = (1-np.clip(np.sort(scores, axis=1)[:, -7:], -1, 1)).mean(axis=1)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    assert valid.all()


def test_shared_smoothing_matches_dense_multiple_signals_and_translation():
    rng = np.random.default_rng(17)
    xy = rng.uniform(0, 1000, (1050, 2))
    values = rng.normal(size=(len(xy), 2))
    neighbors = np.linalg.norm(xy[:, None]-xy[None, :], axis=2) <= 250
    expected = neighbors @ values / neighbors.sum(axis=1)[:, None]
    actual = core.smooth(values, xy, 250, 31)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(core.smooth(values, xy+1e6, 250, 31), actual, atol=1e-12)


def test_grid_drift_and_grid_collisions_rejected():
    with pytest.raises(ValueError, match='regular grid'):
        core.connected_fragment(np.array([[0., 0.], [100000.1, 0.]]), 1, 1, 7)
    with pytest.raises(ValueError, match='same grid'):
        core.connected_fragment(np.array([[0., 0.], [.0001, 0.]]), 1, 1, 7)


def new_cohort(source, destination):
    """New identities on synthetic vectors, only to exercise compatibility checks."""
    shutil.copytree(source.parent, destination)
    manifest = json.loads((destination / 'dataset.json').read_text())
    for model in manifest['models']:
        model['slides'] = [p for p in model['slides'] if '/test-' in p]
        for relative in model['slides']:
            file = destination / relative
            values = data.npz(file)
            metadata = json.loads(str(values['metadata']))
            for key in ('slide_id', 'patient_id'):
                metadata[key] = 'new-' + metadata[key]
            values['tile_ids'] = np.array(['new-'+s for s in values['tile_ids']])
            values['metadata'] = json.dumps(metadata)
            np.savez_compressed(file, **values)
    data.write_json(destination / 'dataset.json', manifest)
    return destination / 'dataset.json'


def test_portable_bundle_and_explicit_new_cohort_detection(tmp_path):
    source = data.demo_data(tmp_path / 'source')
    cal = tmp_path / 'cal.json'
    benchmark.calibrate(source, CONFIG, cal)
    shutil.copytree(source.parent, tmp_path / 'moved')
    relocated = tmp_path / 'moved/dataset.json'
    assert data.load_dataset(source)['fingerprint'] == data.load_dataset(relocated)['fingerprint']
    detection.checked_calibration(relocated, cal)
    incoming = new_cohort(source, tmp_path / 'incoming')
    with pytest.raises(ValueError, match='changed since calibration'):
        detection.detect(incoming, cal, tmp_path / 'strict', split='test')
    csv_path = detection.detect(incoming, cal, tmp_path / 'detected', split='test', new_cohort=True)
    assert csv_path.exists()
    receipt = json.loads((csv_path.parent / 'detection.json').read_text())
    assert receipt['new_cohort'] is True
    assert len(receipt['slides']) == 12
    with pytest.raises(ValueError, match='overlaps'):
        detection.checked_calibration(source, cal, new_cohort=True)


@pytest.mark.parametrize('change,reason', [('text', 'contract'), ('scale', 'physical scale'),
                                         ('patient', 'overlaps'), ('tiles', 'overlaps')])
def test_new_cohort_rejects_incompatible_or_leaked_exports(tmp_path, change, reason):
    source = data.demo_data(tmp_path / 'source')
    cal = tmp_path / 'cal.json'
    benchmark.calibrate(source, CONFIG, cal)
    incoming = new_cohort(source, tmp_path / 'incoming')
    # Modify both model caches where geometry/identity must stay aligned.
    for model in ('synthetic-encoder-1', 'synthetic-encoder-2'):
        file = incoming.parent / model / ('text.npz' if change == 'text' else 'test-prostate-0.npz')
        arrays = data.npz(file)
        if change == 'text':
            arrays['prototypes'][0, 0] += .1
        else:
            meta = json.loads(str(arrays['metadata']))
            if change == 'scale':
                meta['tile_size_um'] = 224.
            elif change == 'patient':
                meta['patient_id'] = 'patient-calibration-prostate-0'
            else:
                arrays['tile_ids'] = np.array([s.removeprefix('new-') for s in arrays['tile_ids']])
            arrays['metadata'] = json.dumps(meta)
        np.savez_compressed(file, **arrays)
    with pytest.raises(ValueError, match=reason):
        detection.checked_calibration(incoming, cal, new_cohort=True)


def test_controlled_corruption_recovery_is_measured_without_tuning(tmp_path):
    path = data.demo_data(tmp_path / 'fixture')
    manifest = json.loads(path.read_text())
    manifest['models'] = manifest['models'][:1]
    model = manifest['models'][0]
    bank_path = path.parent / model['text_bank']
    bank = data.npz(bank_path)
    bank['prototypes'] = np.eye(3)
    np.savez_compressed(bank_path, **bank)
    # Intentionally separable engineering fixture: a strong colon donor can
    # overturn weak prostate top-K predictions. This is not clinical evidence.
    vectors = {'prostate': [.8, .6, 0], 'colon': [0, 1, 0], 'lung': [0, 0, 1]}
    for relative in model['slides']:
        file = path.parent / relative
        slide = data.npz(file)
        meta = json.loads(str(slide['metadata']))
        slide['features'] = np.tile(vectors[meta['organ']], (len(slide['tile_ids']), 1))
        np.savez_compressed(file, **slide)
    data.write_json(path, manifest)
    out = tmp_path / 'run'
    workflow.run(path, out, CONFIG)
    summary = json.loads((out / 'development/summary.json').read_text())
    supported = [r for r in summary['curves'] if r['method'] == 'combined' and r['corrupted_correct_cases'] > 0]
    assert supported, 'Fixture must actually produce eligible recovery cases'
    assert all(r['recovery_rate'] == 1 for r in supported)
    # The combined detector can harm native tissue despite perfect recovery.
    # Keep this counterexample: the report must expose the tradeoff, not claim
    # that restored classification implies correct localization.
    assert any(r['host_removal_rate'] > 0 for r in supported)
    organ_only = [r for r in summary['curves'] if r['method'] == 'organ_only' and r['corrupted_correct_cases'] > 0]
    assert all(r['host_removal_rate'] == 0 for r in organ_only)
    assert all(r['prediction_change_rate'] == 0 for r in summary['curves'] if r['method'] == 'oracle')


def test_report_support_diagnostics_are_not_success_claims():
    row = dict(model='x', pooling='topk', regime='same_organ', method='no_filter', dose=.1,
               corrupted_correct_cases=0, initially_correct_cases=3)
    messages = benchmark.report_diagnostics([row], [])
    assert any('recovery is unmeasured' in m for m in messages)


def test_static_plot_does_not_bridge_unmeasured_doses():
    rows = [dict(method='combined', dose=d, error_rate_initially_correct=.5) for d in [0, .01, .1]]
    svg = report.curve_svg(rows, [0, .01, .05, .1])
    assert svg.count('<polyline') == 2


def test_bounded_map_overviews_preserve_full_tile_exports(tmp_path, monkeypatch):
    path = data.demo_data(tmp_path / 'cache')
    monkeypatch.setattr(benchmark, 'MAX_MAP_TILES', 7)
    out = tmp_path / 'run'
    workflow.run(path, out, CONFIG)
    root = out / 'development'
    examples = json.loads((root / 'examples.json').read_text())
    artifacts = json.loads((root / 'tile_artifacts.json').read_text())
    for example in examples:
        assert len(example['xy_um']) == example['displayed_tiles'] == 7
        item = next(a for a in artifacts if a['kind'] == 'mixture' and a['model'] == example['model']
                    and a['host'] == example['host'] and a['donor'] == example['donor']
                    and a['requested_dose'] == example['requested_dose'])
        full = data.npz(root / item['path'])
        assert len(full['tile_ids']) == example['total_tiles'] > 7
        for method in ('organ_only', 'visual_only', 'combined'):
            assert example['flagged_counts'][method] == int(full[method+'_flagged'].sum())
