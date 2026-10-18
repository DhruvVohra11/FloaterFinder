import csv
import json
from pathlib import Path

import numpy as np
import pytest

from floaterfinder import benchmark, core, data, detection, report, workflow

CONFIG = Path(__file__).parents[1] / 'configs/cpu_pilot.json'


@pytest.fixture
def cache(tmp_path):
    return data.demo_data(tmp_path / 'cache')


def test_dose_denominator_and_integer_boundary():
    # 7% of 100 final tiles: 93 host + 7 donor. Binary float ceil used to give 8.
    assert core.donor_count(93, .07) == 7
    assert core.donor_count(100, .1) == 12
    assert core.donor_count(93, 0) == 0
    assert core.donor_count(1, .005) == 1
    for value in (-.1, 1, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            core.donor_count(100, value)


def test_class_specific_topk_and_small_bag_clamping():
    scores = np.array([[.9, 0], [0, .8], [0, .8]])
    assert core.predict(scores, 'topk', 1) == 0
    assert core.predict(scores, 'topk', 2) == 1
    assert core.predict(scores, 'topk', 100) == core.predict(scores, 'mean', 1)


def test_large_finite_embeddings_still_normalize():
    np.testing.assert_allclose(core.unit(np.array([[1e30, 1e30]])), [[2**-.5, 2**-.5]])


@pytest.mark.parametrize('scores,truth,auc,ap', [
    ([4, 3, 2, 1], [1, 1, 0, 0], 1., 1.),
    ([4, 3, 2, 1], [0, 0, 1, 1], 0., (1/3 + 2/4)/2),
    ([1, 1, 1, 1], [0, 1, 0, 1], .5, .5),
    ([3, 2, 2, 1], [1, 0, 1, 0], .875, (1 + 2/3)/2),
])
def test_ranking_metrics_with_ties(scores, truth, auc, ap):
    result = core.ranking_metrics(scores, truth)
    assert result['auroc'] == pytest.approx(auc)
    assert result['average_precision'] == pytest.approx(ap)


def test_rank_undefined_when_only_one_class():
    assert core.ranking_metrics([.2, .9], [False, False]) == {'auroc': None, 'average_precision': None}


def test_exact_knn_against_independent_brute_force():
    rng = np.random.default_rng(14)
    x, xy = core.unit(rng.normal(size=(21, 6))), rng.uniform(0, 1000, (21, 2))
    values, available = core.far_knn(x, xy, 600, 4, 3)
    for i in range(len(x)):
        eligible = [j for j in range(len(x)) if i != j and np.linalg.norm(xy[i]-xy[j]) > 600]
        assert available[i] == (len(eligible) >= 4)
        if available[i]:
            distances = sorted(1-float(x[i] @ x[j]) for j in eligible)
            assert values[i] == pytest.approx(np.mean(distances[:4]), abs=1e-6)


def test_complete_development_then_frozen_test_workflow(cache, tmp_path):
    out = tmp_path / 'development-run'
    page = workflow.run(cache, out, CONFIG)
    assert page.exists()
    assert not (out / 'test').exists()
    assert json.loads((out / 'run.json').read_text())['status'] == 'complete'
    audit = json.loads((out / 'intake.json').read_text())
    assert audit['models']['synthetic-encoder-1']['splits']['test']['patients'] == 6
    artifacts = json.loads((out / 'development/tile_artifacts.json').read_text())
    assert len({a['path'] for a in artifacts}) == len(artifacts)
    mixture = next(a for a in artifacts if a['kind'] == 'mixture' and a['requested_dose'] > 0)
    tiles = data.npz(out / 'development' / mixture['path'])
    cal = json.loads((out / 'calibration.json').read_text())
    np.testing.assert_array_equal(tiles['combined_flagged'],
        tiles['combined_score'] > cal['models'][mixture['model']]['thresholds']['combined'])
    assert int(tiles['is_inserted'].sum()) > 0
    assert len(set(tiles['source_slide_ids'])) == 2
    summary = json.loads((out / 'development/summary.json').read_text())
    for row in summary['curves']:
        if row['dose'] > 0 and row['method'] == 'combined':
            assert 0 <= row['macro_auroc'] <= 1
            assert 0 <= row['macro_average_precision'] <= 1
            assert row['ranking_cases'] == row['cases']
    with pytest.raises(FileExistsError):
        workflow.run(cache, out, CONFIG)
    with pytest.raises(ValueError, match='frozen'):
        workflow.run(cache, tmp_path / 'unfrozen', CONFIG, split='test')
    test_page = workflow.run(cache, tmp_path / 'final-run', calibration_path=out / 'calibration.json', split='test')
    assert test_page.exists()


def test_original_slide_detection_has_no_insertion_truth(cache, tmp_path):
    cal = tmp_path / 'nested/calibration.json'
    benchmark.calibrate(cache, CONFIG, cal)
    path = detection.detect(cache, cal, tmp_path / 'candidates')
    with path.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 * 6 * 144
    assert {r['model'] for r in rows} == {'synthetic-encoder-1', 'synthetic-encoder-2'}
    index = json.loads((path.parent / 'tile_artifacts.json').read_text())
    assert len(index) == 12
    assert all('is_inserted' not in data.npz(path.parent / a['path']) for a in index)
    cache.write_text(cache.read_text() + ' ')
    with pytest.raises(ValueError, match='changed since calibration'):
        detection.detect(cache, cal, tmp_path / 'stale')


def test_failed_run_is_marked_failed(cache, tmp_path):
    with pytest.raises(FileNotFoundError):
        workflow.run(cache, tmp_path / 'failed', config_path=tmp_path / 'missing-config.json')
    assert json.loads((tmp_path / 'failed/run.json').read_text())['status'] == 'failed'


def test_report_cannot_execute_metadata_as_html(cache, tmp_path):
    out = tmp_path / 'run'
    page = workflow.run(cache, out, CONFIG)
    summary_path = page.parent / 'summary.json'
    summary = json.loads(summary_path.read_text())
    malicious = '</script><script>alert("untrusted metadata")</script>'
    summary['data_provenance'] = [malicious]
    data.write_json(summary_path, summary)
    report.render(page.parent, page)
    content = page.read_text()
    assert malicious not in content
    assert '\\u003c/script\\u003e' in content
    assert 'https://cdn' not in content
