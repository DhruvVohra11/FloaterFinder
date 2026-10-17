import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from floaterfinder import core, data, benchmark, report
from floaterfinder.service import ModelService, make_text_bank, export_manifest


@pytest.fixture
def dataset(tmp_path):
    return data.demo_data(tmp_path / 'data')


def cfg():
    return data.load_config(Path(__file__).parents[1] / 'configs/cpu_pilot.json')


def test_projection_normalization_and_pooling():
    scores = core.cosine_scores(np.array([[2.,0.],[0.,4.]]), np.eye(2))
    np.testing.assert_allclose(scores, np.eye(2))
    x = np.array([[.9,.1],[.1,.6],[.1,.6]])
    assert core.predict(x, 'mean', 1) == 1
    assert core.predict(x, 'topk', 1) == 0
    assert core.predict(x[:0], 'mean', 1) is None


def test_floater_cannot_vouch_for_itself():
    x = np.array([[1.,0.]]*4 + [[0.,1.]]*2)
    xy = np.array([[0,0],[112,0],[224,0],[336,0],[2000,0],[2112,0]],float)
    near, _ = core.far_knn(x, xy, 0, 1, 2)
    far, valid = core.far_knn(x, xy, 500, 1, 2)
    np.testing.assert_allclose(near[-2:], 0)
    np.testing.assert_allclose(far[-2:], 1)
    assert valid.all()


def test_blocked_neighbors_match_full_computation():
    rng = np.random.default_rng(1)
    x, xy = rng.normal(size=(25,8)), rng.uniform(0,1000,(25,2))
    a, va = core.far_knn(x, xy, 200, 3, 4)
    b, vb = core.far_knn(x, xy, 200, 3, 25)
    np.testing.assert_allclose(a,b,atol=1e-6)
    np.testing.assert_array_equal(va,vb)


def test_insufficient_neighbors_do_not_produce_nan():
    x = np.eye(3)
    values, valid = core.far_knn(x, np.zeros((3,2)), 1000, 10, 2)
    assert not valid.any()
    np.testing.assert_array_equal(core.zscore(values,valid),0)
    np.testing.assert_array_equal(core.zscore(np.ones(4)),0)


def test_smoothing_uses_geometry():
    result = core.smooth(np.array([3.,0.,9.]), np.array([[0,0],[1,0],[10,0]]), 2, 2)
    np.testing.assert_array_equal(result, [1.5,1.5,9])


def test_injection_preserves_host_and_oracle_restores(dataset):
    m = next(iter(data.load_dataset(dataset)['models'].values()))
    h, d = m['slides']['test-prostate-0'], m['slides']['test-colon-0']
    bag = core.inject(h,d,.1,7)
    n = len(h['features'])
    np.testing.assert_array_equal(bag['features'][:n],h['features'])
    np.testing.assert_array_equal(bag['xy_um'][:n],h['xy_um'])
    assert bag['actual_fraction'] >= .1
    assert (bag['xy_um'][n:,0] > h['xy_um'][:,0].max()).all()
    clean = core.cosine_scores(h['features'],m['prototypes'])
    mixed = core.cosine_scores(bag['features'],m['prototypes'])
    for pooling in ('mean','topk'):
        assert core.predict(clean,pooling,10) == core.predict(mixed[~bag['is_foreign']],pooling,10)
    np.testing.assert_array_equal(bag['donor_indices'],core.inject(h,d,.1,7)['donor_indices'])


def test_same_patient_injection_is_rejected(dataset):
    m = next(iter(data.load_dataset(dataset)['models'].values()))
    h = next(iter(m['slides'].values()))
    with pytest.raises(ValueError,match='different patients'):
        core.inject(h,h,.01,7)


def test_pair_plan_is_reciprocal_deterministic_and_split_safe(dataset):
    slides = next(iter(data.load_dataset(dataset)['models'].values()))['slides']
    pairs = benchmark.pairs_for(slides, 'test', 7)
    assert pairs == benchmark.pairs_for(dict(reversed(list(slides.items()))), 'test', 7)
    assert len({(p['host'], p['donor']) for p in pairs}) == len(pairs)
    assert {p['regime'] for p in pairs} == {'same_organ', 'different_organ'}
    grouped = {}
    for p in pairs:
        grouped.setdefault(p['pair_id'], []).append(p)
        h, d = slides[p['host']]['meta'], slides[p['donor']]['meta']
        assert h['split'] == d['split'] == 'test'
        assert h['patient_id'] != d['patient_id']
        assert (h['organ'] == d['organ']) == (p['regime'] == 'same_organ')
    for directions in grouped.values():
        assert len(directions) == 2
        a, b = directions
        assert (a['host'], a['donor']) == (b['donor'], b['host'])


def test_reciprocal_mixtures_use_unchanged_originals(dataset):
    slides = next(iter(data.load_dataset(dataset)['models'].values()))['slides']
    directions = benchmark.pairs_for(slides, 'test', 7)[:2]
    originals = {p['host']: {key: slides[p['host']][key].copy()
                            for key in ('features', 'xy_um', 'tile_ids')} for p in directions}
    bags = benchmark.reciprocal_bags(slides, directions, .1)
    for p, bag in zip(directions, bags):
        host, donor = originals[p['host']], originals[p['donor']]
        np.testing.assert_array_equal(bag['features'][~bag['is_foreign']], host['features'])
        np.testing.assert_array_equal(bag['features'][bag['is_foreign']],
                                      donor['features'][bag['donor_indices']])
        for key in host:
            np.testing.assert_array_equal(slides[p['host']][key], host[key])


def test_failed_reverse_mixture_skips_both_directions(dataset, tmp_path, monkeypatch):
    config = Path(__file__).parents[1] / 'configs/cpu_pilot.json'
    cal = tmp_path / 'cal.json'
    benchmark.calibrate(dataset, config, cal)
    slides = next(iter(data.load_dataset(dataset)['models'].values()))['slides']
    directions = benchmark.pairs_for(slides, 'test', 7)[:2]
    reverse = directions[1]
    original = core.inject
    def fail_reverse(host, donor, dose, seed):
        if (host['meta']['slide_id'], donor['meta']['slide_id']) == (reverse['host'], reverse['donor']) and dose > 0:
            raise ValueError('No connected donor component is large enough')
        return original(host, donor, dose, seed)
    monkeypatch.setattr(core, 'inject', fail_reverse)
    out = tmp_path / 'result'
    summary = benchmark.evaluate(dataset, cal, out)
    skipped = summary['skipped_cases']
    assert skipped and len(skipped) % 2 == 0
    mixtures = json.loads((out / 'mixtures.json').read_text())
    assert not any(m['pair_id'] == reverse['pair_id'] and m['requested_dose'] > 0 for m in mixtures)


def test_disconnected_fragment_is_not_silently_joined():
    with pytest.raises(ValueError,match='connected'):
        core.connected_fragment(np.array([[0,0],[1000,0]]),100,2,7)


def test_patient_split_leakage_rejected(dataset):
    path = dataset.parent / 'synthetic-encoder-1/test-prostate-0.npz'
    values = data.npz(path)
    meta = json.loads(str(values['metadata']))
    meta['patient_id'] = 'patient-calibration-prostate-0'
    values['metadata'] = json.dumps(meta)
    np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match='Patient crosses'):
        data.load_dataset(dataset)


def test_mixed_embedding_spaces_rejected(dataset):
    path = dataset.parent / 'synthetic-encoder-1/text.npz'
    values = data.npz(path)
    values['encoder_fingerprint'] = 'wrong-checkpoint'
    np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match='fingerprints'):
        data.load_dataset(dataset)


def test_duplicate_image_content_rejected(dataset):
    path = dataset.parent / 'synthetic-encoder-1/test-prostate-0.npz'
    values = data.npz(path)
    values['image_sha256'] = np.array(['duplicate'] * len(values['features']))
    np.savez_compressed(path,**values)
    with pytest.raises(ValueError,match='Duplicate source image'):
        data.load_dataset(dataset)


def test_model_cohorts_must_match(dataset):
    manifest = json.loads(dataset.read_text())
    manifest['models'][1]['slides'].pop()
    data.write_json(dataset,manifest)
    with pytest.raises(ValueError,match='identical slide cohorts'):
        data.load_dataset(dataset)


def test_calibration_and_full_evaluation(dataset,tmp_path):
    config = Path(__file__).parents[1] / 'configs/cpu_pilot.json'
    cal = tmp_path / 'cal.json'
    c = benchmark.calibrate(dataset,config,cal)
    assert all(m['calibration_slides']==6 for m in c['models'].values())
    assert all(v <= .011 for m in c['models'].values() for v in m['calibration_flagged_tile_fraction'].values())
    out = tmp_path / 'result'
    summary = benchmark.evaluate(dataset,cal,out,'test')
    assert summary['case_rows'] > 0
    assert not summary['skipped_cases']
    mixtures = json.loads((out / 'mixtures.json').read_text())
    grouped = {}
    for m in mixtures:
        grouped.setdefault((m['model'], m['pair_id'], m['requested_dose']), []).append(m)
        assert len(m['donor_tile_ids']) == len(m['inserted_xy_um'])
        assert all(tile.startswith(m['donor'] + '-tile-') for tile in m['donor_tile_ids'])
    assert all(len(directions) == 2 for directions in grouped.values())
    assert summary['directed_recipes_per_dose'] == 2 * summary['unordered_source_pairs']
    assert summary['source_patients'] == 6
    for row in summary['curves']:
        if row['method']=='oracle':
            assert row['prediction_change_rate']==0
            assert row['host_removal_rate']==0
        if row['method']=='no_filter' and row['dose']==0:
            assert row['prediction_change_rate']==0
    report.render(out,out/'report.html')
    content = (out/'report.html').read_text()
    assert 'SYNTHETIC' in content and 'Unmodified-slide harm' in content
    with pytest.raises(FileExistsError):
        benchmark.evaluate(dataset,cal,out,'test')


def test_cache_change_invalidates_calibration(dataset,tmp_path):
    cal = tmp_path/'cal.json'
    benchmark.calibrate(dataset,Path(__file__).parents[1]/'configs/cpu_pilot.json',cal)
    dataset.write_text(dataset.read_text()+' ')
    with pytest.raises(ValueError,match='changed since calibration'):
        benchmark.evaluate(dataset,cal,tmp_path/'result')


def test_recovery_denominator_only_counts_corrupted_correct_cases():
    common = {'model':'x','regime':'different','pooling':'mean','method':'combined','dose':.1,
              'true_class':0,'true_positive_tiles':1,'false_positive_tiles':0,
              'foreign_tiles':1,'host_removal_rate':0}
    rows = [dict(common,clean_prediction=0,contaminated_prediction=1,prediction=0),
            dict(common,clean_prediction=1,contaminated_prediction=1,prediction=1)]
    r = benchmark.aggregate(rows)[0]
    assert r['recovery_rate']==1 and r['initially_correct_cases']==1
    assert r['accuracy']==.5


def test_remote_gpu_call_blocked_by_default():
    class Forbidden:
        def post(self,*a,**kw):
            raise AssertionError('Network must not be called')
    client = ModelService('http://example.invalid','placeholder',session=Forbidden())
    with pytest.raises(RuntimeError,match='disabled'):
        client.embed(['does-not-exist.png'])


def test_service_contract_with_mock_only(tmp_path):
    image = tmp_path/'tile.png'
    image.write_bytes(b'fake bytes for mocked transport')
    class Session:
        def post(self,url,**kw):
            assert url.endswith('/embed')
            assert kw['headers']['Authorization']=='Bearer test-placeholder'
            assert not kw['allow_redirects']
            return SimpleNamespace(status_code=200,json=lambda:{'embeddings':[[3.,4.]]})
    client = ModelService('http://example.invalid','test-placeholder',True,Session())
    np.testing.assert_allclose(client.embed([image]),[[.6,.8]])


def test_gated_export_requires_access_before_any_network():
    with pytest.raises(ValueError,match='approved Hugging Face'):
        export_manifest(SimpleNamespace(allow_remote_inference=True,model='conch',hf_access_approved=False))


def test_text_bank_averages_normalized_prompts(tmp_path):
    path = tmp_path/'prompts.npz'
    x = np.array([[[10.,0.],[0.,1.]], [[0.,2.],[-1.,0.]]])
    np.savez_compressed(path,prompt_features=x,labels=['a','b'],model_id='mock',encoder_fingerprint='mock1')
    make_text_bank(path,tmp_path/'bank.npz')
    np.testing.assert_allclose(data.npz(tmp_path/'bank.npz')['prototypes'],
                               core.unit(np.array([[1.,1.],[-1.,1.]])))


def test_gdc_planner_selects_distinct_patients():
    import importlib.util
    path = Path(__file__).parents[1]/'scripts/plan_gdc_cohort.py'
    spec = importlib.util.spec_from_file_location('cohort_planner',path)
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    hits = [{'file_id':str(i),'access':'open','cases':[{'case_id':str(i//2)}]} for i in range(20)]
    selection = planner.select_cases(hits,6,7)
    assert len({s['cases'][0]['case_id'] for s in selection}) == 6
    assert selection == planner.select_cases(hits,6,7)


def test_random_removal_matches_combined_budget(dataset,tmp_path):
    import csv
    cal = tmp_path/'cal.json'
    benchmark.calibrate(dataset,Path(__file__).parents[1]/'configs/cpu_pilot.json',cal)
    out = tmp_path/'results'
    benchmark.evaluate(dataset,cal,out,'development')
    with (out/'cases.csv').open() as f:
        rows = list(csv.DictReader(f))
    counts = {}
    for r in rows:
        if r['method'] in ('combined','random_matched'):
            key = tuple(r[k] for k in ('model','host','donor','dose','pooling'))
            counts.setdefault(key,{})[r['method']] = r['flagged_tiles']
    assert all(v['combined']==v['random_matched'] for v in counts.values())
