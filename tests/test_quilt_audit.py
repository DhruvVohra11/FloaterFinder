import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('quilt_audit', Path(__file__).parents[1] / 'scripts/audit_quilt.py')
quilt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quilt)


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'shared'
    image_dir = root / 'images_part_3/quilt_1m/subfolder'
    image_dir.mkdir(parents=True)
    (image_dir / 'a.jpg').write_bytes(b'not decoded by metadata audit')
    with (root / 'quilt_1M_lookup-002.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_path', 'caption', 'pathology', 'magnification', 'irrelevant'])
        writer.writeheader()
        for i in range(40):
            writer.writerow(dict(image_path='quilt_1m/subfolder/a.jpg', caption=f'line {i}\nsecond line',
                                 pathology='example', magnification='20x', irrelevant='do not retain'))
    return root


def test_audit_is_bounded_reproducible_and_does_not_invent_patient_metadata(source, tmp_path):
    result = quilt.audit(source, tmp_path/'out1', max_rows=20, sample_size=5)
    quilt.audit(source, tmp_path/'out2', max_rows=20, sample_size=5)
    assert result['records_examined'] == 20 and not result['reached_end_of_csv']
    assert result['sample_records'] == 5 and result['sample_unique_resolved_images'] == 1
    assert not result['wsi_benchmark_ready']
    assert 'No recognized patient column' in result['metadata_gaps']
    assert result['gpu_requests'] == result['image_copies'] == 0
    a = (tmp_path/'out1/review-sample.json').read_text()
    assert a == (tmp_path/'out2/review-sample.json').read_text()
    assert 'irrelevant' not in a
    assert all(s['path_status'] == 'resolved' for s in json.loads(a))
    assert len(list(source.rglob('*'))) == 5


def test_audit_rejects_shared_output_and_overwrites(source, tmp_path):
    with pytest.raises(ValueError, match='outside'):
        quilt.audit(source, source/'results')
    quilt.audit(source, tmp_path/'out', 100, 3)
    with pytest.raises(FileExistsError):
        quilt.audit(source, tmp_path/'out')
    assert json.loads((tmp_path/'out/audit.json').read_text())['reached_end_of_csv']


def test_resolver_rejects_ambiguous_escaping_and_missing_paths(source, tmp_path):
    duplicate = source / 'images_part_4/quilt_1m/subfolder/a.jpg'
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b'duplicate')
    assert quilt.resolve_image(source, 'quilt_1m/subfolder/a.jpg')[1] == 'ambiguous_path'
    assert quilt.resolve_image(source, '../a.jpg')[1] == 'unsafe_path'
    assert quilt.resolve_image(source, str(tmp_path/'outside.jpg'))[1] == 'outside_dataset'
    assert quilt.resolve_image(source, 'missing.jpg')[1] == 'not_found'
    assert quilt.resolve_image(source, 'images_part_3/quilt_1m/subfolder/a.jpg')[1] == 'resolved'
