"""Read-only, bounded-memory Quilt metadata audit. No inference or image copies."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random


DEFAULT_ROOT = '/work/nvme/bfqi/shared/datasets/quilt'
FIELDS = {
    'image': ('image_path', 'image', 'filename', 'file_name'),
    'caption': ('caption',),
    'pathology': ('pathology',),
    'magnification': ('magnification',),
    'patient': ('patient_id', 'case_id', 'patient'),
    'slide': ('slide_id', 'wsi_id', 'slide'),
    'x_um': ('x_um', 'tile_x_um'),
    'y_um': ('y_um', 'tile_y_um'),
    'x_pixel': ('x', 'tile_x', 'x_coord'),
    'y_pixel': ('y', 'tile_y', 'y_coord'),
    'mpp': ('mpp', 'microns_per_pixel'),
    'tile_size_um': ('tile_size_um',),
    'stride_um': ('stride_um',),
    'organ': ('organ', 'organ_label', 'tissue_site'),
    'source': ('source', 'dataset', 'source_dataset'),
    'video': ('video_id', 'video_url', 'youtube_id'),
}


def resolve_image(root, value):
    """Check only documented layouts. Never guess among duplicate basenames."""
    root = Path(root).resolve()
    value = str(value).strip()
    if not value:
        return None, 'missing_path'
    path = Path(value)
    if '..' in path.parts:
        return None, 'unsafe_path'
    candidates = []
    if path.is_absolute():
        if not path.is_relative_to(root):
            return None, 'outside_dataset'
        candidates.append(path)
    else:
        candidates.append(root / path)
        parts = path.parts
        if parts and parts[0].startswith('images_part_'):
            relative = None  # An explicit part number disambiguates the source.
        elif 'quilt_1m' in parts:
            relative = Path(*parts[parts.index('quilt_1m') + 1:])
        elif parts:
            relative = path
        else:
            relative = None
        if relative is not None:
            candidates.extend(root / f'images_part_{i}' / 'quilt_1m' / relative for i in range(1, 11))
    found = {p.resolve() for p in candidates if p.is_file() and p.resolve().is_relative_to(root)}
    if len(found) == 1:
        return str(found.pop()), 'resolved'
    return None, 'ambiguous_path' if found else 'not_found'


def audit(root, output, max_rows=10000, sample_size=120, seed=7):
    root, out = Path(root).resolve(), Path(output).resolve()
    if out.is_relative_to(root):
        raise ValueError('Outputs must be outside the shared dataset')
    if out.exists():
        raise FileExistsError('Use a new output directory')
    if max_rows < 1 or not 1 <= sample_size <= 2000:
        raise ValueError('Use a positive row limit and 1–2000 review samples')
    source = root / 'quilt_1M_lookup-002.csv'
    rng, samples = random.Random(seed), []
    csv.field_size_limit(16 * 1024 * 1024)
    # Streaming parses one CSV record at a time, retains only selected columns
    # in a bounded reservoir, and performs filesystem probes for that sample.
    with source.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        by_lower = {h.strip().lower(): h for h in headers}
        mapping = {name: [by_lower[a] for a in aliases if a in by_lower]
                   for name, aliases in FIELDS.items()}
        if not mapping['image']:
            raise ValueError(f'No recognized image path column. Headers: {headers}')
        selected = sorted({field for fields in mapping.values() for field in fields})
        nonempty = {field: 0 for field in selected}
        examined, complete = 0, False
        for index in range(max_rows):
            row = next(reader, None)
            if row is None:
                complete = True
                break
            examined += 1
            kept = {field: (row.get(field) or '').strip() for field in selected}
            for field, value in kept.items():
                nonempty[field] += bool(value)
            entry = {'csv_record': index + 1, 'metadata': kept}
            if len(samples) < sample_size:
                samples.append(entry)
            else:
                position = rng.randrange(examined)
                if position < sample_size:
                    samples[position] = entry
    for entry in samples:
        value = entry['metadata'][mapping['image'][0]]
        entry['resolved_path'], entry['path_status'] = resolve_image(root, value)
        entry['review_status'] = 'unreviewed'
    gaps = []
    for role in ('patient', 'slide', 'organ', 'tile_size_um', 'stride_um'):
        if not mapping[role]:
            gaps.append(f'No recognized {role} column')
    physical = bool(mapping['x_um'] and mapping['y_um'])
    convertible = bool(mapping['x_pixel'] and mapping['y_pixel'] and mapping['mpp'])
    if not (physical or convertible):
        gaps.append('No recognized physical coordinates or pixel coordinates with microns-per-pixel')
    if convertible and not physical:
        gaps.append('Pixel coordinate level/convention must be verified before conversion')
    for field, count in nonempty.items():
        if count < examined and any(field in mapping[r] for r in ('patient', 'slide')):
            gaps.append(f'{field} missing in {examined-count} examined records')
    statuses = {status: sum(s['path_status'] == status for s in samples)
                for status in sorted({s['path_status'] for s in samples})}
    paths = [s['resolved_path'] for s in samples if s['resolved_path']]
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset_root': str(root), 'metadata_file': str(source),
        'metadata_size_bytes': source.stat().st_size,
        'columns': headers, 'recognized_fields': mapping,
        'records_examined': examined, 'reached_end_of_csv': complete,
        'nonempty_counts': nonempty, 'sample_records': len(samples),
        'sample_unique_resolved_images': len(set(paths)), 'sample_path_statuses': statuses,
        'sampling': f'Seed {seed}; reservoir over first at most {max_rows} records, not a full-dataset random sample',
        'wsi_benchmark_ready': False,
        'metadata_gaps': gaps,
        'remaining_checks': [
            'Column presence does not verify patient identity, organ labels, or coordinate semantics.',
            'Patient/slide linkage must come from source metadata, never filename or caption guesses.',
            'Magnification text alone does not establish physical scale.',
            'Review images for tissue content, overlays, existing contamination and near duplicates.',
            'Video/source grouping helps limit leakage but is not patient-separated validation.',
            'QuiltNet was trained on Quilt; a Quilt run is not an independent QuiltNet benchmark.',
        ],
        'gpu_requests': 0, 'image_copies': 0,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'review-sample.json').write_text(json.dumps(samples, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-rows', type=int, default=10000)
    parser.add_argument('--sample-size', type=int, default=120)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    result = audit(args.root, args.output, args.max_rows, args.sample_size, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
