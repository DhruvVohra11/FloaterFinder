"""Compare current exact detection against a recorded repository revision.

Run from the repository root, e.g.:
python scripts/profile_detector.py --baseline-ref a9bde03 --output outputs/profile.json
Only local committed source is loaded; no model or network access is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from floaterfinder import core, data


def profile(baseline_ref, output):
    source = subprocess.check_output(['git', 'show', f'{baseline_ref}:floaterfinder/core.py'], cwd=ROOT, text=True)
    old = types.ModuleType('baseline_core')
    exec(compile(source, 'baseline_core.py', 'exec'), old.__dict__)
    cfg = data.load_config(ROOT / 'configs/cpu_pilot.json')
    rows = []
    for n, dimensions in ((512, 128), (2048, 128), (4096, 256)):
        rng = np.random.default_rng(19)
        features = rng.normal(size=(n, dimensions))
        xy = rng.uniform(0, 10000, size=(n, 2))
        prototypes = rng.normal(size=(3, dimensions))
        times, results = {}, {}
        for name, implementation in [('baseline', old), ('current', core)]:
            elapsed = []
            for repeat in range(4):
                start = time.perf_counter()
                result = implementation.foreignness(features, xy, prototypes, 0, cfg)
                if repeat:  # one warmup, then three measured runs
                    elapsed.append(time.perf_counter()-start)
            times[name] = float(np.median(elapsed))
            results[name] = result
        errors = {m: float(np.max(np.abs(results['baseline'][1][m]-results['current'][1][m])))
                  for m in results['current'][1]}
        np.testing.assert_array_equal(results['baseline'][2], results['current'][2])
        for method in errors:
            np.testing.assert_allclose(results['baseline'][1][method], results['current'][1][method], atol=2e-5)
        b = cfg['chunk_size']
        rows.append({'tiles': n, 'dimensions': dimensions, 'median_seconds': times,
            'speedup': times['baseline']/times['current'], 'max_absolute_signal_errors': errors,
            'baseline_max_pairwise_entries': min(b, n)*n,
            'current_max_pairwise_entries': min(b, n)*min(max(b, 1024), n),
            'note': 'Pairwise entries describe one matrix, not peak process memory. Search remains quadratic.'})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {'baseline_ref': baseline_ref,
              'baseline_core_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'current_core_sha256': data.sha256(ROOT / 'floaterfinder/core.py'),
              'numpy': np.__version__, 'config': cfg, 'measurements': rows}
    with path.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(profile(args.baseline_ref, args.output), indent=2))
