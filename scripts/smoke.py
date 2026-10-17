"""Run the whole offline workflow. No network, no torch, no GPU."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
output = root / 'outputs' / ('smoke-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
output.mkdir(parents=True)
dataset = output / 'data/dataset.json'
commands = [
    ['demo-data', '--output', str(output / 'data')],
    ['validate', '--dataset', str(dataset)],
    ['calibrate', '--dataset', str(dataset), '--output', str(output / 'calibration.json')],
    ['evaluate', '--dataset', str(dataset), '--calibration', str(output / 'calibration.json'),
     '--split', 'test', '--output', str(output / 'test')],
]
for command in commands:
    subprocess.run([sys.executable, '-m', 'floaterfinder', *command], cwd=root, check=True)
print(f'Synthetic software demonstration only. Open {output / "test/report.html"}')
