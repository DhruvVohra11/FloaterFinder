import argparse
from pathlib import Path
from . import benchmark, data, report


def main():
    p = argparse.ArgumentParser(description='CPU-first floater stress test and candidate detector')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('demo-data')
    q.add_argument('--output', required=True)
    q.set_defaults(fn=lambda a: data.demo_data(a.output))
    q = sub.add_parser('validate')
    q.add_argument('--dataset', required=True)
    q.set_defaults(fn=lambda a: f"Validated {len(data.load_dataset(a.dataset)['models'])} model caches")
    q = sub.add_parser('calibrate')
    q.add_argument('--dataset', required=True)
    q.add_argument('--config', default='configs/cpu_pilot.json')
    q.add_argument('--output', required=True)
    def calibration(a):
        benchmark.calibrate(a.dataset, a.config, a.output)
        return f'Calibration saved: {a.output}'
    q.set_defaults(fn=calibration)
    q = sub.add_parser('evaluate')
    q.add_argument('--dataset', required=True)
    q.add_argument('--calibration', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--split', choices=['development','test'], default='development')
    def evaluation(a):
        benchmark.evaluate(a.dataset, a.calibration, a.output, a.split)
        report.render(a.output, Path(a.output) / 'report.html')
        return f'Report saved: {a.output}/report.html'
    q.set_defaults(fn=evaluation)
    q = sub.add_parser('report')
    q.add_argument('--run', required=True)
    q.add_argument('--output', required=True)
    q.set_defaults(fn=lambda a: report.render(a.run, a.output))
    q = sub.add_parser('make-text-bank')
    q.add_argument('--input', required=True)
    q.add_argument('--output', required=True)
    def text_bank(a):
        from .service import make_text_bank
        make_text_bank(a.input, a.output)
        return a.output
    q.set_defaults(fn=text_bank)
    q = sub.add_parser('service-export', help='Optional future Delta command; consumes remote GPU compute')
    q.add_argument('--model', choices=['conch','musk','quiltnet','plip'], required=True)
    q.add_argument('--services', default='configs/services.json')
    q.add_argument('--tiles', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--encoder-fingerprint', required=True)
    q.add_argument('--batch-size', type=int, default=16)
    q.add_argument('--allow-remote-inference', action='store_true')
    q.add_argument('--hf-access-approved', action='store_true')
    def export(a):
        from .service import export_manifest
        return export_manifest(a)
    q.set_defaults(fn=export)
    args = p.parse_args()
    print(args.fn(args))
