import argparse
from pathlib import Path
from . import benchmark, data, report


def main():
    p = argparse.ArgumentParser(description='CPU-first floater stress test and candidate detector')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('assemble', help='Validate and package a folder of slide/text NPZ exports')
    q.add_argument('--input', required=True)
    q.add_argument('--output', required=True)
    def assembly(a):
        from .intake import assemble
        return assemble(a.input, a.output)
    q.set_defaults(fn=assembly)
    q = sub.add_parser('preflight', help='Check reciprocal mixture feasibility without running kNN')
    q.add_argument('--dataset', required=True)
    q.add_argument('--config', default='configs/cpu_pilot.json')
    q.add_argument('--split', choices=['development', 'test'], default='development')
    q.add_argument('--output', required=True)
    def feasibility(a):
        import json
        from .workflow import preflight
        result = preflight(a.dataset, a.config, a.split)
        path = Path(a.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as f:
            f.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
        if result['status'] == 'blocked':
            p.exit(2, 'Preflight blocked: ' + '; '.join(result['errors']) + f' (see {path})\n')
        return f'Preflight ready: {result["positive_directed_mixtures_per_model"]} positive-dose mixtures per model; {path}'
    q.set_defaults(fn=feasibility)
    q = sub.add_parser('inspect', help='Validate teammate cache and summarize split/organ coverage')
    q.add_argument('--dataset', required=True)
    q.add_argument('--output')
    def inspection(a):
        import json
        from .workflow import inspect_dataset
        audit = inspect_dataset(a.dataset)
        if a.output:
            path = Path(a.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x') as f:
                f.write(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        return json.dumps(audit, indent=2)
    q.set_defaults(fn=inspection)
    q = sub.add_parser('run', help='Validate, calibrate, benchmark, export and visualize')
    q.add_argument('--dataset', required=True)
    q.add_argument('--output', required=True)
    choice = q.add_mutually_exclusive_group()
    choice.add_argument('--config')
    choice.add_argument('--calibration')
    q.add_argument('--split', choices=['development', 'test'], default='development')
    def workflow(a):
        from .workflow import run
        return run(a.dataset, a.output, a.config, a.calibration, a.split)
    q.set_defaults(fn=workflow)
    q = sub.add_parser('detect', help='Export calibrated per-tile candidates on original slides')
    q.add_argument('--dataset', required=True)
    q.add_argument('--calibration', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--split', choices=['development', 'test'], default='development')
    def detection(a):
        from .detection import detect
        return detect(a.dataset, a.calibration, a.output, a.split)
    q.set_defaults(fn=detection)
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
    q.add_argument('--static', action='store_true', help='Render the print-friendly report without JavaScript')
    q.set_defaults(fn=lambda a: (report.render_static if a.static else report.render)(a.run, a.output))
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
