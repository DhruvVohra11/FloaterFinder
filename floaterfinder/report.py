from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np

COLORS = {'no_filter':'#bd4545','organ_only':'#9c6b17','visual_only':'#7d5eb2',
          'combined':'#007d78','random_matched':'#738391','oracle':'#2866c3'}


def esc(value):
    return html.escape(str(value))


def curve_svg(rows, doses=None):
    doses = doses or sorted({r['dose'] for r in rows})
    maxdose = max(.01, max(doses))
    svg = ['<svg viewBox="0 0 550 270" role="img" aria-label="Errors after contamination and filtering">',
           '<path d="M50 20V220H520" fill="none" stroke="#a1acb5"/>']
    for tick in (0, .25, .5, .75, 1):
        y = 220 - 200*tick
        svg.append(f'<text x="8" y="{y+4}">{tick:.0%}</text>')
    for tick in (0, maxdose/2, maxdose):
        x = 50 + 470*tick/maxdose
        svg.append(f'<text x="{x-10}" y="239">{tick:.1%}</text>')
    for method, color in COLORS.items():
        points = []
        by_dose = {r['dose']: r for r in rows if r['method'] == method}
        for dose in doses:
            r = by_dose.get(dose)
            if r is not None and r['error_rate_initially_correct'] is not None:
                points.append(f"{50+470*r['dose']/maxdose:.2f},{220-200*r['error_rate_initially_correct']:.2f}")
            elif points:
                svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
                points = []
        if points:
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
    svg.append('<text x="170" y="262">Requested foreign-tile fraction</text></svg>')
    return ''.join(svg)


def map_svg(example):
    xy = np.asarray(example['xy_um'])
    size = example['tile_size_um']
    extent = xy.max(axis=0) - xy.min(axis=0) + size
    scale = min(500/extent[0], 260/extent[1])
    base = xy.min(axis=0)
    parts = ['<svg viewBox="0 0 530 290" role="img" aria-label="Synthetic placement and detector flags">']
    for point, inserted, flagged in zip(xy, example['inserted'], example['flagged']):
        x, y = (point - base)*scale + 10
        fill = '#f0b75a' if inserted else '#b9dbd8'
        stroke = '#bd3348' if flagged else 'white'
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{size*scale:.2f}" height="{size*scale:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{2 if flagged else .4}"/>')
    return ''.join(parts) + '</svg>'


def render(run_path, output):
    from .explorer import render_explorer
    return render_explorer(run_path, output)


def render_static(run_path, output):
    run = Path(run_path)
    summary = json.loads((run / 'summary.json').read_text())
    examples = json.loads((run / 'examples.json').read_text())
    protocol = json.loads((run / 'protocol.json').read_text())
    parts = ['''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FloaterFinder experiment</title>
<style>body{margin:0;background:#f3f6f7;color:#19313d;font:16px/1.55 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:30px}h1{font-size:40px;margin-bottom:0}h2{font-size:22px}h3{font-size:17px}section,article{background:white;padding:22px;border-radius:10px;border:1px solid #dce4e6;margin:16px 0}.notice{padding:18px;background:#fff0cc;border-left:5px solid #be822b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px}.grid article{margin:0}svg{width:100%;font:12px system-ui}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #e1e6e8}.scroll{overflow:auto}.legend span{display:inline-block;margin:0 15px 8px 0}code{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style></head><body><main><h1>FloaterFinder</h1><p>Contamination stress test · specimen-aware candidate detection · CPU analysis</p>''']
    parts.append('<div class="notice"><b>Data provenance</b><br>' + '<br>'.join(esc(x) for x in summary['data_provenance']) +
                 '<br>Task: organ classification. This is not evidence of clinical floater detection.</div>')
    parts.append(f'<p>Split: <b>{esc(summary["split"])}</b> · {summary["case_rows"]:,} case/method rows · analysis time {summary["runtime_seconds"]:.2f} seconds · skipped scenarios {len(summary["skipped_cases"])}</p>')
    parts.append('<p><b>Interpretation:</b> the catcher receives the stated organ, also the organ-classification target. Recovery here measures consistency with that supplied context, not independent diagnostic improvement. Tile localization and host-removal harm are separate outcomes.</p>')
    if summary.get('diagnostics'):
        parts.append('<section><h2>Measurement support</h2><ul>' + ''.join(f'<li>{esc(x)}</li>' for x in summary['diagnostics']) + '</ul></section>')
    parts.append('<section><h2>Clean baseline sanity check</h2><p>Inspect all organs before interpreting flip curves. High baseline error makes stability alone misleading.</p><div class="scroll"><table><tr><th>Model / pooling</th><th>Balanced accuracy</th><th>Per-organ recall</th></tr>')
    for name, s in summary['sanity'].items():
        recalls = ', '.join(f'{label}: {r:.1%}' if r is not None else f'{label}: absent' for label,r in zip(s['labels'], s['per_class_recall']))
        parts.append(f'<tr><td>{esc(name)}</td><td>{s["balanced_accuracy"]:.1%}</td><td>{esc(recalls)}</td></tr>')
    parts.append('</table></div></section><h2>Error curves</h2><p>Fraction of initially correct cases that become wrong or abstain after contamination and the indicated filtering. Repeated pairs are correlated; no confidence intervals are implied.</p><div class="legend">')
    parts.extend(f'<span style="color:{color}">● {esc(name)}</span>' for name,color in COLORS.items())
    parts.append('</div><div class="grid">')
    keys = sorted({(r['model'],r['pooling'],r['regime']) for r in summary['curves']})
    for model,pool,regime in keys:
        rows = [r for r in summary['curves'] if (r['model'],r['pooling'],r['regime']) == (model,pool,regime)]
        parts.append(f'<article><h3>{esc(model)} · {esc(pool)} · {esc(regime)}</h3>{curve_svg(rows, protocol["config"]["doses"])}</article>')
    parts.append('</div><section><h2>Unmodified-slide harm check</h2><p>These are controls, not certified floater-free slides. A 99th-percentile calibration threshold does not guarantee a 1% slide alarm rate.</p><div class="scroll"><table><tr><th>Model / pooling / detector</th><th>Mean tiles flagged</th><th>Slides with any flag</th><th>New errors / initially correct</th><th>Unavailable Q2</th></tr>')
    controls = summary['unmodified_controls']
    keys = sorted({(c['model'], c['pooling'], c['method']) for c in controls})
    for key in keys:
        rows = [c for c in controls if (c['model'],c['pooling'],c['method']) == key]
        denom = sum(c['clean_correct'] for c in rows)
        parts.append(f'<tr><td>{esc(" / ".join(key))}</td><td>{np.mean([c["flagged_fraction"] for c in rows]):.1%}</td><td>{np.mean([c["any_flag"] for c in rows]):.1%}</td><td>{sum(c["new_error"] for c in rows)} / {denom}</td><td>{np.mean([c["q2_unavailable_fraction"] for c in rows]):.1%}</td></tr>')
    parts.append('</table></div></section><h2>Placement and flag maps</h2><p>Teal: host tiles. Gold: inserted tiles. Red outline: detector flag. These are coordinate schematics, not histology images or pathologist annotations.</p><div class="grid">')
    for ex in examples:
        shown, total = len(ex['xy_um']), ex.get('total_tiles', len(ex['xy_um']))
        parts.append(f'<article><h3>{esc(ex["model"])} · {esc(ex["host"])}</h3><p>Donor: {esc(ex["donor"])} · actual dose {ex["dose"]:.1%} · showing {shown}/{total} tiles</p>{map_svg(ex)}</article>')
    parts.append('</div><section><h2>Limits and protocol</h2><ul>' + ''.join(f'<li>{esc(x)}</li>' for x in summary['limitations']) + '</ul><details><summary>Frozen analysis configuration</summary><pre>' + esc(json.dumps(protocol,indent=2)) + '</pre></details></section></main></body></html>')
    Path(output).write_text(''.join(parts))
    return Path(output)
