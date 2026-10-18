// Execute the report's own inline code in a DOM test runtime. No browser or
// network is used. Python/runtime users do not need this development dependency.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {JSDOM, VirtualConsole} from 'jsdom';

const latest = () => fs.readdirSync('outputs').filter(name => name.startsWith('smoke-')).sort().at(-1);
const path = process.argv[2] ?? `outputs/${latest()}/final-run/test/report.html`;
const html = fs.readFileSync(path, 'utf8');
const errors = [];
const console = new VirtualConsole();
console.on('jsdomError', error => errors.push(error.message));
function open(content) {
  return new JSDOM(content, {runScripts: 'dangerously', virtualConsole: console});
}
const dom = open(html), window = dom.window, document = window.document;
const $ = id => document.getElementById(id);
function select(id, value) {
  $(id).value = String(value);
  $(id).dispatchEvent(new window.Event('change', {bubbles: true}));
}
const payload = JSON.parse($('payload').textContent);
assert.ok($('baseline').textContent.includes('Balanced accuracy'));
let selections = 0;
for (const model of [...$('model').options].map(o => o.value)) {
  for (const pool of ['mean', 'topk']) {
    for (const regime of [...$('regime').options].map(o => o.value)) {
      select('model', model); select('pool', pool); select('regime', regime);
      for (const dose of [...$('dose').options].map(o => o.value)) {
        select('dose', dose);
        const rows = payload.summary.curves.filter(r => r.model === model && r.pooling === pool && r.regime === regime && r.dose === Number(dose));
        assert.equal($('comparison').querySelectorAll('tbody tr').length, rows.length);
        if (Number(dose) === 0) assert.ok($('placement').textContent.includes('No stored example'));
        selections++;
      }
    }
  }
}
for (const metric of [...$('metric').options]) {
  select('metric', metric.value);
  assert.equal($('curve-title').textContent, metric.textContent);
  assert.ok(!/NaN|Infinity/.test($('curve').innerHTML));
}
for (const method of ['organ_only', 'visual_only', 'combined']) {
  select('detector', method);
  const example = payload.examples[Number($('example').value)];
  assert.equal($('placement').querySelectorAll('rect').length, example.xy_um.length);
  $('truth').checked = true; $('truth').dispatchEvent(new window.Event('change'));
  assert.equal($('placement').querySelectorAll('rect[fill="#f0b75a"]').length, example.inserted.filter(Boolean).length);
}
// A missing dose must break the line, not imply measured performance between
// its neighboring doses. Exercise the exact HTML with a modified JSON payload.
const modified = structuredClone(payload);
modified.summary.curves = modified.summary.curves.filter(r => r.dose !== .025);
modified.summary.data_provenance = ['</script><script>window.INJECTED=true</script>'];
const safe = JSON.stringify(modified).replaceAll('<', '\\u003c').replaceAll('>', '\\u003e');
const altered = html.replace(/(<script type="application\/json" id="payload">)[\s\S]*?(<\/script>)/, (_, a, b) => a + safe + b);
const other = open(altered);
assert.equal(other.window.INJECTED, undefined);
const lines = other.window.document.querySelectorAll('#curve path[data-method="no_filter"]');
assert.equal(lines.length, 3, 'Six doses with middle dose absent have three segments');
assert.deepEqual(errors, []);
dom.window.close(); other.window.close();
process.stdout.write(`Report DOM checks passed: ${selections} filter combinations, all metrics/detectors, ground truth, missing-dose gaps and metadata escaping.\n`);
