# FloaterFinder

**Measure how foreign tissue changes a pathology model's predictions, then test whether a training-free candidate detector helps without removing genuine tissue.**

This implementation runs on **CPU and saved embeddings**. It does not fine-tune models. It treats the prediction task as **organ classification**, not cancer diagnosis. It simulates insertion of real donor-tile embeddings when available; the included demonstration uses explicitly synthetic vectors.

## What works now

**Team split:** model inference, dataset processing and real embeddings belong to the upstream workstream. This repository owns the complete analysis loop. Start with the [teammate handoff and experimental conventions](docs/TEAM_HANDOFF.md).

```bash
python -m floaterfinder run --dataset data/dataset.json \
  --config configs/cpu_pilot.json --output outputs/development-001
```

This validates the cache, records coverage, calibrates on separate slides, benchmarks development mixtures, exports every tile score and flag, and writes `outputs/development-001/development/report.html`. The offline report supports model/pooling/dose/detector selection, metric curves, confusion matrices, candidate maps, and host-removal checks. Final `run --split test` requires `--calibration` from the frozen workflow. Use `inspect` for intake and `detect` for per-tile original-slide candidate exports.

For the first real embedding drop, use `CacheWriter` in the encoder job or `assemble` on a folder of existing slide/text NPZ exports. `preflight` checks reciprocal fragment feasibility before kNN; new development runs perform it automatically. The [handoff guide](docs/TEAM_HANDOFF.md#automatic-export-and-intake) has copyable commands and an exporter example. Create a source-only handoff archive with `python scripts/build_handoff.py --output outputs/teammate-handoff.zip`.

The [readiness audit](docs/READINESS.md) records numerical verification, report interaction checks, performance measurements and remaining limits. Detection supports explicit `--new-cohort` scoring with frozen compatibility checks. Exact kNN now bounds pairwise working memory along both axes; large report maps are labeled sampled overviews, with complete tile exports retained.

| Component | Status |
|---|---|
| Patient-separated cache validation and image/text space checks | Implemented and tested |
| Connected donor fragment selection and simulated placement | Implemented and tested |
| Reciprocal A-with-B and B-with-A mixtures from unchanged originals | Implemented; paired recipes and inserted tile IDs exported |
| Method A: mean / class-specific top-K stress test | Implemented and tested |
| Method B: organ mismatch + distant kNN + spatial averaging | Implemented and tested |
| Calibration on separate unmodified slides, after smoothing | Implemented and tested |
| Organ-only, visual-only, combined, no-filter, random, oracle comparisons | Implemented and tested |
| Recovery and host-removal harm measurements | Implemented and tested |
| Organ confusion matrices, error curves, coordinate flag maps | Implemented; standalone HTML and machine-readable results |
| Tie-aware per-mixture AUROC / average precision, F1 and abstentions | Implemented; denominators and achieved-dose ranges exported |
| Single-command workflow and original-slide candidate exports | Implemented; development defaults and frozen-calibration test runs |
| GDC metadata cohort planner | Executed against public GDC metadata; no slide images downloaded |
| Quilt metadata audit on Delta | CPU-only script prepared and fixture-tested; awaiting authenticated Delta access |
| Organizer `/embed` client | Implemented and mocked in tests; no real inference request sent |
| Text-embedding import and prompt averaging | Implemented; matching real text vectors still needed |
| Whole-slide download, tiling, real image/text encoding | Not executed |
| Pathologist-reviewed real-floater validation | Not available |

**No GPU computation is needed for the local workflow.** The optional `service-export` command does use the organizer's remote GPU; it is disabled unless explicitly enabled. It must be run inside Delta. Source code contains endpoint addresses and environment variable names, never the supplied API keys.

## Run locally

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,service]'
python -m pytest -q
python scripts/smoke.py
```

The smoke command creates a new folder under `outputs/`, validates two distinct mock embedding spaces, calibrates three detectors, evaluates both pooling methods and six removal strategies on development and test, exports original-slide candidates, and writes `final-run/test/report.html`. It uses no network or model inference. The synthetic encoders are deliberately not named CONCH or MUSK.

## The actual experiment

1. Choose a cohort and record real patient IDs. Separate development, calibration and test **before** choosing donor-host pairs.
2. Review the organ/site context, tile quality and suspected existing contamination. Original slides should be uncontaminated, but may contain cancer. Public metadata alone does not confirm the absence of floaters. Keep unusual native tissue; removing all difficult native examples would bias evaluation.
3. Tile at a recorded physical scale; save tile coordinates and unique IDs. Preserve one shared tile selection across model comparisons.
4. Obtain one model's projected image embeddings and matching text embeddings. See [API integration](docs/API_INTEGRATION.md).
5. Construct the cache manifest in [the cache contract](docs/CACHE_CONTRACT.md), then run the commands below.
6. Inspect development results. Set any hyperparameters before recalibrating and before final testing. Default visual weight is fixed at 1; this implementation does not tune it on injected examples.
7. Evaluate the held-out test pairs only after freezing those decisions. Report both detection/recovery and false removal of host tissue.

```bash
python -m floaterfinder validate --dataset data/dataset.json
python -m floaterfinder calibrate --dataset data/dataset.json --config configs/cpu_pilot.json --output outputs/calibration.json
python -m floaterfinder evaluate --dataset data/dataset.json --calibration outputs/calibration.json --split development --output outputs/development-001
# After settings are fixed:
python -m floaterfinder evaluate --dataset data/dataset.json --calibration outputs/calibration.json --split test --output outputs/test-001
```

Create the `outputs` directory first. Every evaluation directory and calibration filename must be new. Cache and core-code changes invalidate calibration. These checks prevent accidental mixing; they cannot prevent deliberate repeated test peeking in new directories.

Every selected A–B pair produces both A-with-B and B-with-A from the originals. `pairs.json` links their directions with a shared pair ID; `mixtures.json` records the inserted source tile IDs and coordinates at each dose. Both patients and every derived mixture stay inside the same split. The recipes double the contaminated examples per selected pair, not the number of independent patients. If one direction cannot be constructed, both directions are skipped at that dose.

## Real data head start

The organizer-provided Quilt collection is available on Delta. See [Quilt intake and commands](docs/QUILT_DELTA.md). Audit patient/slide linkage and physical coordinates before importing it into this WSI benchmark; image–caption pairs alone do not establish those fields.

`scripts/plan_gdc_cohort.py` queries **public file metadata only**. It selects one diagnostic slide per case across TCGA-PRAD, TCGA-COAD and TCGA-LUAD, records checksums and sizes, and writes a standard download manifest plus a proposed split. The organ is inferred from the project and must be reviewed against specimen metadata.

```bash
python scripts/plan_gdc_cohort.py --output outputs/gdc-pilot-plan --per-project 6
```

The executed pilot selected 18 slides from 18 patients, with two patients per organ in each split. Full files total about 15.6 GB. This is a small engineering pilot, not a statistically powered validation cohort. No slides were downloaded. Prefer organizer-provided images or embeddings if they cover the required metadata. The proposed files are a seeded convenience sample from one metadata page, not a random sample of all TCGA.

## What the results mean

- Error curves measure correct-to-incorrect changes among originally correct **organ classifications**. They also report overall accuracy and changes regardless of correctness.
- Same-organ donor experiments measure candidate detection and adverse effects; organ classification cannot measure a change in a same-organ disease subtype.
- The detector sees the current contaminated bag and declared organ only. It does not receive the inserted-tile mask, donor identity or uncontaminated reference features. Those are used by the evaluator and oracle only.
- A calibration quantile of 0.99 describes a pooled **tile-level** threshold on unmodified calibration data. It is not a guarantee of 1% false alarms on new slides, and not a 1% slide alarm rate.
- All native tiles are preserved during insertion. Filtering is a reversible analysis intervention, not an automatic clinical action.
- Unmodified public slides are not proven floater-free. High-scoring real regions require review.
- Donor-host mixtures sharing patients are correlated. The report does not invent patient confidence intervals or clinical performance claims.

Read [the exact method and limits](docs/METHOD.md) before interpreting results.
