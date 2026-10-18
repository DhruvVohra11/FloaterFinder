# Embeddings → benchmark → candidate review

This is the proposed integration agreement for the two workstreams. The defaults below are implemented, not yet an agreement confirmed by both teammates. Set any experimental changes on development data before freezing calibration and opening test results.

## Ownership

**Embedding workstream:** model access/inference, dataset processing and review, tiling at a recorded physical scale, real source patient/slide IDs, and image/text embedding exports. No mixture construction is needed upstream.

**Analysis workstream (this repository):** cache validation, pairing and reciprocal contamination, frozen thresholds, organ/visual/combined detectors, all six removal comparisons, per-tile results, summary tables, and interactive visualization. It runs on CPU without model inference or network calls.

## Common format

Deliver one folder containing `dataset.json`, one text-bank NPZ per model, and one slide NPZ per original slide/model. Use relative paths in the manifest. See [CACHE_CONTRACT.md](CACHE_CONTRACT.md) for required fields and [../examples/dataset.template.json](../examples/dataset.template.json) for the index shape.

Each slide contains projected `features[N,D]`, `xy_um[N,2]`, globally unique string `tile_ids[N]`, and scalar JSON `metadata`. Metadata includes the original `patient_id`, `slide_id`, `organ`, `split`, `model_id`, `encoder_fingerprint`, `tile_size_um`, `stride_um`, and a truthful `provenance` description including review status. Coordinates are top-left positions in microns, not model-input pixel positions. Use NumPy arrays without Python object dtype (`allow_pickle=False` is enforced).

The text bank contains `prototypes[C,D]`, ordered string `labels[C]`, scalar `model_id`, and scalar `encoder_fingerprint`. Image and text vectors must come from the same checkpoint and projection head. Raw vectors are normalized on load. Multiple prompt embeddings can be converted with `make-text-bank`.

Models must share the same original patients, slides, tile IDs, tile order, coordinates, scale, and split. Embedding dimensions can differ between models. Split patients before encoding or pairing; a patient cannot cross development/calibration/test. Supply calibration slides and at least two eligible patients in the evaluated split. Same-organ experiments require two distinct patients of that organ; both regimes should be represented in each evaluated split.

Minimal upstream save example (variables come from the teammate's real pipeline):

```python
import json
import numpy as np

np.savez_compressed("slide.npz", features=image_embeddings,
                    xy_um=tile_top_left_microns,
                    tile_ids=np.asarray(source_tile_ids, dtype=str),
                    metadata=json.dumps(slide_metadata))
np.savez_compressed("text.npz", prototypes=organ_text_embeddings,
                    labels=np.asarray(organ_labels, dtype=str),
                    model_id=model_id, encoder_fingerprint=encoder_fingerprint)
```

Do not fabricate missing patient IDs or coordinates. Record source uncertainty in provenance and resolve it before interpreting experiments.

### Automatic export and intake

Your encoder job can use `CacheWriter` to build the manifest automatically. A complete callable example is [../examples/export_embeddings.py](../examples/export_embeddings.py).

```python
from floaterfinder.intake import CacheWriter

with CacheWriter("exports/pilot-v1") as cache:
    cache.add_model(model_id, encoder_fingerprint, organ_labels, organ_text_embeddings)
    for slide in encoded_original_slides:
        cache.add_slide(features=slide.features, xy_um=slide.xy_um,
                       tile_ids=slide.tile_ids, metadata=slide.metadata)
# dataset.json is published only after the whole cache passes validation.
```

Register each model's text bank before adding its slides. Multiple models may use the same writer. Required metadata is never inferred, and numeric IDs are rejected rather than silently rewritten. A failed export retains its files and `export-status.json` with the error, without publishing `dataset.json`; retry in a new directory after fixing the source.

If the teammate already has slide/text NPZ files matching the contract, assemble a dedicated drop folder instead:

```bash
python -m floaterfinder assemble --input incoming/pilot-v1 --output data/pilot-v1
python -m floaterfinder preflight --dataset data/pilot-v1/dataset.json \
  --config configs/cpu_pilot.json --output outputs/pilot-v1-preflight.json
python -m floaterfinder run --dataset data/pilot-v1/dataset.json \
  --config configs/cpu_pilot.json --output outputs/pilot-v1-development
```

Assembly makes a self-contained copy with relative paths, preserves contract-required vectors, coordinates, IDs, metadata and optional image hashes, records source checksums, and rejects duplicate text banks, unknown NPZ files, split leakage, or mismatched model cohorts. Extra non-contract arrays are not copied. Use a folder containing only the intended slide/text NPZ files; raw prompt arrays must first pass through `make-text-bank`. Outputs must be outside the input folder, and no existing output directory is overwritten.

Preflight checks the exact reciprocal geometry and dose feasibility, without computing detector scores or classification results. It lists both ready and skipped pair/dose scenarios and achieved tile fractions. It exits with code 2 when blocked. New development `run` workflows perform this check automatically and save `preflight.json` before calibration. A ready result means the recipe can run, not that provenance or tissue quality has been independently verified. Exact kNN remains quadratic; start with a small selected tile cohort.

## Experimental conventions to align

| Choice | Implemented convention |
|---|---|
| Contamination dose | `f = donor_tiles / (host_tiles + donor_tiles)`; decimal fraction, so `0.01` means 1% |
| Number inserted | `ceil(f * N_host / (1-f))`; native tiles are retained |
| Other denominator | If upstream uses `r = donor_tiles / host_tiles`, convert with `f = r/(1+r)`; 10% host-relative means 9.09% final-bag |
| Rounding | Requested and achieved doses are both exported; at 100 host tiles, requested 10% inserts 12, achieving 10.714% |
| Mean pooling | Mean of per-tile cosine scores for each class, then argmax |
| Top-K pooling | For **each class independently**, average its largest `min(K,N_remaining)` cosine scores, then argmax; default K=10 |
| Not interchangeable | Top percentage, softmax/logit averaging, pooling embeddings first, and selecting one shared top-K tile set are different experiments |
| Ties/empty bags | First class in text-bank order wins ties; empty bags abstain and count as incorrect |
| Reciprocal pairs | A+B and B+A both use unchanged originals and remain in one split; skip both at a dose if either fragment fails |
| Pair/dose/model alignment | Seeded pairs and donor fragments use the same tile geometry across models; source IDs and placement are exported. Different doses are not guaranteed nested if donor components differ in size |
| kNN | Exact cosine kNN outside an exclusion radius; fewer than K eligible neighbors means unavailable Q2, recorded explicitly |
| Calibration | Per-detector threshold on pooled unmodified calibration tile scores, after smoothing; flags use strict `>` |
| Ranking evaluation | Tie-aware AUROC and average precision within each bag; macro mean across valid mixtures; no cross-slide score ranking |

Default doses: 0%, 0.5%, 1%, 2.5%, 5%, 10%. Default kNN K=10, exclusion radius=1000 μm, smoothing radius=160 μm, visual weight=1, calibration quantile=0.99, seed=7. The default configuration is [../configs/cpu_pilot.json](../configs/cpu_pilot.json). Physical radii must be chosen using development data at the actual exported scale.

## Run the complete loop

From the repository root, with the local environment installed:

```bash
# Validate the teammate's drop and inspect coverage before running experiments.
python -m floaterfinder inspect --dataset data/dataset.json --output outputs/intake.json

# Default: development only. Validates, calibrates, benchmarks, exports, renders.
python -m floaterfinder run --dataset data/dataset.json \
  --config configs/cpu_pilot.json --output outputs/development-001

# Score original slides for review without simulated contamination or oracle labels.
python -m floaterfinder detect --dataset data/dataset.json \
  --calibration outputs/development-001/calibration.json \
  --split development --output outputs/candidates-001

# After protocol decisions are fixed: reuse frozen thresholds on test.
python -m floaterfinder run --dataset data/dataset.json \
  --calibration outputs/development-001/calibration.json \
  --split test --output outputs/final-001
```

Open `outputs/development-001/development/report.html` or `outputs/final-001/test/report.html`. Each output directory must be new. `run.json` records completion or failure. A new test directory is not a fresh holdout. A changed cache or analysis implementation requires recalibration. Fingerprints bind cache contents and relative layout; an unchanged relative-path bundle can move between machines. Do not edit the manifest to switch splits. Omitting `--config` uses packaged CPU pilot defaults, including outside the source checkout.

### Scoring future embeddings

Once thresholds are frozen, a later independent cohort can be scored explicitly:

```bash
python -m floaterfinder detect --dataset data/new-cohort/dataset.json \
  --calibration outputs/development-001/calibration.json \
  --new-cohort --split test --output outputs/new-cohort-candidates
```

The new manifest contains only the new cohort, using the usual split labels. It may contain a subset of the calibrated models. Each model must retain the exact encoder fingerprint, normalized text prototypes, label ordering and dimension, with physical scale represented in calibration. Reuse the frozen text bank. Known patient, slide, tile and optional image-hash overlap with the original cohort is rejected. This flag is only for candidate detection; benchmark evaluation remains bound to the original frozen cohort. The result contains no invented ground truth. Passing compatibility checks does not prove calibration transfers across sites or tissue distributions.

For the fully synthetic software check: `python scripts/smoke.py`. It exercises development, original-slide detection, and frozen test evaluation with two mock encoders. It never downloads data or runs a real model.

## Outputs and review loop

| Artifact | Purpose |
|---|---|
| `intake.json` | Model dimensions, split/patient/tile counts, organ coverage warnings |
| `preflight.json` | Reciprocal mixture feasibility, per-dose counts, exact skip reasons; generated before new calibration |
| `calibration.json` | Thresholds, configuration, cache fingerprint, source hashes |
| `<split>/report.html` | Offline filters for model, pooling, regime, dose, metric and detector; clean confusion matrix, harm checks and candidate maps |
| `<split>/curves.csv` | All aggregate metrics with denominators, achieved-dose ranges, abstentions and kNN availability |
| `<split>/cases.csv` | Per-mixture, per-method, per-pooling predictions, patient IDs, detection and harm metrics |
| `<split>/pairs.json`, `mixtures.json` | Reciprocal source pairs and exact donor tile/placement recipes |
| `<split>/tile_artifacts.json` | Index of every original-slide and mixture NPZ under `tiles/` |
| `tiles/*.npz` | Tile IDs, source slide IDs, coordinates, class scores/labels, all three detector scores and flags, kNN availability; mixtures additionally contain `is_inserted` |
| `candidates-001/candidates.csv` | Every original tile with scores and flags for review; includes unflagged tiles so reviewers can examine false negatives |

The artifact index uses generated safe filenames; source IDs are retained in the index and arrays. It contains no input embeddings. Join the tile IDs to the teammate's tile images or slide viewer. Original-slide exports never invent insertion truth or report real-data precision/recall. Reviewers can inspect candidates and record false alarms, including unusual genuine host tissue. The report shows representative coordinate maps; full NPZ exports cover every direction and dose.

For a print report without JavaScript:

```bash
python -m floaterfinder report --run outputs/final-001/test \
  --output outputs/final-001/test/static-report.html --static
```

## Interpretation and current boundary

Evaluate localization and host-tissue harm together. Compare no filtering, organ-only, visual-only, combined, count-matched random removal, and oracle removal. Recovery only includes originally correct cases made incorrect by insertion; a blank denominator is not zero recovery. Zero-dose duplicate pair cases are useful consistency checks; use the unique original-slide controls for clean false-alarm estimates.

Cases share patients, donors and doses; do not treat method rows as independent samples. No patient-level confidence intervals are claimed. Same-organ donors cannot alter the organ target, so disease-level recovery needs a separate trustworthy task label. The detector sees the stated organ, so organ recovery is context consistency, not evidence of independent diagnostic improvement.

The current end-to-end verification uses synthetic vectors. Real embeddings, reviewed source tissue, histology overlays and expert-confirmed floater truth are not supplied by this analysis implementation. The next integration step is one small, complete real embedding drop satisfying the contract above.
