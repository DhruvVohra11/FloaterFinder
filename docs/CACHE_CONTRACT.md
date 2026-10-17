# Cache contract

The analysis needs projected image embeddings and text vectors from the **same model, checkpoint and retrieval head**. Equal vector dimensions alone do not establish compatibility. The encoder fingerprint is a recorded assertion of that provenance, not an automatic verification of the neural-network internals.

## Dataset index

`examples/dataset.template.json` shows the schema. Paths are relative to this JSON file or absolute. A single model is sufficient. Multiple model entries are processed independently; their vectors are never combined. For comparisons, all entries must use exactly the same slide IDs, tile IDs, coordinates and physical scale.

## Slide NPZ

Use `np.savez_compressed` with these arrays and a JSON string:

| Key | Shape / type | Meaning |
|---|---|---|
| `features` | float `[N,D]` | Finite, nonzero projected image vectors; normalized on load |
| `xy_um` | float `[N,2]` | Tile top-left coordinates in microns, not raw pixels |
| `tile_ids` | string `[N]` | Globally unique source tile IDs |
| `metadata` | scalar JSON string | Fields listed below |
| `image_sha256` | optional string `[N]` | Exact source-image hashes; duplicates across distinct tile IDs are rejected when provided |

Metadata fields:

```json
{
  "slide_id": "source-slide-id",
  "patient_id": "source-case-id",
  "organ": "prostate",
  "split": "development",
  "model_id": "conch",
  "encoder_fingerprint": "record-checkpoint-revision-and-projection-settings",
  "tile_size_um": 112.0,
  "stride_um": 112.0,
  "provenance": "source collection, preprocessing, encoder version and review state"
}
```

Allowed splits are `development`, `calibration`, `test`. Patient identity must be real deidentified source metadata, never invented to pass validation. The detector uses `organ` as specimen context; it does not take a disease diagnosis. This field also defines organ ground truth in the pilot, so organ metadata errors are a limitation.

Convert coordinates from the coordinate convention actually exported by your tiler. For level-zero coordinates use level-zero microns-per-pixel, not the model input resolution's scale. The 112 μm example corresponds to 224 pixels at 0.5 μm/pixel; do not assume that scale from the label “20×” alone. Donor/host scales must match.

The connected-fragment sampler requires donor coordinates on a regular stride grid. Disconnected tissue is allowed, but a requested fragment must fit within a single four-connected component. Insufficient components are recorded as skipped experiments rather than silently replaced with unrelated tissue.

## Text bank NPZ

Required arrays:

- `prototypes`: float `[C,D]`, one nonzero vector per organ.
- `labels`: string `[C]`, unique organ names matching slide metadata.
- `model_id`: scalar string.
- `encoder_fingerprint`: scalar string matching the slide cache.

To average multiple phrasings, first export `prompt_features[C,P,D]` with `labels`, `model_id`, and `encoder_fingerprint`. Then:

```bash
python -m floaterfinder make-text-bank --input data/prompt-features.npz --output data/text.npz
```

The tool normalizes each prompt vector, averages within class, and normalizes the class mean. It requires the same number of paraphrases per class. The included text prompts are provisional, not pathologist-approved.

## Data integrity limits

The validator checks known patient splitting, unique tile IDs/coordinates, optional byte-level image duplicates, model provenance labels and matching tile selections across models. It does not detect re-encoded or near-duplicate tissue, incorrect organ metadata, clinically invalid tile regions, or foundation-model pretraining exposure. These still need data review.
