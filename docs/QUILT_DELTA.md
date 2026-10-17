# Quilt on Delta: real-image intake

The organizer reports images at `/work/nvme/bfqi/shared/datasets/quilt/images_part_1/quilt_1m/` through `images_part_10/quilt_1m/`, and metadata at `/work/nvme/bfqi/shared/datasets/quilt/quilt_1M_lookup-002.csv`. Read these files in place. Write artifacts only to your own allocation directory or your team's directory; do not copy the shared images.

## Current status

The local pipeline has only synthetic vectors. A Delta connection attempted on October 17, 2026 reached the login service but could not authenticate noninteractively. No Quilt metadata, images or model embeddings have been read from Delta yet. The audit below is prepared and tested against local fixtures, not the actual shared dataset.

## First action: metadata audit, no GPU

`scripts/audit_quilt.py` uses Python's standard library. It streams at most 10,000 CSV records, retains only recognized metadata fields in a 120-record reservoir, and checks sampled image paths across the ten documented directories. It does not decode or copy images, load model weights, call inference endpoints, or load the entire CSV into RAM. Multiple captions may refer to the same image, so sampled CSV records are not a count of distinct images. The default sample covers a prefix of the CSV and is not representative of the whole collection.

After authenticating with `ssh delta` on your laptop, transfer this small script into your own directory:

```bash
ssh delta 'mkdir -p /work/nvme/bfqi/dvohra/floaterfinder/tools'
scp /Users/dhruvvohra/Desktop/Ashby/FloaterFinder/scripts/audit_quilt.py delta:/work/nvme/bfqi/dvohra/floaterfinder/tools/
```

Inside Delta, run:

```bash
python3 /work/nvme/bfqi/dvohra/floaterfinder/tools/audit_quilt.py \
  --output /work/nvme/bfqi/dvohra/floaterfinder/quilt-audit-001
```

Use a new output name for each audit. It writes `audit.json` and `review-sample.json` (metadata and references to original files only). No API key is needed. The complete list of CSV headers is recorded even though only recognized columns are retained in the sample. This permits a follow-up inspection of an unexpected metadata schema without guessing its meaning.

## Why the metadata gate matters

Quilt is an image–text collection assembled partly from educational videos; it is not automatically a collection of spatially indexed, patient-linked whole-slide tiles. See the [authors' description](https://quilt1m.github.io/) and [repository](https://github.com/wisdomikezogwo/quilt1m).

For the existing reciprocal floater benchmark we need real source patient and slide identities, reviewed organ context, tile coordinates, physical scale and enough neighboring tiles per donor fragment. Both A-with-B and B-with-A must stay in the same split. Captions, filenames and video IDs must not be relabeled as patient identities. Magnification labels alone cannot provide microns-per-pixel or spatial coordinates.

- If the supplied metadata has the required linkage, verify its conventions, review source images, and build the existing tile manifest from those records.
- If that metadata is absent, Quilt can support an explicitly image-level model smoke test or an artificial image-bag experiment. It cannot establish patient-separated or spatial floater performance. Such an experiment needs its own design, not invented metadata to satisfy the WSI loader.
- For the original patient-A/patient-B experiment, use verified WSI-derived tiles or additional source mappings. The GDC download plan remains an alternative, with contamination review still required.

The audit always leaves `wsi_benchmark_ready` false: recognized column names alone do not verify their values, semantics or tissue review status. No real experiment should be labeled complete based on this audit.

## Model stage after data eligibility

Start with a small reviewed selection and one model. PLIP does not require the CONCH/MUSK gated-access confirmation; CONCH and MUSK require the runner's own approved access. Shared `/embed` requests consume the organizer's GPU even when sent from a CPU login shell. Keep the pilot small and cache outputs. Do not send the full Quilt dataset to the shared service.

QuiltNet was trained on Quilt data, so results on this collection are not independent validation of QuiltNet. Audit possible pretraining overlap for other chosen models as well. The complete current detector additionally needs matching text vectors; see [API integration](API_INTEGRATION.md). A successful image embedding request alone does not complete that dependency.
