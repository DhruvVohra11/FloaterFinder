# Organizer service integration

Verified against the organizer's [student guide](https://github.com/ncsa/ashby-hackathon-2026/blob/main/STUDENT_GUIDE.md), [server implementation](https://github.com/ncsa/ashby-hackathon-2026/blob/main/serve.py), and [model adapters](https://github.com/ncsa/ashby-hackathon-2026/blob/main/models.py) on October 17, 2026. Pin the actual deployed revision with the organizer before interpreting real results.

The four supplied services are CONCH, MUSK, QuiltNet and PLIP. They run inside Delta. A request sent from a CPU login node still uses the server's GPU. No model endpoint was called during this build.

## Important API limitation

`/embed` supplies image vectors. `/classify` supplies only the winning class and its score. The current API has no public text-embedding endpoint and does not return every class's cosine score. A winning softmax score is not interchangeable with the report-mismatch signal.

For the complete offline experiment, get the matching text prototype vectors separately. Options are:

1. Ask the organizer to export a small prompt bank using the exact deployed adapter.
2. With approved model access and a compatible environment, load the same adapter locally and encode the prompt bank on CPU.

No request to the organizers has been sent by this build. Neither option is an automatic remote fallback. Do not repeatedly query subsets of classes to reconstruct hidden scores.

`scripts/export_text_cpu.py` prepares option 2 using a local checkout of the organizer repository. It disables CUDA and uses existing cached weights by default. It was not executed against real weights in this build. Model packages must already be installed, and the fingerprint must match the deployed image encoder. `--allow-model-download` is an explicit option if downloads are wanted later.

```bash
python scripts/export_text_cpu.py --organizer-repo /path/ashby-hackathon-2026 \
  --model conch --encoder-fingerprint VERIFIED_CHECKPOINT_AND_ADAPTER_REVISION \
  --output data/prompt-features.npz --hf-access-approved
python -m floaterfinder make-text-bank --input data/prompt-features.npz --output data/text.npz
```

The inspected organizer CONCH and MUSK adapters explicitly enable their retrieval projections. This is useful evidence about the published code, but actual checkpoint/revision identity must still match the text bank. The service response does not carry a checkpoint hash.

## Prepared client for later GPU use

`service-export` reads `examples/tiles.template.json` with real image paths, identifiers, physical coordinates and split metadata. It is limited to 2,000 tiles per invocation and 32 images per request. It makes no automatic retry that could silently repeat GPU work.

Only run this inside Delta when ready to use remote inference. Set the model's key in its named environment variable from `configs/services.json`. Actual keys were not saved. For CONCH/MUSK, `--hf-access-approved` is an attestation that your own Hugging Face access is approved, not a substitute for approval.

```bash
python -m floaterfinder service-export \
  --model conch \
  --tiles data/tiles.json \
  --encoder-fingerprint VERIFIED_CHECKPOINT_AND_ADAPTER_REVISION \
  --output data/conch-export \
  --allow-remote-inference \
  --hf-access-approved
```

Without the explicit remote-inference flag, the command refuses before contacting a service. It writes per-slide caches, exact image hashes and an export manifest. Add the matching text bank to create the dataset index. The code avoids printing headers and does not follow redirects carrying credentials.

Throughput in `export.json` is **end-to-end tiles/second including network and server queuing**, not isolated GPU encoder throughput. The API does not expose GPU memory usage; do not invent that metric. Whole datasets should use a dedicated GPU job, following organizer guidance.

## What to ask for next

- Existing image embedding caches with tile coordinates and patient metadata, if any.
- A small matching text-vector export for the reviewed organ prompts.
- Exact deployed checkpoint/adapter identity and confirmed projected embedding dimension.
- Locations of organizer-provided slide images or tiles to avoid unnecessary downloads.

PathGen-CLIP is not one of the supplied endpoint configurations. It can be used later by importing conforming image/text caches as a separate model experiment.
