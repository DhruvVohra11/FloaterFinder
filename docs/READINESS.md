# Analysis readiness audit

This audit establishes software behavior on controlled fixtures. It does not establish real-floater detection accuracy. Real embedding datasets and expert-reviewed contamination are still required for that claim.

## Verification

- Python regression coverage includes intake, patient separation, reciprocal pairing, exact dose rounding, pooling, geometry, thresholds, per-tile exports, recovery denominators, native-tissue harm and complete development/test workflows.
- Exact kNN is compared with an independent dense reference across reference-block boundaries and three query chunk sizes. Spatial averaging is checked against dense multi-signal averaging and translated coordinates.
- The recovery fixture deliberately creates classification errors. It confirms measured recovery and oracle restoration, while preserving a counterexample in which combined detection removes native tissue despite restoring classification. Recovery alone is not a localization success criterion.
- New-cohort tests accept disjoint compatible exports, reject changed prototypes/scales and reused identities, and preserve strict dataset matching for benchmark evaluation.
- Extreme finite embeddings normalize without overflow/underflow. Off-grid donor coordinates and coordinates collapsing onto the same grid cell are rejected.
- Report code is executed in jsdom without browser/network access: 48 model/pooling/regime/dose combinations, every metric and detector, ground-truth toggles, missing-dose gaps, and escaped metadata. This checks DOM behavior, not pixel-level visual layout or browser-specific rendering.
- Both interactive and static curves break at unmeasured doses. Large maps have an explicit display limit; regression checks confirm the full NPZ exports and full-bag flag counts are preserved.
- A built wheel was installed into a temporary target, imported outside the source checkout, and used for a complete development run with packaged default configuration.
- GitHub Actions runs the Python suite and synthetic end-to-end/report checks on Python 3.10, 3.12 and 3.13. Local completion does not itself imply a remote CI run passed; inspect the GitHub run status.

## Measured optimization

[Machine-readable profile](validation/detector-profile-20261018.json) compares `core.foreignness` with commit `a9bde03`. Each measurement uses one warmup plus the median of three runs on this machine, NumPy 2.5.4, three class prototypes, and the recorded CPU pilot configuration. Source hashes and maximum numerical differences are included. Inputs are seeded synthetic vectors and coordinates.

| Tiles × dimensions | Previous median | Current median | Speedup | Largest pairwise matrix entries, before → after |
|---|---:|---:|---:|---:|
| 512 × 128 | 21.3 ms | 6.1 ms | 3.50× | 65,536 → 65,536 |
| 2,048 × 128 | 353.2 ms | 78.6 ms | 4.49× | 262,144 → 131,072 |
| 4,096 × 256 | 1,069.2 ms | 306.6 ms | 3.49× | 524,288 → 131,072 |

The maximum observed standardized/smoothed signal difference was below 0.000016. This is numerical agreement, not bitwise identity; samples arbitrarily close to a threshold can change flags. Recalibration is required after the implementation change.

These are local microbenchmarks, not end-to-end runtime guarantees. Pairwise matrix entries are not peak process RAM. Search remains O(N²D), all loaded embedding arrays still consume memory, and NPZ/CSV output incurs additional time and storage. No tile subsampling or approximate neighbor search was introduced into detection or metrics. Only displayed coordinate overviews are sampled.

## Reproduce

```bash
python -m pytest -q
python scripts/smoke.py
npm ci --ignore-scripts --no-audit --no-fund
npm run test:report
python scripts/profile_detector.py --baseline-ref a9bde03 \
  --output outputs/new-detector-profile.json
```

The Node dependency is needed only for report testing, not for analysis or viewing exported HTML. Use a fresh output filename for profiling. The profile command requires the baseline commit to exist in the local Git history.

## Future embedding acceptance

1. Export matching projected image/text spaces with truthful metadata using the [handoff contract](TEAM_HANDOFF.md).
2. Run `assemble` if needed, then `preflight` and the development workflow. Inspect physical geometry, organ coverage, kNN availability and actual dose rounding.
3. Use development data to fix choices before freezing calibration and opening test results. Recalibrate after core or cache changes. Relative-path bundles can move unchanged between machines.
4. For later original-slide candidates, use `detect --new-cohort` with a compatible, disjoint manifest and the frozen text bank. Compatibility is not evidence of calibration transfer under domain shift.
5. Inspect precision/recall and native-tissue removal together. Undefined recovery means there were no eligible cases, not that recovery was perfect or zero.

Still outstanding for scientific conclusions: real embedding experiments, independent source/tissue review, adequate independent patient counts, patient-level uncertainty estimates, disease targets withheld from the detector when claiming diagnostic recovery, and expert-confirmed real-floater validation. The current synthetic placement assumption and within-bag normalization also remain experimental limitations.
