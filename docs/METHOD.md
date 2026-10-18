# Exact method and boundaries of this implementation

## A. Controlled contamination

Start with two original slides from different patients, A and B. Both are intended to be uncontaminated; they may contain cancer. Create two separate virtual samples: all original A tiles plus a connected fragment from original B, and all original B tiles plus a connected fragment from original A. Neither mixture is ever used as a donor, and no tiles are removed from the originals.

For each source slide, sample one same-organ and one different-organ partner when available, deduplicate unordered pairs, then emit both directions exactly once per selected pair. Thus a slide can participate in multiple pairs; this is not a disjoint matching. Selection is seeded and recorded in `pairs.json` with a shared `pair_id` for the two directions. Models share the exact pairing and source tile geometry. If either direction cannot supply a fragment at a dose, both directions are skipped for that model and dose, with reasons recorded.

`mixtures.json` records each successful direction, dose, original host tile count, selected donor tile IDs and inserted coordinates. These recipes reference saved original embeddings; they do not create edited whole-slide images. Two originals yield two contaminated samples at one dose (four samples if counting their clean controls), but still represent only two independent patients. Multiple doses, models, pooling rules and removal methods must not be counted as additional independent patients.

For requested fraction f and N host tiles, insert `ceil(f*N/(1-f))` donor tiles. Report the achieved fraction separately because small integer counts cause rounding. Doses describe the fraction of tiles in the resulting bag, not square millimeters or clinical prevalence. Overlapping tiles require extra care before interpreting tile fractions as area fractions.

Choose a four-connected donor fragment by breadth-first traversal from a seeded anchor. Preserve its relative coordinates and place it beyond the host's right edge without overlapping host tiles. This is an explicit synthetic placement assumption. It does not simulate physical overlap, cutting artifacts, stain changes, or a real glass-slide contamination event. Existing host tissue is never overwritten.

Slide predictions use class-prototype cosine scores. Mean pooling averages each class's score across all tiles. Top-K pooling independently averages the highest K scores for each class, then selects the largest class average. This evaluates an encoder + prompt + pooling pipeline, not an intrinsic diagnostic capability of a bare encoder. Empty filtered bags abstain and never count as recovered.

## B. Candidate detector

Q1 is the highest alternative-organ cosine similarity minus the stated-organ similarity. Q2 is average cosine distance to the K nearest embeddings **outside** a physical exclusion radius. Excluding nearby tiles reduces a fragment's ability to vouch for itself but does not eliminate it: large or repeated fragments can still find foreign neighbors elsewhere.

The search is exact and blocked along both query and reference axes. With query batch B and reference block R=max(B,1024), pairwise working storage is O(BR+BK), while time remains O(N²D). Embeddings and other per-tile arrays still occupy memory proportional to the loaded cohort. Spatial smoothing shares a single geometry pass across the organ and visual signals, then combines the smoothed values by linearity. CPU is practical for selected tile bags; the implementation does not promise seconds for whole-slide-scale repeated searches. If fewer than K distant tiles exist, Q2 is unavailable for that tile and contributes zero after normalization. The report shows that fraction.

Each signal is standardized within the current bag using its mean and standard deviation. Constant signals produce zero. The combined score is `z(Q1) + visual_weight*z(Q2)`. The default weight is fixed at 1. Scores are averaged over spatial neighbors within the smoothing radius. This encourages local consistency but does not enforce a minimum clump size or guarantee isolated noise is removed.

Three detector variants are calibrated independently: organ-only, visual-only, and combined. The calibration threshold is the configured quantile of pooled tile scores **after** smoothing, on separate unmodified calibration slides. A tile is flagged only if its score is strictly greater than the threshold. Larger slides contribute more tiles to this pooled calibration. This is a nominal tile-level calibration, not a guaranteed false-positive rate on independent patients or a slide-level alarm rate.

Within-slide z-scoring is a baseline, not an established optimum: a contaminated distribution can shift its own normalization, and tiny raw differences can become relatively large z-scores. Robust normalization, component-level exclusion and uncertainty-aware thresholds are future ablations.

## C. Comparisons and harm

Six strategies are compared: no filtering, organ-only, visual-only, combined, random removal matched to combined's removed-tile count, and an oracle that removes exactly the inserted tiles. Oracle recovery checks implementation consistency; it is not an achievable detector result.

The benchmark separately reports prediction changes, correctness, correct-to-incorrect transitions, conditional recovery, inserted-tile precision/recall and host-tile removal. Recovery's denominator includes only cases that were originally correct and became incorrect after insertion. Host removal is measured against known original host tiles, even if the source slide contains an undiscovered real contaminant.

Threshold-free detection uses tie-aware AUROC and non-interpolated average precision within each mixture. Aggregate ranking metrics are macro means over bags containing both inserted and host tiles, with the contributing count recorded. We do not pool standardized scores across slides into one ranking. Precision, recall and F1 aggregate tile counts within each model/regime/pooling/method/dose slice; host-removal rate averages per-case host fractions. Zero-dose ranking and recall are undefined. Non-detector baselines have no continuous ranking metric. The report also records abstentions and the achieved dose range.

All original slides and mixtures export tile IDs, source slide IDs, coordinates, class scores, detector scores/flags and kNN availability in indexed NPZ files. Only simulated mixtures carry `is_inserted` evaluator labels. The `detect` command applies the frozen calibration to original slides without producing oracle labels or claiming measured real-floater precision/recall.

Unmodified-slide controls report both the fraction of tiles flagged and the fraction of slides receiving any flag, plus new classification errors after filtering. Native unusual tissue must remain in these controls. Candidate maps are coordinate schematics: teal host tiles, gold inserted tiles, red detector outlines. They are not WSI overlays or expert labels.

## D. Evaluation discipline

Use development cases to decide methods; calibration slides to choose the fixed score thresholds; and test cases only for the final evaluation. Both host and donor patients stay inside their split. If weight/radius/prompt choices are tuned on development insertions, describe that tuning explicitly even though no network weights change.

Calibration records the dataset fingerprint and core source hashes. Changed caches or core implementation require recalibration. Output directories cannot be overwritten. These are audit aids, not security barriers against deliberately rerunning a test. Repeated mixtures of one patient are not independent observations; patient-level uncertainty estimates are not yet implemented.

Fingerprinting is relative to the manifest directory: moving an unchanged relative-path cache bundle preserves its identity. `evaluate` still requires the exact frozen cohort. `detect --new-cohort` explicitly allows a disjoint future cohort with the same encoder fingerprint, ordered labels, normalized text prototypes, dimensions, and a physical scale represented in calibration. Original patient/slide/tile identities and optional image hashes cannot be reused. Compatibility checks do not establish that thresholds generalize to a new site, stain distribution or scanner; those shifts require validation. Old calibration artifacts must be regenerated after this implementation change.

The report caps each coordinate overview at 2000 deterministically sampled tiles so large caches do not create an unbounded SVG. It states the displayed/total tile counts and retains full-bag flag counts. Small flagged regions can be omitted in an overview; the indexed NPZ artifacts always retain all tiles. Missing dose measurements break plotted lines, and a lack of eligible recovery cases is called out explicitly.

Before a real final run, inspect clean-classification confusion matrices, tile quality, prompt validity, specimen metadata, exclusion-radius coverage and near duplicates. Do not automatically discard low-performing models after seeing test results. Predeclare any eligibility threshold using development data.

The source requirement is absence of known contamination, not benign histology. Public download metadata alone cannot verify that requirement. Review source slides for suspected foreign tissue and record reviewer status and exclusions before making controlled-mixture claims; unreviewed sources must remain labeled as such.

## E. Scope for the next stage

The current organ task is an engineering baseline. The catcher receives the stated organ, which is also the organ-classification target. Restoration therefore measures consistency with supplied context, not independent diagnostic improvement: simply returning the stated organ would trivially solve organ classification. The inserted-tile localization and host-removal outcomes remain useful, but a diagnostic-recovery claim needs a disease target withheld from the catcher. Same-organ contaminants may be indistinguishable in this space; a successful same-organ disease-misclassification study needs an additional disease task with trustworthy labels. Metastasis, normal mixed tissue and genuine rare abnormalities can all resemble foreign tissue. Flag for review; do not claim patient-origin identification or automatic clinical tissue removal.

No real clinical effectiveness has been measured. Synthetic vectors verify software. Injected real tissue would validate controlled contamination behavior. Expert-confirmed real floaters would be an additional, separate validation stage.
