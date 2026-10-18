from __future__ import annotations

from collections import deque
from decimal import Decimal, ROUND_CEILING
import numpy as np


def unit(x):
    x = np.asarray(x)
    if x.dtype.kind not in 'fiu' or x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError('Embeddings must be real vectors with a nonempty feature dimension')
    x = x.astype(np.float64)
    scale = np.max(np.abs(x), axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (scale == 0).any():
        raise ValueError('Embeddings must be finite and nonzero')
    # Scaling first avoids overflow/underflow for finite future encoder outputs.
    x = x / scale
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def cosine_scores(features, prototypes):
    return unit(features) @ unit(prototypes).T


def donor_count(host_tiles, fraction):
    """Final-bag fraction; decimal arithmetic avoids ceil(7.000000000000001)."""
    if host_tiles < 1 or not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('Need positive host count and finite dose in [0,1)')
    f = Decimal(str(fraction))
    return int((f * int(host_tiles) / (1 - f)).to_integral_value(rounding=ROUND_CEILING))


def ranking_metrics(scores, truth):
    """Tie-aware AUROC and non-interpolated average precision for one bag.

    Scores are standardized per bag, so aggregate these metrics across bags,
    never by treating scores from different slides as a shared ranking.
    """
    scores, truth = np.asarray(scores, dtype=float), np.asarray(truth, dtype=bool)
    if scores.ndim != 1 or scores.shape != truth.shape or not np.isfinite(scores).all():
        raise ValueError('Expected finite scores and aligned binary truth')
    positives, negatives = int(truth.sum()), int((~truth).sum())
    if not positives or not negatives:
        return {'auroc': None, 'average_precision': None}
    order = np.argsort(-scores, kind='stable')
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered) != 0), len(scores)-1]
    tp = np.cumsum(truth[order])[ends]
    fp = ends + 1 - tp
    recall, fpr = np.r_[0., tp / positives], np.r_[0., fp / negatives]
    auroc = np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2)
    ap = np.sum(np.diff(recall) * tp / (tp + fp))
    return {'auroc': float(auroc), 'average_precision': float(ap)}


def predict(scores, pooling, top_k):
    if len(scores) == 0:
        return None  # Filtering every tile is abstention, never a recovery.
    if pooling == 'mean':
        pooled = scores.mean(axis=0)
    elif pooling == 'topk':
        k = min(top_k, len(scores))
        if k < 1:
            raise ValueError('top_k must be positive')
        # Class-specific top K, followed by mean cosine similarity per class.
        pooled = np.partition(scores, len(scores) - k, axis=0)[-k:].mean(axis=0)
    else:
        raise ValueError('Unknown pooling')
    return int(pooled.argmax())


def zscore(values, valid=None):
    values = np.asarray(values, dtype=float)
    valid = np.ones(len(values), dtype=bool) if valid is None else valid
    out = np.zeros(len(values), dtype=float)
    if valid.any():
        sd = values[valid].std()
        if sd > 1e-6:
            out[valid] = (values[valid] - values[valid].mean()) / sd
    return out


def _geometry(xy_um, n, radius_um, chunk_size):
    xy = np.asarray(xy_um, dtype=float)
    if xy.shape != (n, 2) or not np.isfinite(xy).all():
        raise ValueError('Coordinates must be finite [N,2]')
    if type(chunk_size) is not int or chunk_size < 1 or not np.isfinite(radius_um) or radius_um < 0:
        raise ValueError('Invalid spatial parameters')
    return xy


def _distance_squared(a, b):
    dx = a[:, None, 0] - b[None, :, 0]
    dy = a[:, None, 1] - b[None, :, 1]
    return dx*dx + dy*dy


def far_knn(features, xy_um, radius_um, k, chunk_size):
    """Exact two-axis blocked cosine kNN. O(N^2 D) time.

    Pairwise blocks use at most B*max(B,1024) entries, plus B*K candidates.

    A tile with fewer than k eligible distant neighbors has unavailable Q2.
    This is reported explicitly; missing Q2 contributes zero after z-scoring.
    """
    x = unit(features)
    xy = _geometry(xy_um, len(x), radius_um, chunk_size)
    if type(k) is not int or k < 1:
        raise ValueError('Invalid kNN parameters')
    n = len(x)
    values, valid = np.zeros(n), np.zeros(n, dtype=bool)
    if k >= n:
        return values, valid
    reference_size = max(chunk_size, 1024)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        best = np.full((stop-start, k), -np.inf, dtype=np.float32)
        counts = np.zeros(stop-start, dtype=np.int64)
        for ref in range(0, n, reference_size):
            end = min(n, ref + reference_size)
            eligible = _distance_squared(xy[start:stop], xy[ref:end]) > radius_um**2
            counts += eligible.sum(axis=1)
            similarity = x[start:stop] @ x[ref:end].T
            similarity[~eligible] = -np.inf
            candidates = np.concatenate([best, similarity], axis=1)
            candidates.partition(candidates.shape[1]-k, axis=1)
            best = candidates[:, -k:].copy()
        ok = counts >= k
        if ok.any():
            values[start:stop][ok] = (1 - np.clip(best[ok], -1, 1)).mean(axis=1)
            valid[start:stop] = ok
    return values, valid


def smooth(values, xy_um, radius_um, chunk_size):
    values = np.asarray(values, dtype=float)
    xy_um = _geometry(xy_um, len(values), radius_um, chunk_size)
    if radius_um <= 0:
        return values.copy()
    out = np.empty_like(values)
    for start in range(0, len(values), chunk_size):
        xy = xy_um[start:start+chunk_size]
        total = np.zeros_like(values[start:start+len(xy)])
        count = np.zeros(len(xy))
        for ref in range(0, len(values), max(chunk_size, 1024)):
            end = min(len(values), ref+max(chunk_size, 1024))
            neighbors = _distance_squared(xy, xy_um[ref:end]) <= radius_um**2
            total += neighbors @ values[ref:end]
            count += neighbors.sum(axis=1)
        out[start:start+len(xy)] = total / (count if values.ndim == 1 else count[:, None])
    return out


def foreignness(features, xy_um, prototypes, stated_organ_index, cfg):
    scores = cosine_scores(features, prototypes)
    own = scores[:, stated_organ_index]
    others = np.delete(scores, stated_organ_index, axis=1).max(axis=1)
    visual, valid = far_knn(features, xy_um, cfg['exclusion_radius_um'],
                             cfg['neighbors'], cfg['chunk_size'])
    q1, q2 = zscore(others - own), zscore(visual, valid)
    # One geometry pass for both signals. Averaging is linear, so combining
    # their smoothed values is the same detector, without three distance scans.
    smoothed = smooth(np.column_stack([q1, q2]), xy_um, cfg['smoothing_radius_um'], cfg['chunk_size'])
    signals = {'organ_only': smoothed[:, 0], 'visual_only': smoothed[:, 1],
               'combined': smoothed[:, 0] + cfg['visual_weight'] * smoothed[:, 1]}
    return scores, signals, valid


def connected_fragment(xy_um, stride_um, count, seed):
    """BFS on the donor's four-connected tile grid; no invented tissue bridges."""
    if count < 0 or count > len(xy_um):
        raise ValueError('Requested fragment exceeds donor tissue')
    if count == 0:
        return np.empty(0, dtype=int)
    origin = xy_um.min(axis=0)
    grid_float = (xy_um - origin) / stride_um
    grid = np.rint(grid_float).astype(int)
    if not np.allclose(grid_float, grid, atol=1e-3, rtol=0):
        raise ValueError('Donor coordinates must lie on the stated regular grid')
    lookup = {tuple(p): i for i, p in enumerate(grid)}
    if len(lookup) != len(grid):
        raise ValueError('Multiple donor coordinates map to the same grid location')
    rng = np.random.default_rng(seed)
    visited = set()
    for anchor in rng.permutation(len(grid)):
        if int(anchor) in visited:
            continue
        queue, selected = deque([int(anchor)]), []
        visited.add(int(anchor))
        while queue:
            i = queue.popleft()
            selected.append(i)
            if len(selected) == count:
                return np.array(selected)
            gx, gy = grid[i]
            for p in [(gx+1, gy), (gx-1, gy), (gx, gy+1), (gx, gy-1)]:
                j = lookup.get(p)
                if j is not None and j not in visited:
                    visited.add(j)
                    queue.append(j)
    raise ValueError('No connected donor component is large enough for this dose')


def inject(host, donor, fraction, seed):
    hm, dm = host['meta'], donor['meta']
    if hm['patient_id'] == dm['patient_id']:
        raise ValueError('Host and donor must be different patients')
    for key in ('model_id', 'encoder_fingerprint', 'stride_um', 'tile_size_um'):
        if hm[key] != dm[key]:
            raise ValueError(f'Incompatible host/donor {key}')
    if not 0 <= fraction < 1:
        raise ValueError('Dose must be in [0,1)')
    n = len(host['features'])
    count = donor_count(n, fraction)
    chosen = connected_fragment(donor['xy_um'], dm['stride_um'], count, seed)
    donor_xy = donor['xy_um'][chosen].copy()
    if count:
        # Non-overlapping synthetic placement to the right of host tissue.
        # Gap exceeds smoothing radius in the example configuration.
        donor_xy -= donor_xy.min(axis=0)
        donor_xy += [host['xy_um'][:, 0].max() + max(hm['tile_size_um'], 2 * hm['stride_um']),
                     host['xy_um'][:, 1].min()]
    return {'features': np.concatenate([host['features'], donor['features'][chosen]]),
            'xy_um': np.concatenate([host['xy_um'], donor_xy]),
            'is_foreign': np.arange(n + count) >= n,
            'donor_indices': chosen, 'actual_fraction': count / (n + count)}


def detection_metrics(flagged, truth):
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    return {'true_positive_tiles': tp, 'false_positive_tiles': fp,
            'precision': tp / int(flagged.sum()) if flagged.any() else None,
            'recall': tp / int(truth.sum()) if truth.any() else None,
            'host_removal_rate': fp / int((~truth).sum()),
            'flagged_tiles': int(flagged.sum())}
