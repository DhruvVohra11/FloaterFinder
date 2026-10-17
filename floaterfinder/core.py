from __future__ import annotations

from collections import deque
import math
import numpy as np


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or (norms < 1e-8).any():
        raise ValueError('Embeddings must be finite and nonzero')
    return x / norms


def cosine_scores(features, prototypes):
    return unit(features) @ unit(prototypes).T


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


def far_knn(features, xy_um, radius_um, k, chunk_size):
    """Exact blocked cosine kNN. O(N^2 D) time; O(chunk_size*N) pair memory.

    A tile with fewer than k eligible distant neighbors has unavailable Q2.
    This is reported explicitly; missing Q2 contributes zero after z-scoring.
    """
    x = unit(features)
    xy = np.asarray(xy_um, dtype=float)
    if k < 1 or chunk_size < 1 or radius_um < 0:
        raise ValueError('Invalid kNN parameters')
    n = len(x)
    values, valid = np.zeros(n), np.zeros(n, dtype=bool)
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        dx = xy[start:stop, None, 0] - xy[None, :, 0]
        dy = xy[start:stop, None, 1] - xy[None, :, 1]
        eligible = dx * dx + dy * dy > radius_um * radius_um
        eligible[np.arange(stop - start), np.arange(start, stop)] = False
        ok = eligible.sum(axis=1) >= k
        if ok.any():
            similarity = x[start:stop] @ x.T
            similarity[~eligible] = -np.inf
            chosen = np.partition(similarity[ok], n-k, axis=1)[:, -k:]
            values[start:stop][ok] = (1 - np.clip(chosen, -1, 1)).mean(axis=1)
            valid[start:stop] = ok
    return values, valid


def smooth(values, xy_um, radius_um, chunk_size):
    if radius_um <= 0:
        return values.copy()
    out = np.empty_like(values)
    for start in range(0, len(values), chunk_size):
        xy = xy_um[start:start+chunk_size]
        d2 = ((xy[:, None, :] - xy_um[None, :, :]) ** 2).sum(axis=2)
        neighbors = d2 <= radius_um ** 2
        out[start:start+len(xy)] = (neighbors @ values) / neighbors.sum(axis=1)
    return out


def foreignness(features, xy_um, prototypes, stated_organ_index, cfg):
    scores = cosine_scores(features, prototypes)
    own = scores[:, stated_organ_index]
    others = np.delete(scores, stated_organ_index, axis=1).max(axis=1)
    visual, valid = far_knn(features, xy_um, cfg['exclusion_radius_um'],
                             cfg['neighbors'], cfg['chunk_size'])
    q1, q2 = zscore(others - own), zscore(visual, valid)
    signals = {'organ_only': q1, 'visual_only': q2,
               'combined': q1 + cfg['visual_weight'] * q2}
    signals = {name: smooth(v, xy_um, cfg['smoothing_radius_um'], cfg['chunk_size'])
               for name, v in signals.items()}
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
    if not np.allclose(grid_float, grid, atol=1e-3):
        raise ValueError('Donor coordinates must lie on the stated regular grid')
    lookup = {tuple(p): i for i, p in enumerate(grid)}
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
    count = int(math.ceil(fraction * n / (1 - fraction))) if fraction else 0
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
