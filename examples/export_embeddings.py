"""Import export_models from an encoder job after image/text embeddings exist.

No models, datasets, patient IDs, labels, or scale metadata are generated here.
The caller supplies NumPy-compatible arrays and genuine source metadata.
"""
from floaterfinder.intake import CacheWriter


def export_models(output, models):
    """Each model supplies its text bank and an iterable of original slide exports.

    models = [{
        'model_id': ..., 'encoder_fingerprint': ...,
        'labels': organ_labels, 'prototypes': projected_text_vectors,
        'slides': iterable_of_slide_dicts,
    }]
    A slide dictionary has features, xy_um, tile_ids, metadata, and optionally
    image_sha256. Metadata is defined in docs/CACHE_CONTRACT.md. The iterable
    can yield slides as encoding finishes; no whole-cohort feature list needed.
    """
    with CacheWriter(output) as writer:
        for model in models:
            writer.add_model(model['model_id'], model['encoder_fingerprint'],
                             model['labels'], model['prototypes'])
            for slide in model['slides']:
                writer.add_slide(**slide)
    return writer.root / 'dataset.json'
