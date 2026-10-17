"""Optional CPU text-vector export using a local checkout of the organizer adapters.

Not executed by the smoke workflow. Requires permitted model weights and packages.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--organizer-repo', required=True)
    parser.add_argument('--model', choices=['conch','musk','quiltnet','plip'], required=True)
    parser.add_argument('--prompts', default='examples/prompts.json')
    parser.add_argument('--encoder-fingerprint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hf-access-approved', action='store_true')
    parser.add_argument('--allow-model-download', action='store_true')
    args = parser.parse_args()
    if args.model in ('conch','musk') and not args.hf_access_approved:
        parser.error('Your own approved model access is required')
    target = Path(args.output)
    if target.exists():
        raise FileExistsError('Use a new output path')
    bank = json.loads(Path(args.prompts).read_text())['prompts']
    labels = list(bank)
    if len(labels) < 2 or len({len(v) for v in bank.values()}) != 1 or not bank[labels[0]]:
        raise ValueError('Use at least two classes and equal nonzero prompt counts')
    # Enforced before model/torch imports; no CUDA devices used by this process.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if not args.allow_model_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    repo = Path(args.organizer_repo).resolve()
    module_file = repo / 'models.py'
    spec = importlib.util.spec_from_file_location('organizer_adapters', module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    adapter = module.resolve_adapter(args.model, device='cpu')
    if adapter.encode_texts is None:
        raise ValueError('Adapter lacks text encoding')
    prompts = [p for label in labels for p in bank[label]]
    features = np.asarray(adapter.encode_texts(prompts), dtype=np.float32)
    if features.ndim != 2 or len(features) != len(prompts) or not np.isfinite(features).all():
        raise ValueError('Invalid adapter text output')
    np.savez_compressed(target, prompt_features=features.reshape(len(labels),len(bank[labels[0]]),-1),
        labels=np.array(labels), model_id=args.model, encoder_fingerprint=args.encoder_fingerprint,
        adapter_source_sha256=hashlib.sha256(module_file.read_bytes()).hexdigest(),
        prompts_sha256=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest())
    print(f'CPU text features saved to {target}. Run make-text-bank next.')


if __name__ == '__main__':
    main()
