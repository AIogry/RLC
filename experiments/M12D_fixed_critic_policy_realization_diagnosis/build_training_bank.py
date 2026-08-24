from __future__ import annotations

import argparse
from pathlib import Path

from impls.diagnostics.banks import build_training_support_bank, save_bank

from common import load_env_dataset, protocol_from_arg, provenance_for_seed, source_config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.yaml')))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batches', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args(argv)
    protocol = protocol_from_arg(args.protocol)
    _, dataset, _ = load_env_dataset(protocol, args.seed)
    config, source, _ = source_config(protocol, 'K1SN')
    provenance = provenance_for_seed(protocol, args.seed)
    spec = protocol['banks']['B_T']
    arrays, manifest, rows = build_training_support_bank(
        dataset, config, seed=args.seed,
        batches=spec['batch_count'] if args.batches is None else args.batches,
        batch_size=spec['batch_size'] if args.batch_size is None else args.batch_size,
        environment=protocol['environment'], dataset_root=protocol['dataset_root'],
        source_commit=protocol['source_commit'], provenance=provenance,
    )
    manifest['dataset_observation_shape'] = list(dataset['observations'].shape)
    manifest['dataset_observation_dtype'] = str(dataset['observations'].dtype)
    manifest['actor_origin_sampler_config'] = source.config
    artifact = save_bank(args.output, arrays, manifest, rows)
    print(f'B_T written: {artifact.root} hash={artifact.manifest["bank_hash"]}')


if __name__ == '__main__':
    main()

