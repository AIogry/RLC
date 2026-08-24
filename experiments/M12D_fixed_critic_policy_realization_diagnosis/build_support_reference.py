from __future__ import annotations

import argparse
from pathlib import Path

from impls.diagnostics.banks import save_bank
from impls.diagnostics.support import build_support_reference

from common import load_env_dataset, protocol_from_arg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=str(Path(__file__).with_name('protocol.yaml')))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-states', type=int, default=None)
    args = parser.parse_args(argv)
    protocol = protocol_from_arg(args.protocol)
    _, dataset, _ = load_env_dataset(protocol, args.seed)
    max_states = protocol['support_proxy']['reference_states'] if args.max_states is None else args.max_states
    arrays, manifest = build_support_reference(
        dataset['observations'], max_states=max_states,
        environment=protocol['environment'], dataset_root=protocol['dataset_root'],
        source_commit=protocol['source_commit'],
    )
    artifact = save_bank(args.output, arrays, manifest)
    print(f'Support reference written: {artifact.root} hash={artifact.manifest["bank_hash"]}')


if __name__ == '__main__':
    main()

