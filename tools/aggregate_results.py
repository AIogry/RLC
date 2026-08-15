"""Aggregate a Study manifest by configuration and environment."""

import argparse

from impls.experiment import aggregate_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--metric', default='final_success')
    parser.add_argument('--output', default=None)
    args = parser.parse_args(argv)
    output = aggregate_manifest(args.manifest, args.output, metric=args.metric)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
