"""Build a Study manifest from planned configurations and observed runs."""

import argparse

from impls.experiment import write_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, help='Path to study.yaml')
    parser.add_argument('--run-root', default='runs')
    parser.add_argument('--output', default=None)
    args = parser.parse_args(argv)
    output = write_manifest(args.study, args.run_root, args.output)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
