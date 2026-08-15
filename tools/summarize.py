"""Summarize one run's existing eval.csv without changing raw artifacts."""

import argparse
import json

from impls.experiment import summarize_eval_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    parser.add_argument('--status', default='completed')
    args = parser.parse_args(argv)
    summary = summarize_eval_csv(f'{args.run_dir}/eval.csv', status=args.status)
    with open(f'{args.run_dir}/summary.json', 'w') as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write('\n')
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
