"""Dependency-light logging for the first HIQL runtime slice."""

import csv
import os
from datetime import datetime


def _as_scalar(value):
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


class CsvLogger:
    """Append scalar metric rows to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.file = None
        self.writer = None
        self.header = None

    def log(self, row, step):
        row = {key: _as_scalar(value) for key, value in dict(row).items()}
        row['step'] = step
        if self.file is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self.file = open(self.path, 'w', newline='')
            self.header = list(row.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.header)
            self.writer.writeheader()
        for key in row:
            if key not in self.header:
                self.header.append(key)
        self.writer.writerow({key: row.get(key, '') for key in self.header})
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def get_exp_name(seed):
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'sd{seed:03d}_{suffix}'
