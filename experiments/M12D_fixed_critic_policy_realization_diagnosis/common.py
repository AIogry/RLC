"""Study-specific orchestration helpers; no training or Git calls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from impls.diagnostics.checkpoints import actor_sources, load_protocol, provenance_summary
from impls.main import _make_config, _parse_args
from impls.utils.env_utils import make_env_and_datasets
from impls.utils.reproducibility import derive_seed

HERE = Path(__file__).resolve().parent


def protocol_from_arg(path=None):
    return load_protocol(HERE / 'protocol.yaml' if path is None else path)


def load_env_dataset(protocol, seed):
    return make_env_and_datasets(
        protocol['environment'], seed=derive_seed(seed, 2),
        dataset_seed=derive_seed(seed, 1), dataset_dir=protocol['dataset_root'],
    )


def source_config(protocol, actor_name):
    from impls.experiment import prepare_run_design
    source = actor_sources(protocol)[actor_name]
    _, configuration = prepare_run_design(source.study, source.config)
    config = _make_config(_parse_args(['--agent', 'crl']), configuration=configuration)
    return config, source, configuration


def provenance_for_seed(protocol, seed):
    return provenance_summary(protocol, seed=seed, run_root=protocol['run_root'])


def task_names_from_env(env, task_ids):
    infos = getattr(env, 'task_infos', None) or getattr(getattr(env, 'unwrapped', None), 'task_infos', None)
    if infos is None:
        raise AttributeError('No formal task_infos')
    return {
        int(task_id): str(infos[int(task_id) - 1].get('task_name', f'task_{task_id}'))
        for task_id in task_ids
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as file:
        json.dump(value, file, indent=2, sort_keys=True, default=_json_default)
        file.write('\n')


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)
