# M9B-1M TwoState convergence extension

## Scope

M9B-1M is a long-training diagnostic of the existing M9B TwoState
architecture. It changes only the training horizon relative to each historical
counterpart: 500,000 to 1,000,000 steps. It does not introduce a new HRM,
placement factor, credit rule, normalization recipe, tokenization, attention,
or adaptive-computation mechanism.

The formal matrix contains four runs only:

| config | algorithm/placement | schedule | historical counterpart |
|---|---|---:|---|
| M9B1M-C001 | CRL actor | H2L1 full BPTT | M9B-C001 |
| M9B1M-C002 | CRL actor | H2L6 full BPTT | M9B-C003 |
| M9B1M-C003 | HIQL high+low actor | H2L1 full BPTT | M9B-C013 |
| M9B1M-C004 | HIQL high+low actor | H2L6 full BPTT | M9B-C015 |

The only environment is `antmaze-large-navigate-v0`; the only training seed is
`0`. Vanilla CRL and HIQL are external immutable references, not configurations
of this Study.

## Historical audit

The historical M9B large/seed-0 runs were completed at 500k under commit
`f30b64bf81e1738235eef4f213d3019820ee918a`. Their relevant protocol was batch
size 1024, learning rate `0.0003`, log interval 5000, evaluation every 100k,
20 episodes per task, all five tasks, temperature 0, no Gaussian noise, and
video episodes 0. Historical numeric checkpoint saving used interval 500k.

Observed historical overall-success curves:

| counterpart | 100k | 200k | 300k | 400k | 500k | best | final |
|---|---:|---:|---:|---:|---:|---:|---:|
| M9B-C001 CRL H2L1 | 0.80 | 0.85 | 0.90 | 0.89 | 0.84 | 0.90 @ 300k | 0.84 |
| M9B-C003 CRL H2L6 | 0.83 | 0.82 | 0.81 | 0.91 | 0.89 | 0.91 @ 400k | 0.89 |
| M9B-C013 HIQL H2L1 | 0.72 | 0.93 | 0.88 | 0.95 | 0.91 | 0.95 @ 400k | 0.91 |
| M9B-C015 HIQL H2L6 | 0.74 | 0.92 | 0.84 | 0.90 | 0.85 | 0.92 @ 200k | 0.85 |

External vanilla references already exist and were not rerun:

| reference | source | steps | best | final | commit |
|---|---|---:|---:|---:|---|
| CRL vanilla | M9A-C002, large, seed 0 | 1M | 0.77 @ 900k | 0.62 | `f30b64bf81e1738235eef4f213d3019820ee918a` |
| HIQL vanilla | M9A-C001, large, seed 0 | 1M | 0.87 @ 1M | 0.87 | `f30b64bf81e1738235eef4f213d3019820ee918a` |

The reference runs use the same batch size, learning rate, evaluation cadence,
episode count, temperature, and dataset root. Their numeric checkpoint interval
was 1M, whereas M9B-1M intentionally uses 100k to preserve the late-training
trajectory.

## Formal protocol

- train from scratch at step 0; do not resume historical 500k checkpoints;
- train steps: 1,000,000;
- batch size: 1024;
- log interval: 5,000;
- evaluation interval: 100,000;
- evaluation tasks: all five;
- evaluation episodes: 20 per task;
- evaluation temperature: 0.0;
- evaluation Gaussian: null;
- video episodes: 0;
- numeric checkpoint interval: 100,000;
- save best and last checkpoints;
- best metric: `evaluation/overall_success`;
- tie rule: strict `>`; equal scores keep the earlier checkpoint.

The expected numeric checkpoints are 100k through 1M, including 500k, plus
semantic `best` and `last` checkpoints. Formal training is not started by this
repository preparation task.

## Accounting

For TwoState, the execution trace has:

```text
N_H = h_cycles
N_L = h_cycles * l_cycles
N_total = h_cycles * (l_cycles + 1)
```

Thus H2L1 is H=2, L=2, total=4; H2L6 is H=2, L=12, total=14. The generic
accounting helper was corrected so its execution counts agree with the topology
trace and existing runtime metadata. Forward computation semantics were not
changed.

## Post-training boundary

After all four runs finish, first inspect training-time curves and compare
500k with 1M, best step, best training-time score, and last@1M against the
external references. Best/last 100-episode reevaluation is a separate later
step and must be approved before execution.
