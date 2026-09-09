# M22 — OGBench Puzzle Baselines in Unified RLC

本文件是 source/config/dataset gate 的可追溯记录，不是训练结果。M22 使用官方 OGBench Puzzle 超参数，但在统一 RLC runtime 中执行；不声称六个算法均为 byte-for-byte upstream implementation。
本次 correction 轮次已完成 CRL 可执行梯度流复核，并将用户明确冻结的 GCIVL/GCIQL post-gradient target 更新记录为非阻塞、已披露的 RLC variant；本轮仍未创建 Study 或启动训练。

## Provenance

- Upstream repository: `https://github.com/seohongpark/ogbench`
- Local checkout: `/home/eai/Research/offline_rl_baselines/ogbench`
- Upstream version/tag: `1.2.1` / `v1.2.1`
- Upstream commit: `1d4140997f60c52c6fb0702ec100dc988b18c548`
- RLC starting HEAD: `824429a431e7350a3020f9d277f54bc82f02eeb3`
- RLC starting branch: `main`
- RLC starting working tree: `clean`
- RLC audit invocation HEAD: `824429a431e7350a3020f9d277f54bc82f02eeb3`

## Official Puzzle overrides

| Environment | Algorithm | Override | eval_episodes | Source line | Status |
| --- | --- | --- | ---: | ---: | --- |
| `puzzle-3x3-play-v0` | `gcbc` | `{}` | 50 | 730 | pass |
| `puzzle-3x3-play-v0` | `gcivl` | `{'alpha': 10}` | 50 | 732 | pass |
| `puzzle-3x3-play-v0` | `gciql` | `{'alpha': 1}` | 50 | 734 | pass |
| `puzzle-3x3-play-v0` | `qrl` | `{'alpha': 0.3}` | 50 | 736 | pass |
| `puzzle-3x3-play-v0` | `crl` | `{'alpha': 3}` | 50 | 738 | pass |
| `puzzle-3x3-play-v0` | `hiql` | `{'high_alpha': 3, 'low_alpha': 3, 'subgoal_steps': 10}` | 50 | 740 | pass |
| `puzzle-4x4-play-v0` | `gcbc` | `{}` | 50 | 743 | pass |
| `puzzle-4x4-play-v0` | `gcivl` | `{'alpha': 10}` | 50 | 745 | pass |
| `puzzle-4x4-play-v0` | `gciql` | `{'alpha': 1}` | 50 | 747 | pass |
| `puzzle-4x4-play-v0` | `qrl` | `{'alpha': 0.3}` | 50 | 749 | pass |
| `puzzle-4x4-play-v0` | `crl` | `{'alpha': 3}` | 50 | 751 | pass |
| `puzzle-4x4-play-v0` | `hiql` | `{'high_alpha': 3, 'low_alpha': 3, 'subgoal_steps': 10}` | 50 | 753 | pass |
| `puzzle-4x5-play-v0` | `gcbc` | `{}` | 50 | 756 | pass |
| `puzzle-4x5-play-v0` | `gcivl` | `{'alpha': 10}` | 50 | 758 | pass |
| `puzzle-4x5-play-v0` | `gciql` | `{'alpha': 1}` | 50 | 760 | pass |
| `puzzle-4x5-play-v0` | `qrl` | `{'alpha': 0.3}` | 50 | 762 | pass |
| `puzzle-4x5-play-v0` | `crl` | `{'alpha': 3}` | 50 | 764 | pass |
| `puzzle-4x5-play-v0` | `hiql` | `{'high_alpha': 3, 'low_alpha': 3, 'subgoal_steps': 10}` | 50 | 766 | pass |
| `puzzle-4x6-play-v0` | `gcbc` | `{}` | 50 | 769 | pass |
| `puzzle-4x6-play-v0` | `gcivl` | `{'alpha': 10}` | 50 | 771 | pass |
| `puzzle-4x6-play-v0` | `gciql` | `{'alpha': 1}` | 50 | 773 | pass |
| `puzzle-4x6-play-v0` | `qrl` | `{'alpha': 0.3}` | 50 | 775 | pass |
| `puzzle-4x6-play-v0` | `crl` | `{'alpha': 3}` | 50 | 777 | pass |
| `puzzle-4x6-play-v0` | `hiql` | `{'high_alpha': 3, 'low_alpha': 3, 'subgoal_steps': 10}` | 50 | 779 | pass |

Official state Puzzle source matches the required values: GCBC has no Puzzle-specific override; GCIVL alpha=10.0; GCIQL alpha=1.0; QRL alpha=0.3; CRL alpha=3.0; HIQL high_alpha=3.0, low_alpha=3.0, subgoal_steps=10.

## Baseline provenance classification

All baseline hyperparameters are authoritative official OGBench values. Implementation semantics are classified per algorithm; GCIVL and GCIQL intentionally retain the current RLC post-gradient target Polyak update.

| Algorithm | hyperparameter_provenance | implementation_semantics | Documented difference |
| --- | --- | --- | --- |
| `gcbc` | `official_ogbench` | `upstream_semantic_match` | — |
| `gcivl` | `official_ogbench` | `rlc_variant_documented` | post-gradient target Polyak update |
| `gciql` | `official_ogbench` | `rlc_variant_documented` | post-gradient target Polyak update |
| `qrl` | `official_ogbench` | `upstream_semantic_match` | — |
| `crl` | `official_ogbench` | `upstream_semantic_match` | — |
| `hiql` | `official_ogbench` | `upstream_semantic_match` | — |

## Canonical semantic gate

| Algorithm | Field | Status | Upstream observation | RLC observation |
| --- | --- | --- | --- | --- |
| `gcbc` | `RLC_computation_extension_when_disabled` | **infrastructure_only_difference** | `no computation-slot ontology in upstream` | `disabled slots resolve to None and select the original MLP path` |
| `gcbc` | `evaluation_device_and_seed_plumbing` | **infrastructure_only_difference** | `upstream main defaults eval_on_cpu=1 and uses process RNG` | `RLC uses explicit deterministic seed streams; JAX policy inference follows selected backend` |
| `gcivl` | `target_network_semantics` | **rlc_variant_documented** | `{'semantics': 'pre_gradient_online', 'line': 116}` | `{'semantics': 'post_gradient_online', 'line': 111}` |
| `gcivl` | `RLC_computation_extension_when_disabled` | **infrastructure_only_difference** | `no computation-slot ontology in upstream` | `disabled slots resolve to None and select the original MLP path` |
| `gcivl` | `evaluation_device_and_seed_plumbing` | **infrastructure_only_difference** | `upstream main defaults eval_on_cpu=1 and uses process RNG` | `RLC uses explicit deterministic seed streams; JAX policy inference follows selected backend` |
| `gciql` | `target_network_semantics` | **rlc_variant_documented** | `{'semantics': 'pre_gradient_online', 'line': 151}` | `{'semantics': 'post_gradient_online', 'line': 154}` |
| `gciql` | `RLC_computation_extension_when_disabled` | **infrastructure_only_difference** | `no computation-slot ontology in upstream` | `disabled slots resolve to None and select the original MLP path` |
| `gciql` | `evaluation_device_and_seed_plumbing` | **infrastructure_only_difference** | `upstream main defaults eval_on_cpu=1 and uses process RNG` | `RLC uses explicit deterministic seed streams; JAX policy inference follows selected backend` |
| `qrl` | `RLC_computation_extension_when_disabled` | **infrastructure_only_difference** | `no computation-slot ontology in upstream` | `disabled slots resolve to None and select the original MLP path` |
| `qrl` | `evaluation_device_and_seed_plumbing` | **infrastructure_only_difference** | `upstream main defaults eval_on_cpu=1 and uses process RNG` | `RLC uses explicit deterministic seed streams; JAX policy inference follows selected backend` |
| `crl` | `ddpgbc_critic_gradient_in_actor_loss` | **match_same_gradient_semantics_different_implementation_style** | `{'semantics': 'joint_actor_and_critic_gradient', 'branch_line': 102, 'evidence_line': 76}` | `{'semantics': 'critic_frozen_inside_actor_loss', 'branch_line': 91, 'evidence_line': 107}` |
| `crl` | `RLC_computation_extension_when_disabled` | **infrastructure_only_difference** | `no computation-slot ontology in upstream` | `disabled slots resolve to None and select the original MLP path` |
| `crl` | `evaluation_device_and_seed_plumbing` | **infrastructure_only_difference** | `upstream main defaults eval_on_cpu=1 and uses process RNG` | `RLC uses explicit deterministic seed streams; JAX policy inference follows selected backend` |
| `hiql` | `RLC_computation_extension_when_disabled` | **infrastructure_only_difference** | `no computation-slot ontology in upstream` | `disabled slots resolve to None and select the original MLP path` |
| `hiql` | `evaluation_device_and_seed_plumbing` | **infrastructure_only_difference** | `upstream main defaults eval_on_cpu=1 and uses process RNG` | `RLC uses explicit deterministic seed streams; JAX policy inference follows selected backend` |

### Original static evidence and correction

The previous CRL material-difference classification is retained as historical evidence, but it was based only on source syntax and is superseded by the executable gradient-flow audit.
- `gcivl.target_network_semantics`: previous status=`material_difference`, upstream=`{'semantics': 'pre_gradient_online', 'line': 116}`, RLC=`{'semantics': 'post_gradient_online', 'line': 111}`.
- `gciql.target_network_semantics`: previous status=`material_difference`, upstream=`{'semantics': 'pre_gradient_online', 'line': 151}`, RLC=`{'semantics': 'post_gradient_online', 'line': 154}`.
- `crl.ddpgbc_critic_gradient_in_actor_loss`: previous status=`material_difference`, upstream=`{'semantics': 'joint_actor_and_critic_gradient', 'branch_line': 102, 'evidence_line': 76}`, RLC=`{'semantics': 'critic_frozen_inside_actor_loss', 'branch_line': 91, 'evidence_line': 107}`.

### Executable CRL DDPG+BC gradient-flow audit

The audit isolates `actor_loss`, differentiates with respect to the full network parameter tree, and separately measures actor/critic subtrees and the action derivative of the critic Q path.

| Implementation | actor-loss | actor grad L2 | critic grad L2 | critic max abs | actor > 0 | critic numerical zero | dQ/da L2 | dQ/da active | Status |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| `upstream` | 1.033530951 | 1.951093197 | 0.000000000 | 0.000000000 | True | True | 1.626919746 | True | True |
| `rlc` | 1.033530951 | 1.951093197 | 0.000000000 | 0.000000000 | True | True | 1.626919746 | True | True |

Correction result: `crl.ddpgbc_critic_gradient_in_actor_loss` is `match_same_gradient_semantics_different_implementation_style`; both implementations have positive actor gradients, numerically zero critic-subtree gradients, and an active dQ/da pathway.

Current blocking material differences:

- None after the executable CRL correction and explicit GCIVL/GCIQL variant disclosure.

Gate decision: **pass**. PASS: no undisclosed or newly discovered material semantic difference detected; the two user-frozen RLC target-update variants are explicitly documented and non-blocking.

## Dataset audit summary

| Environment | Train SHA256 | Validation SHA256 | Train obs/action | Validation obs/action | Status |
| --- | --- | --- | --- | --- | --- |
| `puzzle-3x3-play-v0` | `bca9b1775ea5c01af5ac3f9b652709d341bddd17fe4b6368056566ac3c5fbb0f` | `897a3f491decf46d72ea73425dc9b2c969530eca190708d763e632850b856550` | 55/5 | 55/5 | pass / pass |
| `puzzle-4x4-play-v0` | `396a029c4c347498eed3e21cd85f99c72c56f786e84c29787715808f27cdfe8a` | `472324d739ed9d029600fd112738d2602b15a7a3039d03ce91265b65c5943d4d` | 83/5 | 83/5 | pass / pass |
| `puzzle-4x5-play-v0` | `7f25e3161280aa4ff1234a279ba7a0a4466df6a8e4f5af64fd2b1bfe8df892cc` | `26bfb8d59290be2838a4c8185fc8221355ff45027d4c4bff4e26f6f4a7c2624c` | 99/5 | 99/5 | pass / pass |
| `puzzle-4x6-play-v0` | `8d3de9cad77874484367d492540baac385cb487cf94b2649547770bf7d0b6c9f` | `5987ac87eb31ccd05acab8d8a6dac1662462fe9798b1b9e2269a16505adbb230` | 115/5 | 115/5 | pass / pass |

Dataset file hashes, byte sizes, transition counts, recovered episode counts, dtypes and exact NPZ members are in `M22_dataset_audit.json`.

Formal M22 training was NOT started automatically.
Git commit/push were NOT performed by Codex.
