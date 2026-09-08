# M20B Cube Entity-Structured Mixer Transfer Screen

对每个 Cube environment 单独比较 `S002 - B000`；不合并 raw success，不拟合实体数量到 gap 的回归，也不做 seed-0 显著性检验。

| Environment | Condition | Curve | final@1M | best | best step | last3 | normalized AUC | Flat headroom |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cube-double-play-v0 | B000 | missing | — | — | — | — | — | — |
| cube-double-play-v0 | S002 | missing | — | — | — | — | — | — |
| cube-single-play-v0 | B000 | missing | — | — | — | — | — | — |
| cube-single-play-v0 | S002 | missing | — | — | — | — | — | — |
| cube-triple-play-v0 | B000 | missing | — | — | — | — | — | — |
| cube-triple-play-v0 | S002 | missing | — | — | — | — | — | — |

## Within-environment contrasts

- `cube-single-play-v0`: raw Δ = —, best Δ = —, last3 Δ = —, AUC Δ = —.
- `cube-double-play-v0`: raw Δ = —, best Δ = —, last3 Δ = —, AUC Δ = —.
- `cube-triple-play-v0`: raw Δ = —, best Δ = —, last3 Δ = —, AUC Δ = —.

Task columns are retained under their exact authoritative `evaluation/<task_name>_success` names.

