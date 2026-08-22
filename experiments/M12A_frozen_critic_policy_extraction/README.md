# M12A — Frozen-Critic Policy Extraction

M12A is a targeted mechanism experiment, not a benchmark-generalization table.
The primary study uses only `antmaze-large-navigate-v0` and three paired critic
seeds: `0, 1, 2`.

For each seed `c`, C001 trains the canonical feed-forward CRL critic for exactly
1,000,000 critic-only updates. C002 and C003 then use the exact same C001 seed-
matched `last@1M` critic checkpoint while training, respectively, the canonical
feed-forward actor and the SingleState K4 actor. The paired contrast is:

```text
Delta_c = J(SS actor against Q_phi_c*) - J(FF actor against Q_phi_c*)
```

The three configurations are expanded over seeds `[0, 1, 2]`, giving 9 formal
Runs: 3 critic-pretraining Runs plus 6 actor-extraction Runs. Seeds belong to
Runs, not Configurations.

Frozen critic selection is fixed-step only:

```yaml
rule: fixed_step
step: 1000000
role: last
best_selection: disabled
```

Stage 1 has true evaluation-disabled semantics (`eval_tasks=none`), no best
checkpoint, and a semantic last checkpoint at 1M. Stage 2 uses the canonical
CRL DDPG+BC actor objective, 20 episodes per evaluation task, temperature 0,
and primary endpoint `final@1M`.

The AntMaze-Giant extension is prespecified as confirmatory but is not an
active M12A-Core configuration and must not be launched in this round.
