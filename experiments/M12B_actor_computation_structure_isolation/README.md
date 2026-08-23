# M12B Actor Computation Structure Isolation

M12B isolates actor computation structure under the same seed-matched frozen
CRL critic.  `SingleState` remains the only recurrent topology; `shared` and
`untied` are its parameter-sharing modes.  Residual FF is a `FeedForward`
topology composed with the generic `ResidualMLPStack` block/body.

The study has nine conceptual conditions. B001 and B004 reuse valid M12A
Stage2 attempt-1 anchors. The seven new conditions B002, B003, B005, B006,
B007, B008 and B009 produce exactly 21 new runs over seeds 0/1/2.

No formal training is started by this directory. Use `preflight.py` and the
sweep `--dry-run` before the user manually launches the formal command.
