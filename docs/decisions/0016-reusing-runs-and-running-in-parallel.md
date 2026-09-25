# 0016: Reusing runs, rescoring them, and running several at once

**Status:** Accepted

## Context

Nothing was reused: every `cosseratbench run` simulated everything it was asked
for again, even when nothing had changed, and a sweep ran one simulation after
another on a machine with many cores. MuJoCo's pile alone takes three minutes
per variant and ten on the stiffest rope, and a profile showed 97% of that time
inside MuJoCo's own step, so no rewrite of the adapter would help.

Two things made this more than adding a cache and a process pool.

A saved run is only reusable while everything its trajectory depended on is
unchanged, and that includes code: the adapter that translated the scenario, and
the solver library behind it.

Wall time is one of the results. Runs sharing a machine slow each other down,
badly when each also starts a pool of threads: dismech's catenary took ten
minutes beside two other runs against twenty seconds alone.

## Decision

- Every result records its **provenance**: a digest of the scenario, a digest of
  the adapter's source and of the scenario module every adapter builds from, and
  the solver library's version (an adapter's optional `backend_version`).
  Results also record the frames asked for.
- `cosseratbench run` reuses a saved run when its provenance, resolution, frames
  and solver options all match, and reports it as `[saved]`; `--force` reruns.
- Metrics are not part of what makes a run reusable. `cosseratbench rescore`
  recomputes metrics and observations from saved trajectories, which decision
  [0002](0002-metrics-judge-trajectories.md) makes sound. It refuses a run whose
  scenario the experiment no longer builds.
- `cosseratbench run -j N` runs up to N simulations at once, each in its own
  spawned process and each limited to one thread of numerical work unless the
  user has set otherwise. Every result records how many ran at once (`jobs`), and
  the viewer shows it beside the wall time.
- A run whose adapter raises an unexpected exception is reported as crashed and
  the rest carry on; the command fails at the end.

## Consequences

- Changes to code outside the adapter and the scenario module (the runner's
  divergence check, a solver library installed from the same version number) are
  not detected; `--force` covers them.
- A third-party adapter spread over several modules is digested by the module
  that defines its class only.
- On the pile's stiffness sweep on PyElastica, three at once took 58 s against
  82 s one after another, with each run's wall time unchanged to 0.5 s.
