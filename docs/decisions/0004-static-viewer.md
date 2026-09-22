# 0004: A static viewer over precomputed results

**Status:** Accepted

## Context

Results should be easy to look at and to share, including publicly, without
running a service. Blender was considered for rendering and rejected for the
interactive viewer: it renders offline.

## Decision

The viewer is a static web page (plain HTML, CSS and JavaScript, three.js from a
CDN, no build step) shipped inside the package. `cosseratbench view` serves it
locally over a results directory; `cosseratbench site OUT` writes it as files for
any static host. Trajectories are converted to raw float32 for the browser; the
`.npz` files remain the canonical results.

## Consequences

- Anything interactive must come from precomputed runs. Adjustable conditions
  are sweeps computed ahead of time and selected in the page
  ([0008](0008-solver-options-and-variations.md)); running solvers live would
  need a server and is out of scope for now.
- Viewing needs a network connection for three.js.
- Results directories are self-describing (`experiment.json` beside each
  experiment's runs), so results can be viewed without the code that made them.
