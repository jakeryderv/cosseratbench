# 0007: Validation and exploration experiments

**Status:** Accepted

## Context

The first four experiments were each built as precise validations against an
analytical or high-accuracy reference. That anchors the project's credibility,
but it is expensive: the twist experiment needed a purpose-built measurement to
reach half a percent. Most of the scenarios the project is for (knots, winding,
cable routing) have no reference at all, and the project's aim is insight into
what works, what is difficult and what breaks, not a single score.

## Decision

Experiments are of two kinds.

- **Validation** experiments isolate one behaviour and score it against a
  reference. They are few, precise, and kept where a reference comes cheaply.
  Currently: catenary, cantilever, pendulum, twist.
- **Exploration** experiments show behaviour in harder or combined situations.
  They carry observations instead of a score: penetration, unwanted stretch,
  energy drift, runtime, whether and how a run failed. Currently: capstan.

## Consequences

- New experiments do not need a reference to be worth adding.
- Validation experiments keep their reference and their measured uncertainty,
  documented with the experiment.
- Observations common to many experiments (penetration, stretch, energy) should
  be shared metrics rather than rewritten per experiment.
