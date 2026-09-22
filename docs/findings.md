# Findings

What the experiments have shown about the solvers, and about measuring them.
Each finding says what happened, what kind of thing it is (see
[decision 0009](decisions/0009-kinds-of-outcome.md)), and the evidence. Numbers are
from PyElastica 1.0.0 and MuJoCo 3.13.0 at each experiment's default resolution
unless stated.

## MuJoCo

**Twisting a cable clamped at both ends diverges.** *Numerical breakdown.*
Once the cable holds about 5 rad of twist, the simulation blows up, whichever end
is driven and whether or not the weld holding the far end rotates. Twisting faster
brings it sooner: at the twist experiment's rate it diverges within 0.05 s. The
same cable twisted by a torque at a free end is fine to 12 rad, so the cable model
is not the cause; the weld constraints that clamp its ends are. Softer welds
survive only by letting the clamped end turn by 20% to over 100% of the twist,
so they do not clamp. MuJoCo can hold a far end only with such a constraint, so
it cannot run the twist experiment or, presumably, any experiment that twists a
clamped cable several turns.

**Other ways a held end breaks it.** *Numerical breakdown.* A sideways load on a
clamped end diverges within 0.05 s. Rotating a clamp in bending while both ends
are held also diverges; that demand is impossible for an inextensible cable, which
cannot shorten its chord to bend, so it may also count as not representable.

**Held ends drift unless constraints are stiff.** *Solver setting.* At MuJoCo's
default constraint softness, a pinned cable end drifted 3.4 mm under the cable's
weight; making the constraint as stiff as the step allows brought it to 1.8 µm.

**The cable cannot stretch.** *Not representable.* On the catenary this costs
1.2% of the sag, which stretch deepens. It also cannot show a cable bouncing along
its length when released.

**Stiff cables are slow.** *Cost.* The cable plugin integrates its stiffness
explicitly, so the time step shrinks with stiffness and element size. A rigid-rod
pendulum was rejected because it would take 6 to 20 minutes per run; the cantilever
at 80 elements took five minutes against PyElastica's eight seconds.

**A diverged run resets and carries on.** *Solver behaviour.* MuJoCo resets the
state and continues after a bad acceleration, so a diverged run looks like it
finished. The adapter checks MuJoCo's warning counter after every frame.

## PyElastica

**Twisted rods buckle about 1% early.** *Open question.* The twist experiment's
critical twist converges with resolution (0.978, 0.989, 0.991 of Greenhill's at
25, 50 and 100 elements) toward about 0.99. The measurement is good to about
±0.5%, and the physics Greenhill leaves out (shear, stretch) is too small to
explain the rest.

**Clamped ends make convergence first order.** *Discretisation.* The clamped
element is held rigid, so the cantilever's tip deflection error halves only as
the element count doubles: 7.3% at 20 elements, 1.8% at 80.

**Friction with obstacles is regularised.** *Solver model.* Friction is the lesser
of Coulomb friction and a viscous term whose coefficient the user picks; static
friction is imitated by making that coefficient large. A rope that should hold
still creeps instead. Contact between rods, or of a rod with itself, has no
friction at all.

**First runs include JIT compilation.** *Cost.* Numba compilation added about 14 s
to a first run against 0.5 s warm. The runner times runs after a short warm-up.

## Both solvers, and measuring them

**A cable released unstretched bounces, and the bounce pumps sideways vibration.**
*Physically plausible event.* An extensible cable released at rest length under
gravity oscillates along its length (25 Hz for the pendulum's cable). The varying
tension pumps sideways modes near half that frequency (parametric resonance),
which grew from any seed to about 0.65 mm. The fix was the scenario, not the
solver: the pendulum now starts in equilibrium
([decision 0005](decisions/0005-rods-can-start-stretched.md)).

**Stable time steps were underestimated.** *Adapter mistake.* Both adapters
missed the twisting mode, whose explicit stability limit is `dl / sqrt(G / rho)`,
independent of radius. MuJoCo diverged on the pendulum's thin cable until it was
included; it also explains an earlier catenary stability probe.

**Settling damping acts on absolute motion.** *Measurement pitfall.* The damping
adapters add in quasi-static scenarios slows every motion, including rigid ones:
a clamp turned 90° drags the rod behind it for seconds. Spinning a rod against it
pushes the rod sideways, so a rod twisted steadily left straight at 1.21 times the
critical twist. Holding the twist steady removes both effects.

**Timing zero crossings is fragile.** *Measurement pitfall.* A 0.65 mm vibration
riding on a 1.25 cm swing gave extra crossings and a 52% period error. Fitting a
sinusoid to every frame is robust to it.
