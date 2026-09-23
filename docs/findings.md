# Findings

What the experiments have shown about the solvers, and about measuring them.
Each finding says what happened, what kind of thing it is (see
[decision 0009](decisions/0009-kinds-of-outcome.md)), and the evidence. Numbers are
from PyElastica 1.0.0 and MuJoCo 3.13.0 at each experiment's default resolution
unless stated.

## MuJoCo

**Twisting a cable clamped at both ends diverges.** *Numerical breakdown.*
Once the cable holds about 5 rad of twist, the simulation blows up, whichever end
is driven and whether or not the weld holding the far end rotates; in the twist
experiment that is about 2 s into the ramp, at every tension and resolution
swept. The same cable twisted by a torque at a free end is fine to 12 rad, so the
cable model is not the cause; the weld constraints that clamp its ends are.
(An earlier note said twisting fast made it fail within 0.05 s. That time was
read after MuJoCo had reset its clock on diverging; it was the same 5 rad limit.) Softer welds
survive only by letting the clamped end turn by 20% to over 100% of the twist,
so they do not clamp. MuJoCo can hold a far end only with such a constraint, so
it cannot run the twist experiment or, presumably, any experiment that twists a
clamped cable several turns.

**Other ways a held end breaks it.** *Numerical breakdown.* A sideways load on a
clamped end diverges. Rotating a clamp in bending while both ends are held also
diverges; that demand is impossible for an inextensible cable, which
cannot shorten its chord to bend, so it may also count as not representable.

**Held ends drift unless constraints are stiff.** *Solver setting.* At MuJoCo's
default constraint softness, a pinned cable end drifted 3.4 mm under the cable's
weight; making the constraint as stiff as the step allows brought it to 1.8 µm.

**The cable cannot stretch.** *Not representable.* On the catenary this costs
1.2% of the sag, which stretch deepens; 10% when the cable is ten times softer,
and 7.4% when it is nearly taut (span 0.95 m for a 1 m cable), where sag depends
most on stretch. It also cannot
show a cable bouncing along its length when released.

**It tolerates larger time steps than its adapter estimates.** *Solver behaviour.*
At twice the adapter's time step (`time_step_scale=2`) MuJoCo still gives the same
answers on the catenary, cantilever and pendulum, in half the time; at four times
it diverges. PyElastica diverges at twice on every experiment, within the first
frame. The adapters' shared estimate of the
stable step is therefore conservative for MuJoCo's `implicitfast` integrator.

**Stiff cables are slow.** *Cost.* The cable plugin integrates its stiffness
explicitly, so the time step shrinks with stiffness and element size. A rigid-rod
pendulum was rejected because it would take 6 to 20 minutes per run; the cantilever
at 80 elements took five minutes against PyElastica's eight seconds.

**Its default friction cone lets ropes slip early.** *Solver setting.* MuJoCo
approximates the friction cone with a pyramid by default, which allows as little as
mu / sqrt(2) of friction for sliding askew to the contact's axes. On the capstan
the rope slid off at 95% of the overhang that holds. With elliptic cones it held
at 95% and slid at 110%, matching the equilibrium, so the adapter uses them.

**Cables that start straight and then curve are built broken.** *Solver bug.*
MuJoCo frames a cable's first segment from the bend at its first vertex. A cable
whose first two edges are in line has no bend there, the frame is undefined, and
where the cable later curves two neighbouring segments come out half a turn apart;
the simulation diverges at once. Arcs and straight cables are unaffected, which is
why no earlier experiment met it. The adapter nudges the third vertex a millionth
of a segment toward the rod's reference normal, which defines the frame.

**Damping from last step's mass matrix spins up free cables.** *Adapter mistake.*
The adapter's settling damping is minus a rate times the mass matrix times the
velocities. Computed with the mass matrix of the previous step, it made any cable
with a free end diverge within a second at ordinary time steps: a thin cable's mass
matrix is badly conditioned (a segment's inertia about its own axis is about 10^4
times smaller than its other terms), so small changes between steps turn into
large spurious spin. Held ends had hidden it. Computing it between MuJoCo's two
half steps, from the current matrix, fixed it: a damped cable now falls at exactly
its terminal speed.

**Its cable contact is soft by default too.** *Solver setting.* On the crossing
experiment a rope sank 0.85 of a radius into the rope it was resting on, and
carried 1.3% too little force through the touching point. Making cable contact as
stiff as the step allows, as the adapter already did for obstacles, left 0.003 of
a radius and 0.05% -- the closest either solver gets to the right contact force.
It moved the capstan's creep from 2.5 mm to 1.8 mm.

**Its cable cannot stretch, and on the crossing experiment that is the whole
error.** *Not representable.* MuJoCo gets the contact force nearly exactly right
there (lift error 0.05%, against PyElastica's 0.3%) but is 1.1% of a rope length
off in shape, which is the 1% of stretch it cannot do.

**Two ropes crossing diverge above about 50 elements.** *Numerical breakdown.*
The crossing experiment runs at 30 and 51 elements and blows up at 100, around
0.35 s in, while PyElastica runs the same scenario at every resolution tried.

**A diverged run resets and carries on.** *Solver behaviour.* MuJoCo resets the
state and continues after a bad acceleration, so a diverged run looks like it
finished. The adapter checks MuJoCo's warning counter after every frame.

**A pile costs it minutes.** *Cost.* The pile experiment, a 1.5 m rope in
self-contact on a floor, takes MuJoCo 173 s at 100 elements and 1107 s at 150,
against PyElastica's 18 s and 34 s. The step count barely changes between the
two resolutions, so the cost per step grows about five times for half again as
many elements: the implicit integrator's dense derivatives, and a contact solver
with a hundred contacts, both grow faster than the chain does.

**Its rope coils tightly.** *Solver behaviour.* Fed onto the floor at 0.4 m/s,
MuJoCo's rope winds into a spiral of three turns about 10 cm across, lying almost
flat (1.5 diameters tall, 5 places where it touches itself); PyElastica's makes
looser loops 16 cm across, also flat, touching itself in 8 places. Neither is
wrong: there is no reference, and the two differ in stretch, in rod-rod friction
(PyElastica has none) and in contact stiffness. They agree that a rope this soft
does not heap at this speed.

## PyElastica

**Twisted rods buckle about 1% early.** *Open question.* The twist experiment's
critical twist converges with resolution (0.978, 0.989, 0.991 of Greenhill's at
25, 50 and 100 elements) toward about 0.99, and is 0.989 and 0.983 at a tenth
and three times the usual tension. The measurement is good to about
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

**Rods slide freely over each other.** *Not representable.* `RodRodContact` and
`RodSelfContact` take a stiffness and a damping and nothing else, so rods that
touch have no friction however rough the scenario says they are. Experiments that
need rods to grip declare `ROD_FRICTION`, which this solver does not claim.

**Self-contact costs 2 to 5 times the rest of the step.** *Cost.* Every pair of a
rod's elements is checked every step, with no broadphase to rule out a rod that
cannot reach itself: with it on, the catenary went from 0.7 s to 3.1 s and the
capstan from 2.5 s to 12.4 s, with identical answers. Scenarios say whether their
rods can reach themselves so that this is only paid where it buys something.

**On a curved obstacle its friction holds less than Coulomb's, at coarse
resolution.** *Open question.* On the capstan at 100 elements (about 23 on the
cylinder), PyElastica's rope slides off at 70% of the overhang that holds, where
MuJoCo holds to 95%. At 200 elements it holds to at least 80%. Stronger friction
makes it worse, not better: at mu = 0.6 it slides below 80%. The cause is not
yet known.

**A rod meeting a plane end-on sinks half an element into it.** *Solver model.*
PyElastica's plane contact acts at element centres, so a rope lowered onto the
floor tip first is not stopped until its last element's centre reaches the
surface, and its end node is then half an element below it: 2 radii at 1 cm
elements, 3 at 1.5 cm, which is what `max_penetration` reports on the pile
experiment (1.98 and 2.96). The rope rides back up once it buckles over. Contact
with a cylinder is at nodes and does not do this.

**First runs include JIT compilation.** *Cost.* Numba compilation added about 14 s
to a first run against 0.5 s warm. The runner times runs after a short warm-up.

## dismech

**Gravity is applied twice.** *Solver bug.* At the pinned commit (3d83a30),
dismech-python subtracts gravity from the residual once before the contact
forces and once after, so a free rod falls at exactly 2 g: 2.44 m in half a
second against 1.23. Its pendulum swung 29% off period. The adapter puts one
gravity back, and a free rod falls as it should.

**The shape it is built in is stress-free.** *Adapter mistake.* dismech takes
every spring's natural strain (stretch, curvature, twist) from the state its
time stepper is constructed with. The adapter first built each rod
unstretched but bent along its starting shape, as it builds PyElastica's, which
always takes a straight rest shape whatever it is given. In dismech that made
the catenary's starting arc the cable's natural curvature: its bend angle rose
toward each pin instead of falling to zero, the sag came out 0.4% shallow at
every resolution, and the error scaled with bending stiffness over weight. Built
straight and unstretched and only then moved, the cable shows the same
moment-free boundary layer at its pins as PyElastica (bend angles within 0.01°)
and the sag is within 0.018% of the reference. Built after the move, the stepper
had also made the pendulum's hanging stretch natural.

**Friction on the floor cannot hold anything still.** *Solver bug.* At the pinned
commit, dismech's floor friction fails in two ways. A rod only partly on the floor
raises an IndexError: the contact force is listed for the touching nodes and the
friction pairs it with every node's velocity. The adapter corrects that. Then,
whenever friction must stick, the Newton solve does not converge in 50
iterations, on the very first step: a rod on a 20° slope with mu = 0.5 (which
should hold) fails at t = 0 whatever the stick-slip velocity tolerance (5e-5 to
1e-2 m/s), while with mu = 0.1 it slides as it should. Its sticking branch also
passes the same derivative twice where the sliding branch passes two different
ones, but correcting that does not make it converge. So a rope landing on a floor
with friction, the pile experiment, is reported as diverged for dismech.

**Crossed ropes drift out of symmetry after settling.** *Open question.* On the
crossing experiment (51 elements) dismech's ropes settle at the right gap by 2 s,
then around 3.5 s the lower rope's middle drifts 3.5 mm out of its plane, growing
exponentially, and around 7.5 s the upper rope's drifts 2.6 mm; the run ends still
moving (settling residual 9e-3) with lift error 2.6% and shape error 6.7e-3.
PyElastica holds the lower rope in plane to 1e-15. Whether it depends on the time
step, the contact stiffness the adapter picks, or dismech's contact model is not
yet known.

**Its pendulum matches PyElastica's.** *Solver behaviour.* With Newmark-beta,
which conserves energy, the pendulum's period is 1.4e-4 off the reference at
50 elements, as PyElastica's is, and its amplitude changes by 9e-6 over the run.

## References

**The catenary's reference ignores bending, which matters on slack spans.**
*Reference limit.* At a 0.6 m span both solvers sit well off the elastic
catenary (sag error 0.77% for PyElastica, 1.3% for MuJoCo, against 0.014% and
1.2% at 0.8 m). Halving the cable's radius, which cuts bending stiffness four
times relative to weight, drops PyElastica's error to 0.24% at 0.6 m and leaves
0.8 m unchanged, so the gap is the reference's, not the solvers'.

**A rope with a point load on it cannot be treated as perfectly flexible.**
*Reference limit.* Bending rounds off the kink under the load over a length
sqrt(EI/T), about 3 cm for the crossing experiment's rope, which lifts the middle
by 6 mm, more than the rope's radius. Two ropes resting on each other would
then need 32% more force between them than the flexible answer says. The crossing
experiment's reference solves the elastica instead. The same correction exists in
the catenary, where pinned ends carry no moment and it is only 0.1 mm: PyElastica's
sag moves toward the bending answer and away from the flexible one as resolution
rises (6.5e-5 m off at 50 elements, 1.2e-5 at 200).

**The plain capstan equation ignores the rope lying on the cylinder.** *Reference
limit.* Its weight presses it on and adds friction. For the capstan experiment's
rope, a 5 cm cylinder under a 15 cm overhang, the overhang that holds is 1.26 times
what the plain equation predicts at mu = 0.3. The experiment's reference includes
it (checked against direct integration), and MuJoCo with elliptic friction cones
agrees.

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

**A buckled rod can come back.** *Measurement pitfall.* Without self-contact a
buckled rod can loop, pass through itself and straighten again, shedding a turn
of twist. Growth measured on frames after that is meaningless; at three times the
usual tension it made the twist experiment read 5.1% where the rods' actual
growth gives 1.7%. Growth is now measured only until a rod first leaves the
exponential range.

**A node sitting exactly where two rods touch skews the contact.**
*Measurement pitfall.* With an even number of elements each rope in the crossing
experiment has a node exactly at the crossing, the two segment ends meet head on,
and the penalty force comes out with a sideways component: it pushed the lower
rope 2.9 mm out of the plane it should stay in. One element more and the ropes
touch mid-segment, the lower rope stays in plane to 1e-15, and the error drops
15-fold -- shape 2.6e-4 against 4.0e-3, lift 1.3e-4 against 2.1e-3, at about 100
elements. The experiment runs at an odd count for that reason, and its resolution
variants show the effect.

**A polyline cuts the corner under a point load.** *Measurement pitfall.* Two
ropes placed one diameter apart at the crossing have segments that already
overlap, by 1.5 radii at 51 elements, because each rope's polyline cuts across
the kink where the load presses on it. The run then opens with a contact impulse,
and the overlap reported is the starting geometry rather than anything a solver
did -- both solvers returned the same number to four figures, which is what gave
it away. The ropes now start a centimetre clear and settle into contact.

**Penetration measured on segments reflects resolution, not contact.**
*Measurement pitfall.* A straight segment between two nodes on a curved obstacle
cuts inside it by the chord's bulge. Measured there, both solvers showed the same
penetration at every overhang on the capstan. It is measured at nodes instead, and
the capstan's rope is drawn so its segments rest on the cylinder.

**A flat imperfection leaves a symmetric solver flat.** *Measurement pitfall.*
The pile experiment seeds the hanging rope with a small bow so that it folds a
definite way. Seeded with a bow in one plane, PyElastica's rope folded back and
forth in that plane for the whole run, never leaving it by more than rounding,
and stacked into a zigzag ten diameters tall that looked like a heap in the
metrics; MuJoCo's rope left the plane at once and coiled. The seed now bows in
two directions at once, and both solvers coil.

**An undamped rope on a floor never settles.** *Measurement pitfall.* Run as
dynamics, with no dissipation but contact damping and friction, PyElastica's rope
was still moving at a third of a rod length per second a second after the clamp
had stopped. The pile experiment is quasi-static: the heap at rest is the result,
and the adapters' settling damping, mild next to the feed, brings it there.

**A clamp lowered into the heap flings the rope.** *Measurement pitfall.* With
the clamp stopping 5 cm above the floor it descended into the standing loop under
it and crushed it, and MuJoCo's rope shot out flat at over 1 m/s. The clamp now
stops at 15 cm, clear of the heap.

**Timing zero crossings is fragile.** *Measurement pitfall.* A 0.65 mm vibration
riding on a 1.25 cm swing gave extra crossings and a 52% period error. Fitting a
sinusoid to every frame is robust to it.
