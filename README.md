# cosseratbench

***A visual benchmark suite for evaluating rope, cable, and Cosserat rod simulation across deformation, contact, and dynamic stress cases.***

| #  | Experiment                         | What it tests                                                           |
| -- | ---------------------------------- | ----------------------------------------------------------------------- |
| 1  | **Twist → plectoneme / knot**      | twist, bending, buckling, extreme curvature, self-contact               |
| 2  | **Rope drop / pile**               | gravity, friction, chaotic self-contact, many simultaneous contacts     |
| 3  | **Catenary / hanging cable**       | basic gravity, tension, sag, stretch; good sanity/validation case       |
| 4  | **Cantilever bend + twist**        | isolated bending stiffness, torsion, large deformation                  |
| 5  | **Pendulum / swinging cable**      | dynamics, inertia, damping, oscillation                                 |
| 6  | **Snap / whip test**               | very fast motion, high curvature, timestep stability                    |
| 7  | **Cylinder wrap / capstan**        | rod-cylinder contact, friction, sliding, tension transfer               |
| 8  | **Pulley / sheave**                | moving contact, bending around small radius, tension under motion       |
| 9  | **Obstacle course**                | repeated contact against cylinders/planes/spheres, sliding and snagging |
| 10 | **Two-rope interaction**           | rod-rod contact, crossing, rubbing, entanglement                        |
| 11 | **Loop / knot tightening**         | persistent dense self-contact and friction under increasing tension     |
| 12 | **Compression / coiling**          | rope pushed into a confined area; buckling and pile formation           |
| 13 | **Container packing**              | rope fed into a box/cylinder; dense 3D self-contact                     |
| 14 | **Parameter/extreme stress sweep** | deliberately push stiffness, friction, speed, resolution, timestep      |

