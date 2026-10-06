# Demo-day Q&A: what to say when visitors ask

A cheat sheet for the stand. For each demo you get:

- a one-breath explanation,
- how it actually runs (the algorithm),
- how it was trained, where there is AI,
- why it needs HPC, with numbers we measured ourselves,
- what the same method does in the real world,
- what the demo leaves out (say this before someone else does),
- the awkward questions people are likely to ask.

Measured numbers come from `benchmarks/demo_day_stats.md` (the showcase runs)
and `benchmarks/scaling/` (the "Why HPC" tables). The presenter lines in
[DEMO_STORIES.md](DEMO_STORIES.md) are the script. This file is for the
follow-up questions.

---

## General questions (any demo)

**"Is this running live?"**
What's on screen is a replay of a run the supercomputer computed. Each saved
run records its Slurm job number, node and timings, and the ⚙ drawer shows
them. Supercomputers are batch machines: you submit a job, it waits in a
queue, runs on dedicated nodes, and writes results to disk. They aren't built
for interactive use, so we compute there and replay here. The laptop can do a
small live run at the Local preset (the molecular walker takes a few seconds),
and the dashboard can submit a real job to the cluster ("Run on").

**"What are Leonardo and Discoverer?"**
- **Leonardo** is a EuroHPC supercomputer at CINECA in Bologna. We used its
  Booster partition, where each node has 4 NVIDIA A100 GPUs (64 GB each) and
  32 CPU cores, and its DCGP partition, where each node has 112 CPU cores and
  no GPUs.
- **Discoverer** is the EuroHPC system in Sofia. Our allocation was on its
  NVIDIA Grace-Blackwell partition: each node has 4 GB200/B200 GPUs and 144
  ARM CPU cores.

**"What makes a supercomputer different from my gaming PC?"**
Three things:

1. **Many fast GPUs per node.** They're linked by NVLink, which is much faster
   than PCIe, so they can share one problem.
2. **Thousands of nodes on a very fast network.** One simulation can run on
   hundreds of nodes at once, using MPI to pass messages between them.
3. **Memory and storage at scale.** A 1-million-body gravity run, or a grid
   with billions of cells, doesn't fit on one consumer card.

The physics is the same. HPC lets you use a bigger problem, higher
resolution, or more copies, and get the answer in hours instead of months.

**"Why GPUs and not CPUs?"**
Almost every demo here does the same small calculation for millions of
independent things: every pixel, every grid cell, every pair of stars, every
car. A GPU has thousands of simple cores built for exactly that. Our clearest
measurement is MUrB with 100,000 bodies: **7.2 s per step on one CPU core**,
**48 ms on all 144 cores of a node**, and **6.5 ms on one GB200 GPU**.

**"What does the 'Why HPC' tab show?"**
The same physics timed on 1, 2 and 4 GPUs of one node, and on 1 to N CPU
cores. It measures physics only, with no drawing. We did 3 warm-up runs and
took the median of 5 timed runs. Point out that **big problems scale well and
small ones don't**. With a small problem the GPUs spend their time waiting on
each other rather than computing. That trade-off is the central lesson of
parallel computing.

**"What language is this written in?"**
Mostly Python. NumPy runs on the CPU. On the GPU, CuPy runs hand-written CUDA
kernels, where one kernel launch advances thousands of steps. The fusion
guardian uses PyTorch. MUrB is C++ with CUDA, OpenMP and MPI. Jobs are
submitted with Slurm, and the frames are JPEGs written on the compute node.
No graphics card or display is needed there.

**"Is this real science or a video game?"**
Real algorithms, reduced models. Each demo solves real equations with real
numerical methods, the same kind research codes use, but smaller or
simplified so it runs in minutes and reads well on a screen. Each section
below lists exactly what's simplified. Being upfront about that earns more
trust than overclaiming.

---

# Discoverer day

## 1. Galaxy collision: full 3-D gravity

**One breath:** "The Milky Way and Andromeda are heading towards each other
at about 110 km/s. This is a million particles, each one pulling on every
other one, run forward 9 billion years."

**How it runs**
- It's a **direct N-body** simulation. Every particle's acceleration is the sum
  of the pull from all the others: a = Σ G·m·r / (r² + ε²)^1.5. The ε term is
  **softening**, which stops two particles that pass very close from flinging
  each other off to infinity.
- Particles include disc stars, bulge stars and **dark-matter halo**
  particles. Each one is a *super-particle* standing for millions of stars.
- The time integrator is **leapfrog (kick-drift-kick)**, which conserves
  energy well over billions of years. The maximum step is about 24 million
  years.
- **Initial conditions:** real Local Group values (masses about 1.5×10¹²
  solar masses each, 770 kpc apart, −109 km/s radial approach; van der Marel
  et al. 2012). The Milky Way disc is shaped using **Gaia DR3** star
  distances. Andromeda's shape and colours come from the **PHAT** Hubble
  survey.
- **GPU kernel:** a tiled all-pairs CUDA kernel. Blocks of particles load into
  fast shared memory, and each thread computes several targets at once
  (register blocking). The block size is auto-tuned on whichever GPU runs
  it. There are fp32, fp64 and "mixed" precision modes.

**Why HPC (our numbers)**
- A million bodies means about **10¹² pair interactions per force
  evaluation**.
- On one GB200 that takes **0.42 s (≈47 TFLOP/s)**. Split across 4 GPUs it
  takes **0.13 s (≈149 TFLOP/s, 3.2× faster)**.
- The showcase run was 1,000,000 bodies over 9 Gyr in **35 minutes** on
  Discoverer. A laptop CPU would take weeks.
- Cost grows as N². Double the particles and the work quadruples. That's why
  the question "how many particles can you afford?" is always an HPC
  question.

**Real world**
- Cosmology and galaxy formation. Simulations such as Millennium,
  IllustrisTNG and Euclid's Flagship run billions to trillions of particles on
  the largest supercomputers. Astronomers compare the results with telescope
  surveys to test dark matter and gravity.
- Large research codes usually use **tree or fast-multipole methods**, which
  cost about N log N, instead of direct N². Direct summation is still used
  where accuracy matters most, such as dense star clusters and black-hole
  dynamics, often with GPU codes like NBODY6++GPU.
- The same "everything pulls on everything" pattern appears in molecular
  dynamics (electrostatics) and plasma physics.

**What's simplified:** super-particles instead of real stars. No gas, so no
new stars form. The starting galaxies aren't fitted equilibrium models. Treat
it as *one plausible* encounter: the sideways velocity is uncertain by tens
of km/s, and that uncertainty changes the outcome.

**Curveballs**
- *"Will this actually happen?"* Yes, most likely, in roughly 4–5 billion
  years. Recent Gaia/Hubble studies put the chance of a merger within the
  next 10 billion years at only about 50%. The Sun almost certainly won't hit
  another star, because the space between stars is enormous.
- *"What are the dim particles?"* Dark matter. It has most of the mass, and
  it's what makes the tidal tails.

## 2. Virtual wind tunnel

**One breath:** "Air flowing past obstacles. At first it's smooth. Then the
wake starts shedding vortices left and right: the same effect that makes
flags flap and power lines hum."

**How it runs**
- It uses the **lattice-Boltzmann method (D2Q9)**. Instead of solving the
  Navier-Stokes equations directly, each grid cell holds 9 numbers: how much
  fluid is moving in each of 9 directions.
- Every step has two parts:
  1. **Stream:** each number moves one cell in its direction.
  2. **Collide (BGK):** the numbers relax toward equilibrium.
- Walls use **bounce-back**, meaning the fluid reflects off them. Pressure is
  p = (ρ−1)/3.
- Each cell only talks to its neighbours, so the method is ideal for GPUs.
  One whole step is a **single fused CUDA kernel**.
- The white streaks are passive tracers carried by the flow. They show
  direction and don't affect the physics.

**Why HPC (our numbers)**
- The showcase grid was **7,680 × 4,320 = 33 million cells**, run for 250,000
  steps. Physics took **1 min 42 s** on one GPU.
- At 133 million cells, one step takes **1.7 ms on one GB200** and **0.68 ms
  on 4 GPUs (2.5×)**.
- **Domain decomposition:** cut the grid into pieces, give each GPU one piece,
  and exchange only the edge cells each step. This is how every large CFD
  code scales to thousands of GPUs.
- Reynolds number ≈ 2,000. Real cars and aircraft sit at millions, which needs
  far finer grids in 3-D, and that's where you need a supercomputer.

**Real world**
- Car and aircraft aerodynamics. Commercial lattice-Boltzmann codes such as
  PowerFLOW are used across the automotive industry. Formula 1 teams have
  legally capped CFD budgets.
- Wind loads on buildings and bridges, wind-farm layout, blood flow in
  arteries, ventilation (COVID aerosol studies), and weather and climate
  models.
- Real simulations are 3-D, with billions of cells, turbulence models and
  realistic geometry.

**What's simplified:** it's 2-D, the Reynolds number is low, and the shapes
are simple. The wakes and vortex shedding are qualitatively real, but you
can't read drag numbers for a real car off it.

**Curveballs**
- *"Why vortices?"* Behind the obstacle the flow separates and forms eddies
  that break off alternately. This is called a **von Kármán vortex street**.
  It makes wires sing and tall structures sway, which is why factory
  chimneys have spiral strakes wrapped around them to break it up.
- *"Why not just solve Navier-Stokes?"* You can. Lattice-Boltzmann gives the
  same result at low Mach number, and because every update is local it
  parallelises beautifully.

## 3. Neuro-Racers

**One breath:** "You design the car's brain. Which senses it gets, how many
neurons. Then evolution, not a programmer, teaches it to drive."

**How it runs**
- Each car is controlled by a small **neural network** (a tanh multilayer
  perceptron) with the architecture the visitor built:
  - inputs are distance sensors, speed, and a track compass;
  - outputs are steering, throttle and brake.
- The sensors are **ray-marched through a signed-distance field** of the track
  walls, like lidar.
- The car physics is a simple top-down **bicycle model**: drag, braking, and
  a grip limit above which the car skids.

**How it's trained: a genetic algorithm (neuroevolution), no gradients**
1. Generation 1: thousands of cars with the same brain layout but **random
   weights**. Most spin or crash.
2. **Fitness** = distance driven along the track in a fixed time, minus a
   small crash penalty.
3. **Selection:** the best cars are kept unchanged (elitism), and parents are
   picked by tournament.
4. **Crossover:** a child takes each weight from one parent or the other.
5. **Mutation:** add small random noise. The noise shrinks over generations.
6. Repeat for 60 generations.

There's no training data, no human driving, and no hand-written rules.

**Why HPC (our numbers)**
- The showcase ran **8,192 cars per population × 16 independent searches**
  on Discoverer in about a minute.
- Scaling: **1,048,576 cars** simulating a full generation (1,500 steps) takes
  **0.25 s on one GB200** and **0.13 s on 4 GPUs**.
- Each car is independent, so this is *embarrassingly parallel*. The 16
  searches show that the result depends on luck, so you run many.
- Real AI research works the same way. The expensive part is running many
  experiments: architectures, random seeds, hyperparameters.

**Real world**
- **Evolution strategies at scale.** OpenAI (2017) showed they compete with
  reinforcement learning when run on thousands of CPU cores. Uber AI's "Deep
  Neuroevolution" did similar work.
- **Neural architecture search:** the visitor chooses the architecture; big
  labs automate that choice on clusters.
- **Robotics:** NVIDIA Isaac Gym trains thousands of simulated robots in
  parallel on GPUs, then transfers the policy to real hardware.
  Self-driving companies test in simulation for billions of virtual
  kilometres.

**What's simplified:** a toy car model (no tyres or suspension). Lap times are
in simulated seconds. The CPU and GPU versions give scientifically the same
result but not bit-identical results, because rounding differences can change
which car wins.

**Curveballs**
- *"Why not backpropagation?"* Fitness is only known at the end of a drive
  ("how far did you get"), and it can't be differentiated with respect to the
  weights. Evolution just needs a score. It's also trivially parallel.
- *"Can a brain with no hidden layer win?"* Often, on simple tracks. That's
  the point of the challenge. Bigger isn't automatically better.
- *"What's the ghost?"* A previous visitor's champion, re-simulated exactly on
  this track.

## 4. Black-hole lensing

**One breath:** "This black hole is real. Gaia found it in 2024. We parked a
virtual camera next to it and traced every light ray exactly through
Einstein's equations to the real star it came from."

**How it runs**
- The black hole is **Gaia BH3**: 32.7 solar masses, 590 parsecs away, and
  the most massive stellar black hole known in our galaxy. You can also
  choose Gaia BH1 or BH2.
- **Sky:** 1.8 million real **Gaia DR3 stars**, moved to where they'd appear
  from the black hole's position. Colour comes from each star's measured
  temperature.
- **Physics:** exact **null geodesics in the Schwarzschild metric**, meaning
  the true paths of light around a non-rotating black hole.
  - A ray's fate depends only on its impact parameter b.
  - Rays with b < 3√3 GM/c² fall in.
  - For the rest, the total bending angle comes from the orbit equation by
    **Gauss-Legendre quadrature**, computed once per camera position into a
    lookup table.
  - Every pixel is then just "look up the bend, sample the sky there".
  - Brightness includes the gravitational blueshift (a g⁴ factor).
- It's validated against the known weak-field formula, the exact shadow size,
  and an independent RK4 integration, agreeing to within 2×10⁻⁵ radians.
- There are **3 × 3 rays per pixel** (supersampling) for smooth edges.

**Why HPC (our numbers)**
- Every pixel is an independent question: *"where did this light come
  from?"*
- An image at **15,360 × 8,640 (16K) with 16 rays per pixel is 2.1 billion
  rays**. That takes **42 ms on one GB200** and **19 ms on 4**.
- The showcase: 600 frames at 2560×1440 took 38 s of physics. Most of the
  time went into drawing, not physics.

**Real world**
- **Event Horizon Telescope.** The images of M87* (2019) and Sagittarius A*
  (2022) were interpreted by comparing them with libraries of millions of
  simulated images. Those come from GRMHD simulations (magnetised gas
  falling into a spinning black hole), ray-traced exactly like this, on
  supercomputers.
- Gravitational-lensing surveys use galaxy clusters as natural telescopes and
  map dark matter (Euclid, JWST).
- The film *Interstellar*: its black hole was rendered with a physics-based
  ray tracer, and the work became a research paper.

**What's simplified:** no spin (a Kerr black hole would be lopsided), and no
glowing disc. That's actually realistic: these black holes are dormant, with
nothing falling in. No companion star. Each frame is a stationary camera, so
there's no aberration from the camera's own motion.

**Curveballs**
- *"Is the black disc the event horizon?"* No. It's the **shadow**, about 2.6×
  bigger. Any ray aimed inside it falls in.
- *"What are the rings at the edge?"* Light that went once, twice, three
  times around the hole before reaching you: the whole sky copied into ever
  thinner rings.
- *"How do they know it's there if it's black?"* Gaia saw a star wobbling
  around an invisible partner of 33 solar masses. Nothing that heavy and dark
  can be anything but a black hole.

## 5. Molecular Machine

**One breath:** "Write a protein: oily beads, water-loving beads, plus and
minus. Then watch heat and water fold it. We also have the molecular machines
the 2016 Nobel Prize was for, and the rotary motor that makes the fuel in
every one of your cells."

**How it runs (all modes)**
- It's a **coarse-grained model**: each bead stands for a group of atoms, not
  one atom.
- **Langevin dynamics** (BAOAB integrator): water isn't simulated molecule by
  molecule. It appears as friction plus random thermal kicks. This is
  **implicit solvent**, and it sets the temperature (310 K, body
  temperature).

**Fold mode (the protein)**
- It's based on the **HP model** (Dill, 1985):
  - H beads are hydrophobic (oily) and attract each other;
  - P beads are polar (water-loving);
  - charged beads interact through **screened (Debye-Hückel) electrostatics**,
    and salt shortens their range.
- Bonds are springs with a bending stiffness. Every pair has a repulsive core
  so beads can't overlap.
- **Every pair of beads is evaluated every step.** A single fused CUDA kernel
  runs up to 2,000 steps per launch.
- The main result is robust and real. Oily beads bury themselves in a core,
  and that's why real proteins have a hydrophobic core. A chain with no oil
  stays a floppy coil. Opposite charges zip together. Heat unfolds it.

**Shuttle mode:** a **rotaxane**, a ring threaded on an axle with a stopper at
each end and two binding stations. The switch only changes which station is
sticky. **Nothing pushes the ring.** It diffuses on heat alone and gets caught.

**Walker mode:** a **flashing Brownian ratchet**, the textbook mechanism
behind kinesin, the protein that walks cargo along your cells' microtubules.
The fuel doesn't push the feet. It switches the track off briefly. The track
is lopsided, so random jiggling is caught more often one step ahead than one
step behind. Add a load and the walker stalls, or goes backwards.

**Rotor mode:** the F1 part of **ATP synthase**. Each fuel event moves the
energy minimum a third of a turn, and the rotor follows. Add cargo and it
slows, stalls, and eventually runs backwards. The real enzyme does exactly
that in reverse to *make* ATP. The gold marker bead copies Noji and
colleagues' 1997 experiment, where they filmed the real motor spinning.

**Why HPC (our numbers)**
- The showcase fold: a **1,000-bead chain, 6 million steps, in 22 minutes on
  one GB200** at 98% GPU utilisation. That's about half a million pair forces
  per step, **≈3×10¹² pair evaluations in total**.
- Now scale up to reality. A real protein in water has about 100,000–1,000,000
  atoms. Time steps are **2 femtoseconds**. Folding takes microseconds to
  milliseconds, which means **10⁹–10¹² steps**. That's why this is one of
  the biggest users of supercomputer time worldwide.
- The walker and rotor are one or two particles, so they run on a laptop in
  seconds. We say so honestly: not everything needs a supercomputer.

**Real world: what molecular dynamics is actually used for**
- **Drug discovery.** Simulate how a candidate drug binds to its target
  protein, how long it stays bound, and why a mutation causes resistance.
  Codes such as GROMACS, NAMD, AMBER and LAMMPS run on Leonardo-class
  machines every day.
- **COVID-19:** all-atom simulations of the spike protein (the 2020 Gordon
  Bell special prize) showed how it opens to infect cells. Folding@home
  briefly became the world's first exascale computer from donated PCs.
- **AlphaFold** (2024 Nobel Prize in Chemistry) predicts structure with AI
  trained on huge GPU/TPU clusters. MD then shows how the structure *moves*.
- **Anton** is a supercomputer purpose-built just for molecular dynamics.
- **Molecular machines** (2016 Nobel: Sauvage, Stoddart, Feringa) are being
  developed for targeted drug delivery, smart materials, and molecular
  electronics.
- **Motor proteins:** kinesin and myosin (muscle), and ATP synthase (1997
  Nobel for its mechanism).

**What's simplified:** it isn't a real force field. There are no atom types
or explicit water, time is in reduced units rather than femtoseconds, and it
doesn't predict any real protein's structure. The motors reproduce the
*shape* of real torque-speed curves, not a specific protein's numbers.

**Curveballs**
- *"Can this cure a disease?"* Not this model. The same physics at atomic
  detail, on a supercomputer, is part of how modern drugs are designed.
- *"If nothing pushes the ring, why does it move?"* At this scale, heat alone
  kicks everything around billions of times a second. The machine doesn't
  create motion. It *rectifies* random motion, choosing which random moves
  to keep.
- *"Doesn't that break thermodynamics?"* No. The switch or fuel costs energy.
  Without fuel, the walker goes nowhere.

---

# Leonardo day

## 6. MUrB N-body (NBody-EuroHPC)

**One breath:** "This isn't our code. It's MUrB, a real C++/CUDA
research-style N-body code. We run it on the supercomputer and draw what it
computed. It reached 92% of an A100's theoretical peak."

**How it runs**
- It's a **direct O(N²) gravity sum** with softening, in fp32, using MUrB's
  own integrator.
- MUrB has several implementations of the same physics:
  - `cpu+naive` and `cpu+simd` (vector instructions);
  - `cpu+omp` (all cores via OpenMP);
  - `gpu+tile+full` (tiled CUDA kernel with all data resident on the GPU);
  - an MPI backend with one GPU per process.
- Our dashboard only launches it and renders the recorded trajectory, using
  the colour scheme of MUrB's own viewer (speed → blue/cyan/white).

**Why HPC (our numbers). This is the best demo for the CPU vs GPU story.**
- Leonardo, 1,000,000 bodies on **one A100: 17.9 TFLOP/s, about 92% of the
  card's single-precision peak**. That's very close to the hardware limit.
- On Discoverer with 100,000 bodies, time per step:

  | | Time per step |
  |---|---|
  | 1 CPU core | 7.2 s |
  | 144 cores | 48 ms |
  | 1 GB200 GPU | 6.5 ms |

  One GPU is **about 1,100× faster than one core**.
- 500,000 bodies: **1.7 s** on a full 144-core node, **122 ms** on 1 GPU, and
  **35 ms on 4 GPUs** via MPI (3.4×).
- Show that 4 GPUs are *slower* than 1 at 100k bodies (7.2 ms vs 6.5 ms). The
  problem is too small, so communication costs more than it saves. Parallel
  speed-up has to be earned with problem size.

**Real world:** the same as the galaxy collision (astrophysics, star clusters,
planet formation). It's also a classic **HPC benchmark and teaching code**:
optimising the N-body kernel teaches tiling, memory hierarchy, SIMD and MPI,
the skills behind every big simulation code.

**What's simplified:** the initial setups (a rotating shell around a central
mass, or a random cloud) are code demonstrations, not a real galaxy. The units
are metres and kilograms.

**Curveballs**
- *"What's a FLOP?"* A floating-point operation: one multiply or add on a
  decimal number. 17.9 TFLOP/s is 17.9 trillion per second, from one card.
  Leonardo has thousands of these cards.
- *"Why not use every GPU on Leonardo?"* O(N²) with a million bodies already
  saturates a few GPUs. To use thousands you'd go to billions of bodies and
  N log N tree methods.

## 7. Dragon Raytracer (video)

> ⚠️ This demo is a video from outside our codebase. **Check the algorithm
> details with whoever made it before demo day.** Below is what's standard
> for this kind of render.

**One breath:** "A dragon in gold, blue and glass, lit inside a Cornell box.
Every pixel is found by following light rays as they bounce around the
scene, on the GPU."

**How it (typically) runs**
- **Ray tracing / path tracing.** For each pixel, fire rays into the scene. At
  each surface, bounce according to the material:
  - metal (gold) reflects sharply;
  - glass reflects and refracts (Snell's law and Fresnel equations);
  - diffuse walls scatter randomly.
- Averaging many random paths per pixel (**Monte Carlo integration**) gives
  soft shadows, colour bleeding from the red and green walls onto the dragon,
  and caustics through the glass.
- The dragon mesh has hundreds of thousands to millions of triangles. A
  **BVH (bounding volume hierarchy)** is a tree of boxes that lets each ray
  test only a few triangles instead of all of them.
- The **Stanford Dragon** is a classic 3-D scan from Stanford's scanning
  repository. The **Cornell box** (Cornell University, 1984) is the standard
  test scene, because a real physical box was built and photographed to
  check renders against.

**Why HPC:** every pixel and every sample is independent, so it's perfectly
parallel. A clean 4K frame might need thousands of samples per pixel, which
is tens of billions of ray paths per frame. Across a film that becomes a
**render farm**.

**Real world**
- Film VFX and animation: Pixar and others run render farms of thousands of
  machines, with hours per frame.
- Architectural lighting, car design visualisation, and optical and lens
  design.
- **Science:** Monte Carlo transport is the same maths used for neutron
  transport in reactor design, radiation therapy planning, and radiative
  transfer in stars and climate models.
- Visualising the output of *other* HPC simulations.

## 8. Star in a Bottle (fusion plasma)

**One breath:** "Fusion powers the Sun. To do it on Earth you need plasma
hotter than the Sun's core, over 100 million degrees, and no wall can touch
it. Magnetic fields make the bottle. Mode 2 hands the magnets to an AI."

### Mode 1: passive confinement

**How it runs:** it integrates the **complex Ginzburg-Landau equation** on a
periodic 2-D grid, then wraps that grid onto a torus (the tokamak shape).
CGL is a universal equation for waves near an instability. It produces
coherent waves, defects and **turbulence**, the key enemy of confinement. The
magnetic field, heating and density controls change its coefficients. The
glowing particles are passive tracers drifting with the field.

**Numbers:** a **2,048² grid, 40,000 steps, 6,000 tracers**. Physics took
2.5 minutes on one A100.

### Mode 2: AI Plasma Guardian

**How it runs**
- The same turbulent field keeps running. On top of it is a **reduced control
  model** of the plasma's position with 6 state variables:
  - radial and vertical position and velocity;
  - pressure;
  - a **tearing-mode risk** proxy.
- Left alone, the model is unstable and the plasma drifts into the wall.
- Tens of thousands of **markers** (tracer particles for the plasma) fill the
  torus. A marker that reaches the wall is a spark, counted as a loss.

**How it's trained: differentiable simulation, not trial and error**
- The policy is a small **PyTorch neural network**:
  - input: 6 noisy diagnostics;
  - output: commands to 3 coil banks (radial, vertical, shaping).
- Because the control model is differentiable, training
  **backpropagates through whole batches of virtual plasma shots**. The loss
  is displacement from centre, tearing risk, velocity, coil effort, and an
  estimate of wall contact.
- The run is a series of **shots**:
  1. During a shot the policy is **frozen**, so what you see is a clean test.
  2. Between shots it trains. Half the training batch is replayed from states
     the last shot visited, half is random.
  3. Every shot is the identical experiment (same seed, same disturbances), so
     any improvement is down to the policy.
- The training budget grows geometrically between shots (2, 5, 11, 26 … 1,500
  updates), so the learning curve is visible.
- **Nothing is pre-trained.** Every run starts from random weights.
- Real Leonardo result: markers lost per shot went **82,150 → 78,780 → 68,097
  → 12,044 → 44,681 → 5,853 → 1,369 → … → 1,256**, against **83,457** with
  the coils off. The step backwards at shot 5 is real, and we kept it.
- Losses never reach zero. The model has built-in edge turbulence, so there's
  a floor that no controller can remove.
- The "re-test" view runs the trained controller against instability drives
  it has **never seen**, each in its own plasma simulation. That checks it
  generalises rather than memorises.

**Why HPC:** 16 plasma simulations running together, a 512² grid, 10 shots
and 1,500 network updates took about **10 minutes on one A100**. Real
controllers are trained on millions of simulated shots, because you can't
experiment freely on a €20-billion machine.

**Real world**
- **ITER** (France) is the international tokamak under construction.
  **JET** (UK) set the fusion energy record in 2023. In the US, NIF
  (laser-driven fusion) achieved ignition in 2022.
- **AI plasma control is real.**
  - DeepMind and EPFL (Nature, 2022) trained a reinforcement-learning
    controller in simulation, then used it to control the real TCV tokamak's
    magnetic coils directly, including holding plasmas in unusual shapes.
  - Princeton and DIII-D (Nature, 2024) used AI to **predict tearing
    instabilities about 300 ms ahead** and steer around them. Our "tearing
    risk" input is a nod to exactly that.
- The physics simulations behind fusion (gyrokinetic turbulence codes such as
  GENE, GYSELA and XGC) are among the heaviest users of Europe's
  supercomputers. EUROfusion has long run its dedicated computing at CINECA,
  the centre that hosts Leonardo.

**What's simplified:** the wave equation is an exhibition-scale model, not a
tokamak code. The magnetic field lines are illustrative, not a solved
equilibrium. The markers have no gyration and no collisions. **Never quote
the loss numbers as a real confinement time.** The AI training, the
uncontrolled comparison and the re-test are genuinely computed.

**Curveballs**
- *"Why can't a human do it?"* Instabilities grow in milliseconds. Real
  tokamaks already use fast automatic feedback, and AI adds prediction and
  handles more complex shapes.
- *"Is this reinforcement learning?"* Not exactly. RL learns from trial and
  reward. Here we can differentiate straight through the simulator, which
  learns much faster. DeepMind's TCV work used RL.
- *"When will we have fusion power?"* ITER plans its first plasma in the
  2030s. Commercial plants are hoped for from the 2040s onwards, though
  timelines have slipped before.

## 9. Cosmic web

**One breath:** "Straight after the Big Bang, matter was spread almost
perfectly evenly, to 1 part in 100,000. Gravity amplified those tiny
ripples into this: filaments, clusters and empty voids. Nobody drew it."

**How it runs**
- **Particle-mesh (PM) gravity**, in 2-D:
  1. Put the particles' mass onto a grid.
  2. Solve Poisson's equation for gravity with an **FFT** (fast Fourier
     transform).
  3. Read the force back at each particle and move it.
- **Initial conditions:** the **Zel'dovich approximation** applied to a random
  field with a power-law power spectrum. This is how real cosmology
  simulations start.
- It integrates in **comoving coordinates** on an expanding universe. There
  are switches for dark energy, warm dark matter (which smooths small-scale
  ripples) and helium fraction.

**Why HPC (our numbers)**
- **2.4 million particles on a 1,024² mesh, 4,000 steps, in 12 seconds of
  physics** on one A100.
- The FFT solve is global: every cell affects every other cell. Splitting it
  across many GPUs needs heavy all-to-all communication. That's the classic
  HPC challenge for cosmology codes, and it's why fast networks matter.
- The "Why HPC" note: we split the particles across GPUs, while the FFT stays
  on one GPU. The meshes cross NVLink every step, so you only gain with many
  particles.

**Real world**
- Large simulations (Millennium, IllustrisTNG, Euclid Flagship with about 4
  trillion particles, Uchuu, Abacus) are built to interpret sky surveys:
  **Euclid** (ESA, launched 2023), **DESI**, and Vera Rubin. They're the only
  way to predict what the universe should look like under a given dark
  matter or dark energy model.
- Real codes use TreePM or adaptive meshes, in 3-D, with gas physics, star
  formation and black holes. Runs take millions of GPU/CPU hours.

**What's simplified:** it's 2-D, the power spectrum is simple, the mesh
assignment is nearest-grid-point, and there's no gas physics. Helium only
changes a smoothing scale. It shows real structure formation, but it isn't a
precision cosmology code.

**Curveballs**
- *"What's dark matter?"* Matter we can't see but can weigh, about 5× more
  than normal matter. Without it, these structures wouldn't have had time to
  form.
- *"Where are we in this?"* The Milky Way sits in a filament, near the edge of
  a void, in the Laniakea supercluster.

## 10. Bat vs Moth

**One breath:** "Two visitors, two brains. The bat hunts blind using echoes.
The moth can hear the bat coming and has a secret weapon. Both evolve at the
same time: an evolutionary arms race inside a supercomputer."

**How it runs**
- It uses the same **genetic algorithm** as Neuro-Racers, but with **two
  populations co-evolving**, each with the architecture a visitor built.
- **Bat senses:**
  - left and right ears, pointing ±40°;
  - echo loudness falls with distance;
  - echo delay, and a simple Doppler reading (closing speed);
  - whiskers that detect rock.
- **Moth senses:** it hears the bat's calls from further away (9 units vs the
  bat's 6), which is also true in nature.
- **Moth weapons:** a power dive, and **jamming**. A jamming moth replaces its
  own echo with a fake one at a random loudness and distance. Jamming and
  diving cost fitness, so they're only worth using selectively.
- **Fitness:**
  - bats score for catches, and for flying towards moths they can hear;
  - moths score for survival and for keeping their distance.
- Moths get a head start with evolution off, so the bats become worth
  countering first.
- **Nothing is scripted.** "Turn toward the louder ear" and "click only when
  a bat is close" are discovered by evolution. The dashboard only says
  "jamming evolved" when moths jam much more with a bat near than far.

**Why HPC (our numbers):** **32,768 caves × 16 independent worlds, 100
generations, in about 4 minutes of physics** on one A100. Each cave runs one
bat and several moths. The GPU step is four fused CUDA kernels, with one
thread per cave. The re-test drops the evolved animals into caves they've
never seen, to check the behaviour generalises.

**Real world**
- **The biology is real.** Tiger moths (*Bertholdia trigona*) jam bat sonar
  with ultrasonic clicks (Corcoran, Barber & Conner, *Science* 2009). Some
  moths evolved sound-absorbing scales, and some bats evolved quieter calls
  in response.
- **Multi-agent AI:** OpenAI's hide-and-seek (2019) produced emergent tool
  use from exactly this kind of competitive co-evolution. DeepMind's
  AlphaStar trained a "league" of agents against each other on large
  clusters. The same idea drives self-play (AlphaGo) and adversarial
  training.
- **Agent-based models** on HPC are used in epidemiology, traffic, ecology and
  economics.
- **Engineering:** sonar and radar design, and electronic countermeasures.
  The bat–moth contest is literally radar jamming.

**What's simplified:** the sonar is a reduced model, not acoustics. Rock
doesn't reflect sound, and there's no frequency content. Distances are
abstract cave units. The moths' head start and "quiet start" (jamming off at
first) are presentation choices, and the viewer states them.

**Curveballs**
- *"What if the bat has only one ear?"* It struggles to tell left from right.
  Try it: catch rate drops a lot. A hand-coded two-eared bat catches about
  68% of moths. A deaf one catches about 6–18%.
- *"Why does jamming sometimes not evolve?"* It has a cost, and it only pays
  off once the bats are good. Evolution isn't guaranteed to find a trick.
  That's why we run 16 worlds.

## 11. Pac-Man Agentic (video)

> ⚠️ Gianluca's project, not in our codebase. **Ask him before demo day**:
> Is it reinforcement learning (e.g. DQN/PPO), or LLM-based agents? How long
> did it train, on how many GPUs? What does the agent see (pixels or game
> state)? Fill in the blanks below.

**Safe general answers**
- **Reinforcement learning:** the agent sees the game, chooses a move, and
  gets rewards (pellets, ghosts eaten, not dying). Over millions of games it
  learns which moves lead to more reward. Nobody tells it the rules.
- History: DeepMind's DQN (Nature, 2015) learned dozens of Atari games from
  pixels. **Ms. Pac-Man was one of the hardest**, because it needs long-term
  planning. Agent57 (2020) was the first to beat the human baseline on all
  57 Atari games.
- **Why HPC:** RL is hungry for experience. Training runs many game copies in
  parallel (hundreds to thousands of environments) and trains the network on
  GPUs. One laptop would take days to weeks.
- **Real world:** the same approach powers robotics, chip layout, data-centre
  cooling control, and the fusion controllers in Star in a Bottle.

---

## If you get stuck

- **"I don't know, but here's how we'd find out"** is a good answer at a
  science stand.
- Every run's **⚙ / metadata panel** has the job number, node, device, grid
  size, step count and timings. Read it out.
- The honest limitations are in [SCIENTIFIC_NOTES.md](SCIENTIFIC_NOTES.md).
  Per-demo details are in `docs/demos/<demo>/README.md`.
