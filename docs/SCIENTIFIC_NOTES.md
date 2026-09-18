# Scientific notes and limitations

The demos aim to make legitimate computational ideas visible, but they intentionally trade production-level physical detail for fast iteration and a coherent exhibition experience.

## Black-hole lensing

The demo has two separate scientific views. **2-D observer image** is an exhibition-grade image-space gravitational lens mapping: every output pixel is mapped back to the source sky in parallel, producing lens-like warping and a stylised emission ring. **3-D ray space** numerically advances photon directions through a three-dimensional point-mass deflection field, renormalising the direction after each step and adding a small signed spin-like transverse term. The displayed paths therefore come from the integrator rather than decorative Bezier curves, and rays can either escape to the source plane or cross the capture radius.

This is still a weak-field educational model, not a full Schwarzschild/Kerr null-geodesic integrator and not GRMHD. The spin term is qualitative rather than a validated frame-dragging solution. For a research-accurate flagship, replace `integrate_rays()` and `lens()` with a validated relativistic ray tracer while keeping the two-mode frame/output interface.

## PBH

The supplied capstone describes a Misner-Sharp / Chebyshev pseudo-spectral / RK4 solver and a critical threshold near `δc ≈ 0.49774` for its chosen setup. The repository does **not** silently claim to reproduce that full numerical-relativity solver. `PBHDemo` is a reduced qualitative exhibition model whose only purpose is to make the collapse-versus-dispersion narrative and parameter sweep testable.

Use `tools/render_pbh_research.py` to convert output from the validated research code into the exhibition frame format. This is the recommended route for the final EuroHPC display.

## Fluid

The fluid module is a real D2Q9 lattice-Boltzmann solver in two dimensions. It
is suitable for qualitative wakes and vortex shedding, not quantitative
aerodynamics around a real aircraft or car. The circular obstacle is
intentionally simple.

The displayed background is flow **speed**, with vorticity as a secondary tint;
an earlier version showed |vorticity| on a fire palette, which read as a heat
map rather than as moving air. Overlaid streaklines are passive tracers
advected by the same velocity field, drawn with a short position history so a
still frame carries flow direction. Their time step is amplified for legibility
— they are a visualisation of the velocity field, not a second physical
integration. Reported pressure is p = (rho - 1)/3 from the lattice density.

## Cosmic web

The cosmic-web code is a real 2-D particle-mesh gravitational model: Zel'dovich
initial conditions drawn from a power-law power spectrum, an FFT Poisson solve
on the density contrast, nearest-grid-point deposition/sampling, and comoving
integration on a matter-dominated expanding background.

It is still **not a cosmological precision code**: two dimensions only, a
power-law P(k) rather than a real transfer function, no cosmological parameter
integration, nearest-grid-point rather than CIC assignment, and no force
resolution study.

### Gas composition

The `helium` parameter is the helium mass fraction. Masses are the physical
ones: hydrogen A = 1.008, helium-4 A = 4.003. Note that helium's atomic
*number* is 2 but its *mass* number is 4, and it is the mass that sets the gas
dynamics. The mean molecular weight of a neutral H/He mix is

    mu = 1 / (X/1.008 + Y/4.003),   X + Y = 1

giving mu = 1.008 for pure hydrogen, 1.229 for the primordial mix (Y = 0.24)
and 4.003 for pure helium. Since the sound speed goes as sqrt(T/mu), a heavier
mean molecular weight means a shorter Jeans length and therefore finer
structure. This is applied as a Jeans-scale smoothing of the potential. It is a
legitimate scaling, not a full two-fluid treatment: there is no thermal
evolution, no cooling, and no ionisation history.

## Galaxy collision

This is a restricted N-body encounter: tracer stars feel two softened (Plummer)
galaxy potentials while the galaxy centres mutually accelerate. Tracers do not
contribute self-gravity, and there is no dynamical friction, so the orbit does
not decay the way a real merger's does.

Positions and velocities are advanced with the same kick-drift-kick leapfrog
scheme on NumPy and CuPy. CPU workers operate only on disjoint tracer slices;
that execution detail does not change the force model or initial conditions.

The default preset uses published Local Group values: M_MW and M_M31 of about
1.5x10^12 solar masses each, a separation of 770 kpc, a radial approach
velocity of -109 km/s and a transverse velocity of about 17 km/s (van der Marel
et al. 2012, ApJ 753, 8; refined by Gaia astrometry). Working units are kpc,
km/s and solar masses, so the on-screen clock in Gyr is meaningful.

Treat the outcome as *a* plausible encounter, not a prediction. The masses are
virial estimates with large error bars and the transverse velocity in
particular is uncertain at the tens-of-km/s level — which is exactly what the
reveal sweep illustrates. Without dynamical friction this model shows the first
passage and tidal bridge faithfully but will not settle into a merger remnant.

The opening discs use observationally motivated morphology (a four-arm Milky
Way disc; a two-arm, ringed M31 disc), and the renderer accumulates each
tracer's represented mass into pixels before assigning brightness and colour.
This makes overlap visibly brighter, but the particle positions are still
generated tracers, **not** an imported Gaia/PHAT star catalogue. The PHAT/PHAST
catalogues resolve tens of millions of M31 stars and are too large to download
inside an exhibition run. The optional `data/m31_catalog_reduced.npz` asset is
the defined catalogue-driven route: `tools/reduce_star_catalog.py` combines a
curator-supplied, deprojected CSV into spatial cells, storing each cell's
weighted centre of mass and total flux/mass. If that asset is present, M31
positions are weighted resamples of it; otherwise the morphology model is
used. This changes the initial tracer distribution, not the restricted-N-body
physics or its limitations.

## Reaction diffusion

This is a genuine Gray-Scott reaction-diffusion finite-difference simulation
with periodic boundaries, integrated on a 16:9 domain so the displayed spots
are not stretched. The step budget is set per profile rather than derived from
the frame count: the pattern needs O(10^4) steps to fill the domain, and a
frame-derived budget stopped near 10^3, leaving the screen almost empty.

## Crystal growth

The crystal module is **not** a phase-field PDE solver. It is a recursive
geometric growth model: a set of line segments generated by applying the same
branching rule at successively smaller scales, with six named habits (classic,
fern, seaweed, star, coral, plate) and a selectable symmetry.

This is an honest description of what it does — it is procedural geometry that
imitates dendritic morphology, not a solved free-boundary problem. Real habit
selection depends on temperature and supersaturation (the Nakaya diagram); here
the habit is simply chosen. Do not present it as a calibrated ice-growth
calculation.

The advantage of the geometric form is that it is resolution independent. The
viewer regenerates the geometry for whatever window is on screen, so magnifying
reveals genuinely finer branch generations rather than enlarged pixels, without
a fixed depth limit.

Be clear about what that means physically: the branching rule is applied at
every scale because it is *defined* at every scale, not because a real crystal
is self-similar without end. Real dendrites stop at the scale set by molecular
attachment kinetics and the diffusion field. The unbounded zoom is a property
of the model, not a claim about ice.

## Neural-network wall

Each model is a small coordinate MLP: it is given a pixel coordinate `(x,y)`
and learns to output three values `(R,G,B)`. That is why it can recreate a
drawn, uploaded or visitor-captured image without recognising what the image
contains. It is fitting a coloured function over a small canvas, not classifying
a face or a digit. Camera consent and capture occur in the browser; only the
captured 128-pixel RGB target is included in the run request.

When PyTorch is installed the module performs genuine training of many small
coordinate MLPs. The networks are held as stacked (N, in, out) weight tensors
and advanced with one batched forward/backward and a hand-written Adam step
carrying a per-network learning rate, so the whole ensemble is a single set of
matmuls — the structure that actually maps onto a GPU. Hidden width is varied
by masking unused units. Networks are laid out as a 2-D hyperparameter grid:
learning rate along a row, width down a column.

Without PyTorch it falls back to a deterministic Fourier reconstruction
surrogate. The surrogate is **not training**: the frame headline, subtitle and
badge all say so explicitly, and the metadata reports `numpy-surrogate`. Do not
call the fallback "neural-network training" during a public presentation.

## Star in a Bottle / fusion plasma

The demo has two selectable modes, chosen through the run's `method`. Both
integrate the same field solver; mode 2 adds a control loop on top of it.

### Mode 1 - passive confinement

This mode integrates the complex Ginzburg--Landau amplitude equation on a
periodic 2-D lattice and maps that field onto a torus. The equation is a real
nonlinear wave model that produces coherent waves, defects and spatiotemporal
turbulence; it is useful for exposing the compute pattern of a plasma field
solver. Magnetic field, heating and density alter its dimensionless drive,
dispersion and damping coefficients.

It is **not** a predictive tokamak code, gyrokinetic simulation, MHD equilibrium
solver or model of any particular reactor. The displayed tesla and megawatt
controls provide an intuitive operating-space narrative, but the coefficient
mapping is illustrative and must not be used to infer fusion performance. The
reveal is nevertheless genuine computation: every tile integrates a separate
field with different control values.

The luminous moving particles are **passive tracers**, not kinetic plasma
particles and not additional degrees of freedom in the field solve. Their
toroidal and poloidal drift is sampled from local phase gradients and rotated
amplitude gradients of the evolved field. They make transport and changing
flow direction visible without feeding back into the simulation. The short
trails are trajectory history; camera rotation is deliberately slower so the
field-driven motion remains distinguishable from the changing viewpoint.

### Mode 2 - AI plasma guardian

Mode 2 keeps the same field solver running and uses it as the turbulence
source, then adds two further reduced models.

The first is a **research-inspired reduced control environment**, not a tokamak
equilibrium, transport, or tearing-mode solver. Its six state variables are
radial and vertical position/velocity plus dimensionless pressure and
tearing-risk proxies. Three aggregate actuator outputs represent radial,
vertical and shaping coil banks, drawn as eight coil rings around the machine
and as their cross-sections in the poloidal overlay; the count and placement are
an exhibition simplification of a real poloidal field coil set. Open-loop positive feedback makes the
reference trajectory approach the vessel boundary; a small PyTorch MLP is
optimized by back-propagating through batches of those virtual trajectories to
minimize displacement, risk, velocity, and coil effort. The demo genuinely
trains this policy and uses its learned weights to draw the policy graph and
its actions to drive the coloured coils. Without PyTorch the run falls back to
an explicit analytical controller, which is labelled as such everywhere and is
**not** learning.

Simulation and training are deliberately separate phases. The run is a sequence
of virtual shots; within a shot the policy is frozen, so the displayed physics
is a clean closed-loop episode that training cannot perturb. Between shots the
shot is scored on its wall losses and the policy is optimized, with half of each
training batch replayed from the states that shot visited and half sampled at
random. Every shot repeats the identical experiment — same field seed, marker
seed, start state and disturbance sequence — so differences on the scoreboard
are attributable to the policy alone. The optimization target includes a
closed-form, differentiable estimate of the marker population's equilibrium
radius derived from the same balance the visible markers obey, so the policy is
genuinely minimizing wall contact rather than a stand-in for it. Nothing is
pre-trained and no weights persist between runs; the learning rate is a
profile setting chosen so the improvement is visible across several shots
rather than complete after the first, and the plateau that follows is real.

The second is a **transport-flavoured marker population** inside the torus.
Each marker carries a toroidal angle, a poloidal angle and a minor radius
measured from the magnetic axis, and the controller's displacement *is* that
axis, so the coil commands really do decide which markers stay inside. Radial
motion balances an edge-weighted turbulent drift sampled from the live field,
a resonant island kick scaled by the tearing proxy, a rare large-angle
scattering channel standing in for collisional losses, and a restoring term
set by the field strength and the shaping command. A marker that reaches the
wall is counted as a loss, leaves a spark at the contact point and is recycled
into the core so the population stays constant. Every spark on screen is a
counted contact.

These markers are **not** a gyrokinetic or full-orbit particle code: there is
no gyromotion, no collision operator, no divertor geometry and no
self-consistent field response. The loss counts are internally consistent
diagnostics of this reduced model and must never be quoted as a confinement
time, a particle flux or a prediction for any device.

The uncontrolled comparison is real: a second marker population starts from the
same seed and the same state and runs with the coils switched off. The reveal
is likewise real additional computation - the trained policy is re-evaluated
against instability drives it never trained on, each tile integrating its own
plasma field and running its own closed loop, and the reported wall load is the
count that evaluation produced.

### Interactive view

Completed fusion runs also write `fusion_view.json`, a compact copy of the
final field texture and tracer or marker state used by the browser's rotatable
view. The optional magnetic view draws nested helical curves and the magnetic
axis to explain toroidal confinement; in guardian mode those curves follow the
axis the policy is holding and respond to its commands. They respond to the
selected field strength through an illustrative pitch mapping, but they are
**not** magnetic field lines calculated by the complex-amplitude solver and
must not be presented as a solved Grad--Shafranov equilibrium or safety-factor
profile.

## Storm Factory / weather ensemble

The atmosphere is a reduced barotropic-vorticity model coupled to an advected
moisture scalar on a latitude/longitude grid. A spectral Poisson inversion
recovers a streamfunction from vorticity; finite differences then advect
potential vorticity and moisture. This is a legitimate reduced geophysical
fluid model and the reveal members really do start from different smoothed
initial perturbations.

It is not an operational weather forecast. It has one vertical layer, stylised
forcing and damping, no assimilated observations, no topography and no full
thermodynamics. The continents are deliberately low-resolution procedural
geography used only for orientation. Forecast hours and warming controls are
part of the exhibition story, not calibrated predictions of a named storm.

## Molecular Machine / molecular dynamics

Coarse-grained 3-D Langevin dynamics in reduced units, in two modes.

**Fold** is an off-lattice version of the HP model (Dill, 1985). Beads are
water-avoiding (H), water-loving (P) or charged. Consecutive beads are joined
by harmonic bonds with a bending stiffness. Every non-bonded pair has a
repulsive core; H-H pairs attract with a strength standing in for the
hydrophobic effect, and charges interact through a screened (Debye-Hückel)
potential whose range shrinks with salt. A BAOAB Langevin thermostat supplies
the random kicks and friction of implicit water. The results that matter are
qualitative and robust: H-rich chains collapse around an oily core, a chain
with no H stays a coil, opposite charges pair up, and heating unfolds the
core. All non-bonded pairs are evaluated, so the pair-evaluation count is real.

**Shuttle** is a cartoon rotaxane: a stiff ring threaded on an axle with two
binding stations and bulky stoppers. The switch only changes which station is
sticky. Nothing pushes the ring; it moves by thermal diffusion and is trapped
where binding is strong. This is the physics of switchable molecular shuttles
(Stoddart and co-workers; 2016 Nobel Prize in Chemistry).

Neither mode is a force field. There is no explicit water, no atom types, no
calibrated time, and no structure prediction: beads stand for groups of atoms.

## Neuro-Racers

The learning is real: every car's steering, throttle and brake come from a
tanh multilayer perceptron with the visitor's architecture, and the weights are
found by a genetic algorithm (per-generation elitism, tournament selection,
uniform crossover and Gaussian mutation with decaying strength). No gradients,
human driving data or hand-written driving rules are used. Fitness is the net
distance driven along the track centre line in a fixed time, with a small
crash penalty.

The car is a reduced top-down kinematic bicycle model: speed with drag,
engine braking and brakes, curvature proportional to steering, and a lateral
acceleration limit above which the car skids and loses speed. It is not tyre,
suspension or vehicle-dynamics modelling, and lap times are in simulated
seconds of this model only. Distance sensors are sphere-traced through a
precomputed signed-distance grid of the walls (1/24 world-unit cells), so a
ray's reading is accurate to about that resolution. The track compass reports
the bearing to a point 1.6 units ahead on the centre line.

CPU (NumPy) and CUDA (fused CuPy kernels) execute the same arithmetic; small
float differences can change which car finishes a lap first, so runs are
scientifically, not bitwise, equivalent. Reveal tiles are independent
evolutions of the same architecture from different seeds. Ghost cars are
earlier champions re-simulated deterministically on the current track.

## Bat vs Moth

Two populations with visitor-built architectures co-evolve with the same
genetic algorithm as Neuro-Racers (stronger tournaments for these noisier
rewards). The behaviours are learned, not scripted: nothing tells a bat to
turn toward the louder ear or a moth to click when a bat is close.

The sonar is a **reduced exhibition model, not acoustics**. When a bat calls,
each moth within 6 units returns an echo whose loudness falls linearly with
distance and with a cardioid gain for ears pointing 40° left and right; its
delay input is proportional to distance. Rock does not block or reflect sound
in the echo model (rock is sensed only by short whisker rays), there is no
frequency content, and "Doppler" is simply the closing speed of the strongest
echo. Moths hear calls within 9 units, reflecting that real moths detect bats
before bats detect them, and hold the call's loudness and bearing for a few
steps.

Jamming is modelled on tiger-moth (e.g. *Bertholdia trigona*) clicks: a
jamming moth within 3 units of a calling bat replaces its own echo with a
phantom of random loudness and delay. Jamming costs survival fitness in
proportion to how long it is used. A power dive is a short, erratic burst of
speed with a rest period. Moths only begin evolving after a head start, and
start with jam and dive switched off; both are presentation choices so the
arms race can be seen within an exhibition run, and they are stated in the
viewer. "Jamming evolved" is reported only for selective jamming (much more
often with a bat near than far).

Bat fitness counts catches (earlier is better) plus a term for flying toward
the nearest audible moth, minus small costs for calls and wall hits. Moth
fitness is survival time minus time spent close to the bat and energy spent
jamming or diving. Speeds and ranges are in abstract cave units and seconds;
they are not calibrated to any bat or moth species.

## MUrB N-body (NBody-EuroHPC)

The physics is entirely the external MUrB code: an O(N²) direct gravitational
sum with softening, fp32, advanced by MUrB's own integrator. The dashboard
only launches it and draws the recorded positions. Initial conditions are
MUrB's: `galaxy` is a rotating shell of light bodies (1–2 × 10⁸ m from a
2 × 10²⁴ kg central mass), `random` a cold cloud offset along z. These are
code-demonstration setups, not models of a real galaxy, and the units are
metres and kilograms as MUrB uses them.

Colour follows MUrB's viewer: speed² normalised per frame, deep blue → cyan →
white. Two deliberate differences: MUrB's beat-synchronised "strobe" flash is
left out, and slow bodies get a brighter blue floor so they stay visible in
compressed frames. Bodies with zero radius (the galaxy's central mass) are not
drawn, as in MUrB.
