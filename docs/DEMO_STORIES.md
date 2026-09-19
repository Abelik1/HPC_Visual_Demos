# Presenter stories

These are short scripts, not mandatory narration. The UI is designed so the story remains understandable even without audio.

**A note on the reveal.** The "show every run" button now exists only where a
population is genuinely being trained or re-tested: the neural-network wall,
Neuro-Racers, Bat vs Moth, and Star in a Bottle in guardian mode. Everywhere
else a run is a single simulation, because for a large model the compute lives
inside that one simulation rather than in how many copies fit on screen. The
closing beat for those demos is the simulation itself, not a pull-back. Lines
below that used to end on a tile wall have been rewritten accordingly; see
[DEMO_MODE.md](DEMO_MODE.md).

## 1. Black-hole lensing

**Opening:** "This black hole is real. Gaia found it by watching a star orbit something we cannot see. We have parked a camera next to it."

As the view settles: "Every star in this sky is a real Gaia star, moved to where it would appear from out there. Nearby stars have shifted; the Milky Way is where it should be."

On the shadow: "The black disc is not the horizon. It is the shadow, two and a half times bigger: every ray aimed inside it falls in."

On the edge: "Right at the edge, light has gone once around the hole before reaching the camera, so you see the whole sky squeezed into a thin ring, and then again, and again."

Switch to the ray paths: "These are the exact routes. The colour is how hard gravity bent each ray: blue barely turned, yellow turned right round, and red never got out. The dashed ring is where light can orbit."

**Closing:** "Every pixel is one question: where did this light come from? The picture is the answer to millions of them, each solved exactly in Einstein's equations."

If someone asks what is missing: no spin, no glowing disc (these black holes are genuinely dormant), and no companion star.

## 2. Primordial black hole

**Opening:** "The Universe is extremely young. We change one tiny density perturbation."

Below threshold: "Pressure wins; the perturbation disperses."

Above threshold: "Gravity wins; the central region grows increasingly dense and the solution heads toward collapse."

**Closing:** "The interesting result is not either outcome. It is how sharp the boundary between them is: a fraction of a percent in the initial density decides it."

## 3. Wind tunnel

**Opening:** "At first the flow is almost boring."

As vortices appear: "The obstacle forces the fluid to organise into vortices, and eventually the wake becomes complex."

**Closing:** "The smooth picture hides a grid of cells being updated over and over. A large domain can be split across GPUs, with only the boundaries exchanged."

## 4. Cosmic web

**Opening:** "The early distribution is almost uniform. Almost is the important word."

**Closing:** "Nobody placed those filaments. Gravity amplified fluctuations that started a hundred thousand times smaller than what you are looking at."

## 5. Galaxy collision

**Opening:** "Two galaxies can start almost peacefully."

At first passage: "The long tails are a memory of the orbit."

**Closing:** "Change the impact parameter or the approach speed and the entire encounter changes. This is one orbit out of a range we genuinely do not know yet."

## 6. Reaction diffusion

**Opening:** "There are only two interacting fields and local rules."

**Closing:** "Two chemicals and one local rule. Move the feed and kill rates slightly and you get spots, stripes, waves or a labyrinth instead."

## 7. Crystal growth

**Opening:** "Every crystal begins from a tiny seed."

**Closing:** "Change the environment a little and the habit changes completely. Zoom in and the same growth rule is still running several scales down."

## 8. Neural-network wall

**Opening:** "This network is learning to reproduce an image from coordinates."

Wait until the reconstruction is recognisable.

Then: "Actually… I forgot something."

As the wall appears: "That was only the best network. We were training many different networks. Different initial weights, widths and learning rates. We were not just training a model — we were searching for one."

## 9. Star in a Bottle

**Opening:** "A fusion plasma is hotter than the centre of the Sun, so no material wall can hold it. Magnetic fields have to make the bottle."

As the luminous trails move: "These are passive tracers following drift derived from the simulated wave field. Heating feeds the plasma, nonlinear waves interact, and coherent motion can turn into turbulent structure."

**Closing:** "Raise the heating and the coherent motion breaks up. Holding that is the whole problem — which is mode 2."

**Interactive view:** "Drag the plasma to inspect the torus from any direction. Switch to Magnetic field to see the helical geometry that explains how toroidal and poloidal field components wrap around the bottle. This geometry is explanatory; the reduced wave model is not solving the reactor's magnetic equilibrium."

## 10. Star in a Bottle, mode 2: AI Plasma Guardian

**Opening:** "Same bottle, harder question. In a real tokamak the plasma has to stay away from the wall, and a small instability can grow faster than a human can react. So we fill the torus with markers and ask a neural network to hold them."

As the first sparks appear: "Every orange burst is a marker that escaped confinement and hit the vessel. The policy starts untrained: it is losing plasma."

As the coils respond: "The network receives noisy diagnostics — position, velocity, pressure and a tearing-risk proxy — and turns them into three magnetic-coil commands. It learned that response by back-propagating through batches of virtual plasma shots. Watch the sparks stop."

**The comparison:** "The readout is counting an identical population running with the coils switched off. It is already on the wall."

**Re-test on unseen drives:** "One trained controller, re-tested against instability drives it never saw. Each tile is its own plasma field and its own closed-loop evaluation, and the wall load is counted."

**Method note:** "This is a reduced control environment and a transport-flavoured marker model, not a prediction of a reactor disruption. The graph shows actual learned policy weights and the coloured coils show its actual output."

## 11. Storm Factory

**Opening:** "This globe starts from our best estimate of the atmosphere now. The clock is racing five days ahead."

As the storm moves: "Vorticity carries rotating weather systems while moisture is transported around the planet."

**Closing:** "The observations are never exact. Raise the initial uncertainty and watch how far the storm track has moved by day five: that is why a forecast has a confidence, not just a position."

## 12. Molecular Machine

**Invite:** "Write a protein. Oily beads, water-loving beads, plus and minus: what shape will it fold into?"

**Opening:** "At this scale nothing sits still. Water kicks every bead billions of times a second, and every bead pulls on every other one."

As it folds: "The oily beads hide from water together. That one rule is why real proteins have a core, and why a chain with no oil just flops about."

**Machine mode:** "This ring is threaded on an axle; the big beads stop it falling off. Watch the green station. When I flip the switch, nothing pushes the ring: it jiggles along on heat alone until the sticky station catches it. The 2016 Chemistry Nobel was for machines like this."

**Motor mode:** "Your cells are full of tiny walking motors carrying cargo. Here's the trick: the fuel doesn't push the feet. It just switches the track off for a moment. The feet jiggle randomly, and because every well has its steep wall right in front, they get caught a step ahead far more often than a step behind. Now add a load and watch it stall."

## 13. Neuro-Racers

**Invite:** "Come and build a brain. You have 40 LEGO points: which senses does your car get? How many neurons? Can it brake?"

**Opening (generation 1):** "Every car has your brain, but random wiring. Watch: most spin in circles or hit the wall."

As laps appear: "Nobody told them how to drive. The furthest cars became parents; their children got slightly mutated copies of the wiring. Now look at generation ten."

**Brain graph:** "These are the actual weights your car evolved. A sensor that stays dark is one evolution decided to ignore."

**Challenges for the next visitor:** "Can a brain with no hidden layer win? Can you beat the ghost of the last visitor? What happens if every sensor points left?"

**Show every search:** "That was one search. Here is the same brain evolved from different random starts: some find a fast lap, some never do. Leonardo runs thousands of these searches at once, and that is how real AI research explores designs."

## 14. Bat vs Moth

**Invite (two visitors):** "One of you builds a bat, the other a moth. Bat: you are blind, but you can hear the echoes of your own calls. Moth: you can hear the bat coming. And you have a secret weapon."

**Opening (bat's senses):** "This is all the bat knows. It shouts, and the echoes come back: that ring lights up the rock, and those glows are moths."

During the head start: "Only the bats are evolving so far. The ones that turn toward the louder ear catch more moths. Look at the arms-race chart climbing."

When moths start evolving: "Now the moths fight back. Watch for magenta: a moth clicking so that its echo appears somewhere it isn't."

**Challenge:** "Bat visitor: try taking an ear away. Moth visitor: what happens without the jamming block?"

**Punchline:** "Real tiger moths jam bat sonar with ultrasonic clicks. Evolution found that trick millions of years ago. Here it has to rediscover it inside a supercomputer."

**Show every cave:** "Now the same bats and moths are released into caves they have never seen. Does the jamming survive? Every tile is its own evolving world, and Leonardo runs thousands of them side by side."
