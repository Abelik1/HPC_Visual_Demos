# Presenter stories

These are short scripts, not mandatory narration. The UI is designed so the story remains understandable even without audio.

## 1. Black-hole lensing

**Opening:** "This is an ordinary view of the sky. Now put a black hole between us and the background."

As lensing appears: "Every output pixel is a question: trace this line of sight backwards and work out where its light came from."

**Reveal:** "That was one observer. A supercomputer lets us ask the same question from many observers or with many black-hole parameters at once."

## 2. Primordial black hole

**Opening:** "The Universe is extremely young. We change one tiny density perturbation."

Below threshold: "Pressure wins; the perturbation disperses."

Above threshold: "Gravity wins; the central region grows increasingly dense and the solution heads toward collapse."

**Reveal:** "Scientists do not run one universe. We scan a whole neighbourhood and find the boundary where the answer changes."

## 3. Wind tunnel

**Opening:** "At first the flow is almost boring."

As vortices appear: "The obstacle forces the fluid to organise into vortices, and eventually the wake becomes complex."

**Reveal:** "The smooth picture hides a grid of cells being updated over and over. A large domain can be split across GPUs, with only boundaries exchanged."

## 4. Cosmic web

**Opening:** "The early distribution is almost uniform. Almost is the important word."

**Reveal:** "Run the history again with a different tiny fluctuation and you get a different detailed universe. HPC lets cosmologists explore the statistics, not just one pretty animation."

## 5. Galaxy collision

**Opening:** "Two galaxies can start almost peacefully."

At first passage: "The long tails are a memory of the orbit."

**Reveal:** "Change the impact parameter or speed and the entire encounter changes. Instead of guessing one orbit, calculate an atlas."

## 6. Reaction diffusion

**Opening:** "There are only two interacting fields and local rules."

**Reveal:** "One set of constants makes this pattern. But the beautiful part is the parameter space: many mathematical worlds evolving at once."

## 7. Crystal growth

**Opening:** "Every crystal begins from a tiny seed."

**Reveal:** "Change the environment by a little and the branches change. Rather than grow one crystal, scan conditions and build a morphology map."

## 8. Neural-network wall

**Opening:** "This network is learning to reproduce an image from coordinates."

Wait until the reconstruction is recognisable.

Then: "Actually… I forgot something."

As the wall appears: "That was only the best network. We were training many different networks. Different initial weights, widths and learning rates. We were not just training a model — we were searching for one."

## 9. Star in a Bottle

**Opening:** "A fusion plasma is hotter than the centre of the Sun, so no material wall can hold it. Magnetic fields have to make the bottle."

As the luminous trails move: "These are passive tracers following drift derived from the simulated wave field. Heating feeds the plasma, nonlinear waves interact, and coherent motion can turn into turbulent structure."

**Reveal:** "A reactor is not designed from one beautiful plasma. We compute an operating map — magnetic field against heating power — and search for regions that remain controllable."

**Interactive view:** "Drag the plasma to inspect the torus from any direction. Switch to Magnetic field to see the helical geometry that explains how toroidal and poloidal field components wrap around the bottle. This geometry is explanatory; the reduced wave model is not solving the reactor's magnetic equilibrium."

## 10. AI Plasma Guardian

**Opening:** "In a real tokamak, the plasma has to stay away from the wall. A small instability can grow faster than a human can react."

As the red outline grows: "The red line is an uncontrolled reference. The bright plasma receives noisy diagnostics: position, velocity, pressure and a tearing-risk proxy."

As the coils respond: "A neural policy turns those readings into three magnetic-coil commands. It has learned that response by testing batches of virtual plasma shots in parallel."

**Method note:** "This is a reduced control environment, not a prediction of a reactor disruption. The graph shows actual learned policy weights and the coloured coils show its actual output."

## 11. Storm Factory

**Opening:** "This globe starts from our best estimate of the atmosphere now. The clock is racing five days ahead."

As the storm moves: "Vorticity carries rotating weather systems while moisture is transported around the planet."

**Reveal:** "The observations are never exact. Each globe begins with a slightly different but plausible atmosphere. A useful forecast needs the whole ensemble before the real weather arrives."

## 12. Molecular Machine

**Opening:** "At this scale, a molecule never sits still. Every particle pulls on its neighbours while temperature shakes the entire structure."

As the chain rearranges: "This is one trajectory through an enormous space of possible shapes."

**Reveal:** "Change temperature, attraction or sequence and the answer changes. A supercomputer turns one slow experiment into a virtual laboratory of molecular trajectories."

## 13. Neuro-Racers

**Invite:** "Come and build a brain. You have 40 LEGO points: which senses does your car get? How many neurons? Can it brake?"

**Opening (generation 1):** "Every car has your brain, but random wiring. Watch: most spin in circles or hit the wall."

As laps appear: "Nobody told them how to drive. The furthest cars became parents; their children got slightly mutated copies of the wiring. Now look at generation ten."

**Brain graph:** "These are the actual weights your car evolved. A sensor that stays dark is one evolution decided to ignore."

**Challenges for the next visitor:** "Can a brain with no hidden layer win? Can you beat the ghost of the last visitor? What happens if every sensor points left?"

**Reveal:** "That was one search. Here is the same brain evolved from different random starts: some find a fast lap, some never do. Leonardo runs thousands of these searches at once, and that is how real AI research explores designs."

## 14. Bat vs Moth

**Invite (two visitors):** "One of you builds a bat, the other a moth. Bat: you are blind, but you can hear the echoes of your own calls. Moth: you can hear the bat coming. And you have a secret weapon."

**Opening (bat's senses):** "This is all the bat knows. It shouts, and the echoes come back: that ring lights up the rock, and those glows are moths."

During the head start: "Only the bats are evolving so far. The ones that turn toward the louder ear catch more moths. Look at the arms-race chart climbing."

When moths start evolving: "Now the moths fight back. Watch for magenta: a moth clicking so that its echo appears somewhere it isn't."

**Challenge:** "Bat visitor: try taking an ear away. Moth visitor: what happens without the jamming block?"

**Punchline:** "Real tiger moths jam bat sonar with ultrasonic clicks. Evolution found that trick millions of years ago. Here it has to rediscover it inside a supercomputer."

**Reveal:** "Now the same bats and moths are released into caves they have never seen. Does the jamming survive? Every tile is its own evolving world, and Leonardo runs thousands of them side by side."
