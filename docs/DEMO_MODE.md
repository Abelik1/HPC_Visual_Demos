# Demo-day mode

`http://127.0.0.1:8000/demo`

A second front end over the same server, the same `/api` endpoints and the same
`runs/` directory as the full dashboard at `/`. Nothing is duplicated: it is a
different set of decisions to put in front of a visitor, not a different
program. Anything you set up in one is visible in the other.

## What it changes

| Full dashboard | Demo mode |
| --- | --- |
| The active demo day's lineup, categories, search, archive tab | The same lineup as large cards, in the lineup's order |
| Every parameter as a slider | Two or three headline parameters as large value boxes and named buttons |
| Overlay dock on the right, each card toggled on | One always-visible caption: what is happening, three live numbers, how to read the picture |
| Quality preset, compute, frames, solver, ensemble and profile values all on the page | All of it behind the ⚙ presenter drawer |
| The reveal, and the ensemble width, in the run panel | The population as two big buttons, and only on the demos where it is the experiment |

## Two demo days

Which demos appear, and in what order, is the active **demo day** (Discoverer
or Leonardo) from `config/lineups.json`. Switch it with the buttons on the
picker, and edit the lists under **⚙ Lineups & HPC**. Demos without a
hand-written entry in `KIOSK` get a default one built from their first three
parameters. The drawer's **Run on** sends *Run it* to a cluster after a
confirmation. See `docs/CLUSTER_RUNS.md`.

## The scale reveal, reconsidered

The old reveal answered "how many of these can we run at once?" on every demo.
For a large model that is the wrong claim: the interesting compute lives inside
the single simulation, not in the tile count.

**This applies to the full dashboard as well**, not only demo mode. There the
rule is two sets at the top of `web/app.js`: `revealDemos` is where the button
appears (plus Star in a Bottle in guardian mode), and `ensembleDemos` lists
every demo whose solver honours an ensemble width. Anything in the second set
but not the first is pinned to `parallel_count = 1`, so it neither offers the
reveal nor spends time computing one. Move an id between the two sets to change
that for a demo. The button is now labelled for what it actually shows — "show
every network", "show every search", "show every cave", "re-test on unseen
drives" — rather than "show scale reveal".

In demo mode specifically:

* **Black hole, wind tunnel, cosmic web, both galaxy collisions** run exactly one
  simulation (`parallel_count = 1`) and have no reveal control at all.
* **Neuro-Racers** and **Bat vs Moth** — where hundreds of agents really are
  being evolved in parallel — show the population in two ways:
  * **Training** — the whole recorded population in one arena, overlaid.
  * **One per box** — the same training, with each of the best cars (up to
    16) on its own copy of the track, or each of the best caves (9) in its
    own box.
* **Neural-network wall** puts the same idea as **Networks trained at once**
  (1 · 4 · 16 · 36 · 64). The run itself does the reveal: it shows the champion
  first, then pulls back to the whole wall.
* **Star in a Bottle** keeps its two modes as the headline choice — passive
  confinement, or the AI plasma guardian.

The population buttons default to whatever the current quality preset already
asks for, and the stand's last choice is remembered in the browser.

## Before the day: curate the showcase

1. Start the viewer as usual (`python app.py`, or `Run_Leonardo_Demos.bat`).
2. Run the big, slow, impressive versions ahead of time — on the dashboard at
   `/` or on the Leonardo nodes — and **★ star** the good ones. Stars show on
   the dashboard's saved runs (with a **★ Favourites** filter), on each demo's
   replay chips, and in demo mode's history. They are stored by the server in
   `runs/_library.json`, so both front ends and every browser see the same set.
3. In `/demo`, open ⚙ on each demo. **Showcase runs** lists that demo's saved
   runs, favourites first, with thumbnails and their headline settings. Click
   up to three, in the order they should play.
   With none picked, the showcase falls back to your favourites, then to the
   newest finished run, so there is always something to play.
4. Set the quality preset for the machine you are on. `local` is a good
   default; `desktop` looks better and takes longer.

### Showcase renders

`scripts/render_showcase_desktop.py` renders two large variants of each
desktop demo (black hole, cosmic web, 2D galaxy collision, image compression,
Neuro-Racers, Bat vs Moth) one at a time on the PC's GPU, and collects the
Discoverer runs (3-D galaxy collision, wind tunnel, Star in a Bottle; see
`docs/DISCOVERER.md`). Every finished run is starred and added to its demo's
showcase. It is resumable: completed runs are skipped.

```bash
python scripts/render_showcase_desktop.py --skip-discoverer   # local queue only
python scripts/render_showcase_desktop.py --skip-local        # collect Discoverer runs
```

Progress is in `runs/_showcase_desktop.log`, and each run's output in
`runs/_showcase_<name>.log`.

## On the day

* **▶ Start slideshow** (on the picker) cycles every demo's showcase runs back
  to back. Each run plays for at least the **slideshow dwell** set in the
  drawer, looping if it is shorter. When someone walks up, **Try this one**
  keeps that demo on screen and hands them the controls.
* **▶ Showcase** (on a demo) does the same for just that demo's picks, with
  ‹ › to step between them. **Try it yourself** returns to the controls.
* **When nobody has touched it for 3 minutes** (drawer): do nothing, go back to
  the picker, or start the slideshow. Off by default so it cannot interrupt a
  presenter.
* **Playback speed**, under the play bar, runs from 0.1× to 4×. It changes how
  quickly the saved frames are shown, never the simulation; it also applies to
  the slideshow and the comparison view.
* **History**, under each demo, lists every saved run of it. Click a thumbnail
  to put that run back on screen; tick two and press **Compare** to play them
  side by side, synchronised start to finish, above a table of their settings
  with every difference highlighted.

A demo opens on its first showcase run, so a visitor sees motion immediately.
**Run it** computes a fresh simulation with their settings; frames appear as
they are written. The action bar is sticky, so **Run it** stays reachable.

Things visitors make themselves are never hidden in demo mode: the wind-tunnel
obstacle grid, the neural wall's own picture (draw, upload or webcam), the
brain builders, and Neuro-Racers' ghost races. Every parameter without a big
control is still available under **More parameters** in the drawer.

The galaxy collisions draw twinkling background stars over the picture. They
are decoration, and the caption says so.

**Nothing tells the story by itself.** Captions describe what is on screen
and never change as frames play; views only switch when you press a button.
The narrative lives in ⚙ → **Talking points**, for you to tell when you choose.

**Full screen**: click the picture (a click, not a drag, so the 3D views still
rotate) or press `F`. Moving the mouse brings up play, **Loop**, seek, speed,
the view buttons and the run's details along the bottom. If a browser refuses
real full screen, the picture fills the window instead. **Loop** is also next
to the play button; with it off, playback stops on the last frame.

## Per demo

* **Neuro-Racers and Bat vs Moth** keep training and inference apart, each
  with its own controls.
  * **Training** (the controls above *Run it*) sets how the visitor's brain
    is trained: track or cave, moths, mutation, ghosts and a **name tag**.
    Every visitor gets the same number of generations (the quality preset's),
    so brains compete on design alone; the heading says how many.
  * The **Training** view replays the evolving population one generation at a
    time, each drive playing to its end before the next starts. A drive ends
    early, after a two-second hold, once every car on screen has crashed or
    stalled, or every moth on screen has been caught, so the hopeless early
    generations go by quickly. **One per box** is the same training split into
    boxes. Bat vs Moth shows both in the bat's senses or the lit cave.
  * **Champion drives / hunts** is inference: the saved champion network,
    frozen, from a random start it has never seen (Bat vs Moth: in the senses
    or the lit cave). The **test world** right under the picture changes the
    world, never the network, and re-runs the champion at once: another track
    or all four at once; another cave layout, more or fewer moths, or 1 · 4 ·
    9 · 16 caves at once, one per box. This is where a brain that learned its
    training world by heart shows it. Neuro-Racers can also race the first,
    middle and final champions against each other, or **race the ghosts**: the
    best saved champions of other visitors on that track, from the same start,
    each car with its owner's name tag.
  * The live network and a training chart sit under the picture, never over
    it. Bat vs Moth runs made before this version have no One per box view.
* **Star in a Bottle**: the Mode 1 / Mode 2 buttons also switch the picture to a
  saved run of that mode (and opening a saved run switches the buttons), and a
  label on the picture says which mode is on screen.
  * **Mode 1**'s 3D torus draws its tracers the way the flat view does (glow,
    ribbons, white heads). A big run draws a quicker pass while you rotate or
    play and the full picture once the view is still.
  * **Mode 2 makes the trained controller the exhibit.** Train one good
    controller ahead of time (a long run; every shot's controller is saved in
    `checkpoints/`) and make it the showcase. Visitors do not train: the
    training controls and *Run it* are replaced by a note, unless ⚙ →
    *Visitors can train a new Star in a Bottle guardian* is ticked.
  * **Test the controller** (the default view of a guardian run) flies the
    final controller, frozen, live: the test panel under the picture sets the
    magnetic field, heating and instability drive; **Compare with no control**
    puts the same plasma with the coils left alone beside it; **Plasmas at
    once** (1 · 4 · 9 · 16) gives each box a harder instability drive. Each
    box counts its wall hits. Tests run on the server's CPU at a small test
    resolution and come back in about a second.
  * **How it learned** flies every training shot's controller side by side in
    the training conditions, with no control last.
  * **3D torus** and **Flat view** replay the training run itself. Its three
    panels underneath show the frozen network flying the shot, the scoreboard
    of training between shots, and the cross-section, above a timeline of
    shots and training steps. Guardian runs made before this version saved no
    controller, so they have no test views: re-render the showcase.
* **Neural image compression** (formerly the neural-network wall): pick a
  default picture, upload, take a photo or draw, choose how small to squeeze it
  (32 to 128 px), and read size in, network size, compression ratio and quality
  underneath. **In → out** and **All networks** are both recorded for every
  frame. The winner is the best network that is actually smaller than the
  picture, and a JPEG of the same size is reported alongside it. Add your own
  default pictures to `data/compression_images/`.
* **Cosmic web**: **Recipe of the Universe** compares our universe with one
  that stays radiation-dominated and one without dark matter. Measured at the
  `local` preset, density contrast grows from 0.7 to 5.1 in ours, to 1.4 with
  radiation in charge, and not at all without dark matter. It is a qualitative
  model and the caption says so.
* **Galaxy collisions**: the 3D view now draws particles in the same style as
  the flat frames.

Keyboard:

* `Space` — play / pause.
* `F` — full screen.
* `Esc` — leave full screen, close the comparison or the drawer, stop a
  presentation, or go back to the picker.
* `/demo?demo=<id>` — open straight into one demo, e.g. `?demo=fusion_plasma`.

## Changing what the stand shows

Everything demo-specific is one object at the top of `web/demo.js`:

```js
const KIOSK={ fusion_plasma:{ tag, blurb, read, story:[…], controls:[…], views:[…], numbers:[…] }, … };
const ORDER=['fusion_plasma','neuro_racers', …];
```

* `ORDER` is the picker order — delete an entry to drop a demo from the stand.
* `controls` entries name a parameter from `config/demo_specs.json`. A slider
  becomes a value box; anything with `choices` or a `choice` kind becomes big
  named buttons; `{method:true}` is the solver as a headline choice and
  `{population:true}` is the population control described above.
* `creator` mounts a visitor-made input above the controls: `'obstacles'`
  (wind-tunnel grid) or `'target'` (the neural wall's own picture). Demos with a
  brain catalogue get their builders automatically. `{ghosts:true}` adds the
  ghost-race toggle; `starfield:true` adds the decorative background stars.
* `about` is the caption; `story` holds the talking points shown in the drawer;
  `read` is the fixed "how to read this picture" line; `numbers` names which
  live readouts to show, in order of preference.

Adding a demo needs no server change — only an entry in `KIOSK` and `ORDER`.

## MUrB N-body and recorded videos

`nbody_murb` (the external NBody-EuroHPC code; see [NBODY_MURB.md](NBODY_MURB.md))
is in the picker after the 3D galaxy collision. Visitors choose the
implementation (OpenMP, SIMD, optimised, naive reference), the starting setup
and the timestep; the readout shows the code's own GFLOP/s, which makes the
implementations easy to compare side by side with History → Compare.

The last tile, **Recorded videos**, opens `/videos?from=demo`, which plays
whatever is in `videos/` (or `LEONARDO_VIDEO_DIR`) and offers a "Back to demos"
link. It is not part of the slideshow.
