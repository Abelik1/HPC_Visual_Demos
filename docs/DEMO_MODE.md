# Demo-day mode

`http://127.0.0.1:8000/demo`

A second front end over the same server, the same `/api` endpoints and the same
`runs/` directory as the full dashboard at `/`. Nothing is duplicated: it is a
different set of decisions to put in front of a visitor, not a different
program. Anything you set up in one is visible in the other.

## What it changes

| Full dashboard | Demo mode |
| --- | --- |
| 14 demos, categories, search, archive tab | 9 demos as large cards, in a fixed order |
| Every parameter as a slider | Two or three headline parameters as large value boxes and named buttons |
| Overlay dock on the right, each card toggled on | One always-visible caption: what is happening, three live numbers, how to read the picture |
| Quality preset, compute, frames, solver, ensemble and profile values all on the page | All of it behind the ⚙ presenter drawer |
| The reveal, and the ensemble width, in the run panel | The population as two big buttons, and only on the demos where it is the experiment |

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
  being evolved in parallel — put the population on the page as a visitor
  control, with two ways to look at it:
  * **All together** — the whole recorded population in one arena, overlaid.
  * **One per box** — one champion per box, one box per independent search.
    The number of boxes is the **Independent searches / caves** control.
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

* **Neuro-Racers and Bat vs Moth** have three views. **Training** replays the
  evolving population one generation at a time, and each drive plays to its
  end before the next generation starts. **Champion drives / hunts** is
  inference: the saved champion network, frozen, driving from a random start
  it has never seen (Neuro-Racers can also race the first, middle and final
  champions against each other). **One per box** animates every independent
  search side by side. The live network and a training chart sit under the
  picture, never over it. Runs made before this version show one per box as a
  still image; run them again to animate it.
* **Star in a Bottle**: the Mode 1 / Mode 2 buttons also switch the picture to a
  saved run of that mode, and a label on the picture says which mode is on
  screen. Guardian runs show three panels underneath: the frozen network flying
  the shot (inference), the scoreboard of training between shots, and the
  cross-section, above a timeline of shots and training steps.
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
