# Demo day: setting up the stand laptop

Everything needed to make a laptop the machine that runs the stand, starting
from nothing. Written for the Leonardo demo day; the Discoverer day is the same
with a different lineup.

The laptop needs **three things that git does not carry**: the saved runs, the
videos and (only if demos will be run live) the Gaia sky. Plan for about
20 GB and an evening of uploading and downloading.

## 1 · On the machine that has the runs

```bash
python tools/export_runs.py
```

The zip lands in `exports/leonardo_runs_<host>_<date>.zip` and holds every
starred run, every showcase pick and every cluster run, with the stars and
showcase picks themselves. `--list` prints the selection and its size first;
`--no-cluster` leaves out the small cluster pilot runs.

Upload to Google Drive, together with:

- the whole `videos/` folder (about 1 GB: the ray-tracer films, and Gianluca's
  Pac-Man films when they arrive);
- `data/gaia_sky.npz` (35 MB) — or skip it and fetch it on the laptop, below.

## 2 · On the laptop

Python 3.10 or newer, and about 25 GB free.

```bash
git clone -b feat/demo-days-cluster-runs https://github.com/Abelik1/HPC_Visual_Demos.git
cd HPC_Visual_Demos
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The `-b` matters: the demo-day work (both lineups, the Molecular Machine
motors, the plasma guardian fix, bundle unpacking under any launcher,
`check_stand.py`) is on `feat/demo-days-cluster-runs`. Until that branch is
merged, a plain `git clone` gets `main`, which has none of it. `git log -1`
should show `config(lineups): open on the Leonardo demo day` or later.

That is all the viewer needs: NumPy, Pillow, FastAPI, uvicorn and pydantic. A
GPU is **not** required — everything on the stand is a replay of a run made on
Discoverer, Leonardo or the workstation, and the demos a visitor starts run on
the CPU at the Local preset.

Then put the downloaded files in place:

| From Drive | Goes to |
|---|---|
| `leonardo_runs_*.zip` | `runs/_import/` (create the folder) |
| the `videos/` folder's contents | `videos/` |
| `gaia_sky.npz` | `data/` |

If you skipped the Gaia file: `python tools/fetch_gaia_sky.py` downloads it.
It is only needed to *run* the black-hole demo live; replaying the saved runs
does not touch it.

## 3 · Start it

```bash
python app.py
```

It prints the address, normally `http://127.0.0.1:8000`, and opens a browser.
The first start unpacks the zip in the background — a 20 GB bundle takes a few
minutes, and runs appear in the gallery as they land. The zip is remembered, so
it is only ever imported once and can be deleted afterwards.

Any launcher works (`uvicorn app:app`, `Run_Leonardo_Demos.bat`): unpacking and
picking up cluster jobs happen on application startup, not in one entry point.

## 4 · Two displays

One server, two browser windows, one per display:

| Display | Address | What it is |
|---|---|---|
| The one the public sees | `http://127.0.0.1:8000/demo` | the walk-up viewer: big controls, the explanation under the picture, everything advanced behind the presenter panel |
| The one you drive | `http://127.0.0.1:8000/` | the full dashboard: every parameter, saved runs, cluster runs, stats |

Both talk to the same server, so a run started on one appears on the other.
Put the public window in full screen (F11). Two dashboards work equally well if
that is what you prefer — nothing is single-window.

**Pick the demo day** on either page (the Demo day buttons on the picker) so
the lineup shows the Leonardo demos.

## 5 · Check it before the doors open

Run the check first — it reads what the stand will actually play and touches
nothing:

```bash
python tools/check_stand.py --machine all
```

It reports, per demo, whether the showcase runs are there and complete, whether
their frames are all on disk, whether the 3-D views and overlays came with
them, and whether the video-only entries have their films. It is how the
missing MUrB frames and two runs left marked "starting" were found the evening
before.

Then, by hand:

- [ ] Every demo in the lineup opens and plays its showcase run.
- [ ] The videos page plays the ray-tracer films (and Pac-Man, if they arrived).
- [ ] "Rotate in 3D" works on the galaxy, the molecule and the fusion torus.
- [ ] Start one live run at the Local preset (the Molecular Machine walker is a
      few seconds) so you know the laptop can compute, not only replay.
- [ ] Turn off sleep, screen blanking and notifications.
- [ ] Battery: the stand should be on mains — a live run will use every core.

## If something is missing

| Symptom | Cause |
|---|---|
| Gallery is empty | the zip was not in `runs/_import/`, or it is still unpacking — watch the terminal |
| A demo shows "Play recorded video (none yet)" | the `videos/` folder did not come across |
| Black hole refuses to start a live run | `data/gaia_sky.npz` is missing; saved runs still replay |
| A cluster button is offered but fails | the laptop has no CINECA certificate (see below); the stand does not need one |

## Optional: launching Leonardo jobs from the laptop

Only if the stand should start cluster runs. Once per laptop:

```powershell
winget install Smallstep.step
# elevated PowerShell:
Set-Service -Name ssh-agent -StartupType Automatic; Start-Service ssh-agent
```

Then on the morning (the certificate lasts 12 hours) — **without a passphrase**,
because the dashboard's `ssh` cannot type one:

```powershell
.\scripts\leonardo_login.ps1 -Email abelik@tcd.ie -User abelik00 -CertOnly -NoPassword
```

The dashboard connects to `login01-ext.leonardo.cineca.it`, whose host key a
fresh laptop does not know; verify and add it as in docs/TROUBLESHOOTING.md
before relying on the button. Discoverer (`login.brainplusplus.bg:2226`) is
unreachable from some networks — if `ssh` times out, try another connection.

More: [DEMO_MODE.md](DEMO_MODE.md) for the walk-up viewer,
[DEMO_STORIES.md](DEMO_STORIES.md) for what to say at each demo,
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) for everything else.
