# Running on Leonardo

The compute side of this project is headless: it writes JPEG frames and JSON
metadata without OpenGL, X11, or a web server. This is the right shape for
Leonardo's batch nodes and remains compatible with CINECA's current mitigation
that disables Xorg on Booster GPU nodes.

The two supported allocations are intentionally separate:

| Mode | Partition | Job shape | Implementation |
|---|---|---|---|
| CPU | `dcgp_usr_prod` | 1 process, 32 CPU cores | NumPy plus chunked shared-memory tracer updates |
| GPU / hybrid | `boost_usr_prod` | 1 process, 1 A100, 8 CPU cores | CuPy solver; hybrid overlaps CPU JPEG encoding |

Leonardo Booster nodes have 32 CPU cores and four 64 GiB A100 GPUs. DCGP nodes
have 112 CPU cores and no GPUs. This demo asks only for the slice it can use;
it does not reserve a whole DCGP node or four A100s merely to make the resource
request look larger.

## 1. Get access

Two separate identities are involved, and confusing them costs a day. The
**UserDB account** (<https://userdb.hpc.cineca.it/>) is the portal where you
register and where a PI adds you to a project. The **HPC username** is what you
log in with: eight characters, the first letter of your given name plus the
first seven of your surname, so Alexander Belik becomes `abelik00`.

Each step unlocks the next, so the order is not negotiable:

1. Register on UserDB, upload an identity document, sign the access policies.
2. Ask the project PI to add you as a collaborator.
3. Press **Submit** on the UserDB "HPC Access" page. Credentials arrive by email
   within about a day, with a 2FA activation link that expires after 12 hours.
4. Activate two-factor authentication at <https://sso.hpc.cineca.it> and scan the
   QR code into an authenticator app. Keep the recovery codes.

Leonardo does not use `authorized_keys`. An ordinary keypair is refused however
it is configured, because the cluster trusts one CINECA certificate authority
and accepts only certificates that authority signed. There is nowhere to upload
a public key. Install the smallstep client and pin the CA once per machine:

```bash
winget install Smallstep.step        # or the platform equivalent
step ca bootstrap --ca-url=https://sshproxy.hpc.cineca.it \
  --fingerprint 2ae1543202304d3f434bdc1a2c92eff2cd2b02110206ef06317e70c1c1735ecd
```

Then, **once every twelve hours**, request a certificate and connect:

```bash
step ssh login 'you@example.org' --provisioner cineca-hpc
ssh abelik00@login.leonardo.cineca.it
```

The identity passed to `step ssh login` is the email registered on UserDB, not
the HPC username. `step ssh login` loads the certificate into `ssh-agent`, which
must be running; on Windows that service ships disabled. Enabling it needs one
elevated command, and `scripts/leonardo_login.ps1` wraps the whole flow —
including a `-CertOnly` mode that writes the certificate to a file for `rsync`
and `scp`, and for machines where the agent is unavailable.

**Issue the certificate without a passphrase** on any machine that launches
cluster jobs — the dashboard, the demo stand, scripts:

```powershell
.\scripts\leonardo_login.ps1 -Email abelik@tcd.ie -User abelik00 -CertOnly -NoPassword
```

By default `step` encrypts the new private key (`~\.ssh\cineca_leonardo`) with
a passphrase. The dashboard's `ssh` runs in batch mode and cannot type it, so
Leonardo *accepts* the certificate and then refuses the login — `ssh -v` shows
`Server accepts key` followed by `Permission denied (publickey,...)`.
`-NoPassword` writes the key unencrypted. That is the intended trade: the key
is useless without a certificate, the certificate dies after 12 hours, and
getting a new one needs your password and OTP. If you would rather keep the
passphrase, load the key once per certificate with
`ssh-add $HOME\.ssh\cineca_leonardo` (the agent must be running) — but that is
one more step to forget on demo morning.

`login.leonardo.cineca.it` is round-robin DNS over four login nodes
(`login01/02/05/07-ext`), each with its own host key. Reaching a node you have
not used before therefore prints a host-key warning that is not an attack.
Verify rather than wave it through: see `docs/TROUBLESHOOTING.md`.

## 2. Configure the account and paths

A project account is not a directory you enter. It is two things: a **storage
area** (`$WORK` and `$FAST`, one pair per project) and a **billing identifier**
passed to SLURM as `-A`. Logging in leaves you in `$HOME`, which is correct;
there is no further step.

Booster and DCGP budgets are **separate accounts** and each may only be spent on
its own partition, so check both:

```bash
saldo -b            # Leonardo Booster
saldo -b --dcgp     # Leonardo DCGP
chprj -l            # projects you belong to, and which is default
cindata             # occupancy of every area you can write to
```

`chprj -d <account>` repoints `$WORK` and `$FAST` when you belong to more than
one project. Until a PI associates you with a project, `$WORK` reads
`/no/project/defined`, `$FAST` is empty, and `saldo` reports "username not
existing" — you can log in and compile, but not submit. Only the PI can fix it.

| Area | Quota | Scope | Lifetime |
|---|---|---|---|
| `$HOME` | 50 GB | user | permanent (backup currently suspended) |
| `$PUBLIC` | 50 GB | user, world-readable | permanent, no backup |
| `$SCRATCH` | none | user | files deleted after 40 days |
| `$WORK` | 1 TB default | project | permanent, no backup |
| `$FAST` | 1 TB default | project, fast I/O, Leonardo only | permanent, no backup |
| `$TMPDIR` | see below | job | removed at job end |

Then copy the example and set the real project-account value:

```bash
cp config/leonardo.env.example config/leonardo.env
```

`config/leonardo.env` is ignored by Git. The submission wrapper requires an
explicit `LEONARDO_ACCOUNT`; it will not silently charge a training account
copied from a slide deck. The SLURM account name may differ from the UserDB
AccountID — EuroHPC projects carry an `EUHPC_` prefix that the portal omits.
`sacctmgr show associations user=$USER format=account,partition,qos` prints the
exact string to use, and returns nothing at all if you have no association yet.

Output defaults to `$FAST/leonardo_visual_demos`, which is shared and suited to
the frame stream. Booster nodes are diskless and expose only a fixed 10 GiB
RAM-backed `$TMPDIR`, so the job does not stage a complete run there. DCGP nodes
have local SSD and can request up to 3 TB with `--gres=tmpfs:<size>`, but that
request is charged against the budget like any other resource.

## 3. Prepare one tested Python environment

Build and install on a login node — CINECA permits compilation there, and it is
the only place with outbound network access. Login nodes have **no GPUs** and a
**10-minute CPU-time limit** per process, so keep parallel builds modest and run
nothing large.

CPU-only setup:

```bash
module purge
module load profile/base
python3 -m venv "$WORK/venvs/leonardo-visual-demos"
source "$WORK/venvs/leonardo-visual-demos/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the CuPy demos on Booster, load Leonardo's CUDA 12.2 module and install the
CUDA 12 wheel family:

```bash
module load cuda/12.2
python -m pip install -r requirements-gpu.txt
```

Do not install the CUDA 13 Windows lock on Leonardo. `requirements-gpu.txt`
contains CuPy for the CUDA 12 family; `requirements-gpu-windows.txt` is only
for the separately tested exhibition workstation. The neural-network wall is
the sole PyTorch demo. If it must train on a Leonardo GPU, prefer a CINECA
`cineca-ai` module and create the environment with `--system-site-packages`.

An alternative is a pre-built Singularity image. See `container/README.md` and
set `LEONARDO_CONTAINER` instead of `LEONARDO_VENV`.

## 4. Submit and validate

The fourth wrapper argument chooses the allocation and backend:

```bash
# DCGP shared-memory CPU run
bash scripts/submit_leonardo.sh galaxy_collision 90 leonardo cpu

# Booster: solver on one A100
bash scripts/submit_leonardo.sh galaxy_collision 90 leonardo gpu

# Booster: A100 solver plus overlapping CPU frame encoding
bash scripts/submit_leonardo.sh galaxy_collision 90 leonardo hybrid
```

The wrapper selects `slurm/run_demo_cpu.sbatch` for CPU work and
`slurm/run_demo.sbatch` for GPU/hybrid work. Both bind the process to its
allocated cores, propagate `SLURM_CPUS_PER_TASK`, write a unique job-id run
directory, and execute `tools/leonardo_preflight.py` before the simulation.
The preflight verifies imports, output writability, and—on Booster—the actual
CuPy-visible device. Explicit GPU requests fail rather than silently falling
back to NumPy.

The example config uses the 30-minute debug QoS values. The job templates also
request 30 minutes. For production runs, select a QoS allowed for the project
and change the time limit in the job file if necessary.

| Partition | Debug QoS | Walltime | Ceiling |
|---|---|---|---|
| `boost_usr_prod` | `boost_qos_dbg` | 00:30:00 | 2 nodes / 64 cores / 8 GPUs |
| `dcgp_usr_prod` | `dcgp_qos_dbg` | 00:30:00 | 2 nodes / 224 cores |

Budget is charged as *equivalent reserved cores × elapsed time*, and the
equivalent-core count is the **maximum** of the cores requested, the GPUs
requested × 8, and the memory requested divided by memory-per-core. On Booster
that means one A100 costs at least eight cores whatever else you ask for, so the
GPU template's `--cpus-per-task=8` is free rather than wasteful; requesting more
memory than roughly 62 GB per GPU, however, silently raises the bill. Each
account also carries a monthly quota, and exceeding it does not block jobs — it
drops their queue priority until the month turns over.

Monitor with:

```bash
squeue --me
scontrol show job JOBID
sacct -Bj JOBID          # what the job was actually charged
```

## 5. CPU and GPU scope

The collision is a restricted N-body model: each tracer feels the two moving
galaxy potentials, so its cost is O(N), not the O(N²) all-pairs benchmark in
the reference MUrB repository. On NumPy, disjoint tracer chunks execute in
parallel using the cores granted to the process. On CuPy, all tracer state
stays on the A100 through the leapfrog substeps and returns to the host once per
frame for Pillow rendering.

This code is single-process and single-GPU. Requesting four GPUs would waste
three of them. Multi-GPU scaling should be added only for independent
encounters/parameter sweeps or a real distributed solver, with one rank per
GPU and explicit result collation.

## 6. Transfer and playback fallback

Synchronise completed frames from the presentation computer with the transfer
host specified by the event organisers:

```bash
rsync -av USER@TRANSFER_HOST:/path/to/run/ ./runs/leonardo_live/
```

Generate at least one good run of each chosen demonstration before the event
and keep copies on the event PC and removable storage. If a live job queues or
the venue connection fails, label the replay honestly as a run computed
earlier on Leonardo.

## Official references

- [CINECA Leonardo hardware, filesystems, partitions, and current known issues](https://docs.hpc.cineca.it/hpc/leonardo.html)
- [CINECA scheduler and job-submission guide](https://docs.hpc.cineca.it/hpc/hpc_scheduler.html)
- [CINECA account registration and HPC access request](https://docs.hpc.cineca.it/general/users_account.html)
- [CINECA two-factor authentication and SSH certificate login](https://docs.hpc.cineca.it/general/access.html)
- [CINECA file systems, quotas, `chprj` and `cindata`](https://docs.hpc.cineca.it/hpc/hpc_data_storage.html)
- [CINECA FAQ, including the login-node host-key procedure](https://docs.hpc.cineca.it/faq.html)
- [CINECA Singularity and Leonardo CUDA/container guidance](https://docs.hpc.cineca.it/services/singularity.html)
- [CINECA Python `cineca-ai` environment guidance](https://docs.hpc.cineca.it/hpc/hpc_cineca-ai-hpyc.html)
