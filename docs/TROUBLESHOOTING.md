# Troubleshooting

## `ModuleNotFoundError`

Activate your venv and install `requirements.txt`.

## CuPy fails to import

Run with NumPy first:

```bash
export LEONARDO_DEMO_BACKEND=numpy
```

Then install a CuPy build matching the CUDA runtime of the actual machine.

## Neural wall says `numpy-fallback`

PyTorch is not installed/usable. The rest of the viewer still works. Install a CUDA-compatible PyTorch build before presenting it as real neural-network training.

## Viewer shows a broken image while a run is live

The poller may have requested a frame during the tiny interval while it was still being saved. The next poll normally fixes it. For maximum robustness, save to a temporary filename and rename atomically; this is a good production hardening task.

## A Leonardo job sits in the queue

Use `scontrol show job JOBID` to inspect the pending reason. Do not make the exhibition dependent on immediate scheduling; keep cached playback runs.

## `$WORK` is `/no/project/defined`, or `saldo` says "username not existing"

These are one problem, not two: the username exists but is associated with no
project in Leonardo's accounting database. `chprj -l` lists nothing, `$FAST` is
empty, and `sacctmgr show associations user=$USER` returns a header with no
rows. `cindata` still shows `$HOME`, `$PUBLIC` and `$SCRATCH`, because those are
personal areas that exist from the moment the account is created.

Login, `git clone` and compilation all work in this state; `sbatch` and `srun`
do not. Only the project PI can associate you, so the fix is an email, not a
command. A newly created account often resolves within a day — `cindata`'s
`FRESH` column tells you how recently your areas were provisioned.

`chprj -d <account>` only switches between projects you already belong to, so it
cannot help here.

## Host key verification failed on `login.leonardo.cineca.it`

`login.leonardo.cineca.it` is round-robin DNS over four login nodes, each with
its own host key, so a warning usually means you reached a node you had not used
before rather than an interception. It is still a warning worth honouring.

Verify before trusting anything. A fingerprint that matches an entry already in
your `known_hosts` from an earlier session is the same host you have been using.
Otherwise read the keys from inside the cluster over your existing session:

```bash
for n in 01 02 05 07; do ssh-keyscan -t rsa,ecdsa login${n}-ext.leonardo.cineca.it 2>/dev/null; done | ssh-keygen -lf -
```

CINECA's own remedy stores all four nodes under a wildcard:

```bash
ssh-keygen -f ~/.ssh/known_hosts -R login.leonardo.cineca.it
for KEYAL in ssh-rsa ecdsa-sha2-nistp256; do
  for n in 1 2 5 7; do
    ssh-keyscan -t $KEYAL login0${n}-ext.leonardo.cineca.it | sed s"/0${n}-ext/\*/" >> ~/.ssh/known_hosts
  done
done
```

If a fingerprint matches nothing you can corroborate, stop and mail
`superc@cineca.it` instead of deleting the entry.

## `step` is not recognised, but it is installed

On Windows, check the length of the user `PATH`:

```powershell
[Environment]::GetEnvironmentVariable('PATH','User').Length
```

`cmd.exe` truncates `PATH` at 2047 characters, so entries past that point exist
in the registry and are invisible to every shell. Installers append to the end,
which is exactly where the cutoff falls. Removing entries already present in the
machine `PATH` — Windows searches that first, so duplicates never win a lookup —
usually reclaims more than enough.

Two related traps. Explorer caches the environment block, so terminals launched
from the Start menu inherit the old `PATH` until Explorer restarts or you sign
out. And `[Environment]::SetEnvironmentVariable(...,'User')` rewrites the value
as `REG_SZ`, silently breaking any `%VAR%` entry that relied on `REG_EXPAND_SZ`;
use `Set-ItemProperty -Type ExpandString` instead.

## `step ssh login` succeeds but `ssh` still refuses the certificate

The certificate lives in `ssh-agent`, which ships **disabled** on Windows. Check
with `Get-Service ssh-agent`; enable it from an elevated shell:

```powershell
Set-Service -Name ssh-agent -StartupType Automatic; Start-Service ssh-agent
```

Where admin rights are unavailable, `scripts/leonardo_login.ps1 -CertOnly`
writes the certificate to a file for use with `ssh -i`. Certificates expire
after 12 hours either way; the agent saves the flag, not the daily re-auth.

## Leonardo "accepts" the key, then says `Permission denied`

`ssh -v` shows `Server accepts key: ... ECDSA-CERT` and then
`Permission denied (publickey,gssapi-keyex,gssapi-with-mic)`. The certificate
is fine; the private key behind it has a passphrase and nothing could type it
(batch mode, the dashboard, a script). Check with
`ssh-keygen -y -P "" -f $HOME\.ssh\cineca_leonardo` — "incorrect passphrase"
confirms it. Reissue without one:

```powershell
.\scripts\leonardo_login.ps1 -Email abelik@tcd.ie -User abelik00 -CertOnly -NoPassword
```

or keep the passphrase and `ssh-add $HOME\.ssh\cineca_leonardo` once per
certificate. The dashboard's **Refresh certificate** button passes
`-NoPassword` itself. See docs/LEONARDO.md §1.

## JSON or `.env` written on Windows is rejected on Leonardo

Windows PowerShell 5.1 writes a UTF-8 **byte-order mark** with `Out-File
-Encoding utf8` and `Set-Content`. Go and many Linux parsers reject it —
smallstep reports `invalid character 'ï' looking for beginning of value`. Write
without a BOM:

```powershell
[System.IO.File]::WriteAllText($path, $body, (New-Object System.Text.UTF8Encoding($false)))
```

## The local fluid demo is slow

Lower `fluid.nx`, `fluid.ny`, or `steps_per_frame` in `config/profiles.json`. The purpose of the local profile is visual development, not final resolution.


## Every demo fails immediately, with a KeyError in the traceback

Almost always a **stale viewer**. The server imports the demo modules once at
startup but re-reads `config/profiles.json` on every request, so a viewer left
running from before an update executes old code against new settings and every
run dies on a missing key. Tracebacks from a stale process are a giveaway: the
line numbers point at comments or docstrings, because Python captured them
against the old file and prints source from the new one.

Close every viewer window and start one fresh. `app.py` now moves to the next
free port if 8000 is taken and prints a warning, rather than failing to bind
and leaving the browser pointed at the old server.

## The viewer starts but PyTorch is missing

Launch it with `scripts/start_viewer_windows.bat`, which uses `.venv`. Running
`python app.py` against whatever interpreter is on PATH picks up a different
environment without the installed packages.
