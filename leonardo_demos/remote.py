"""Run a dashboard simulation on Discoverer or Leonardo and bring it home.

One click in either front end becomes:

1. **sync**    - the solver source (and, once, the data files) is streamed to
                 the cluster checkout as a tarball, only when it has changed;
2. **submit**  - the validated run request is written to ``job.json`` in a new
                 remote run directory and submitted with ``sbatch``;
3. **wait**    - ``squeue``/``sacct`` and the remote ``meta.json`` are polled,
                 so the page can show queue state and frame progress;
4. **fetch**   - the finished run directory is streamed home as a compressed
                 tar into ``runs/<run id>``, where it is an ordinary saved run.

Everything talks to the cluster through the system ``ssh`` in batch mode, the
same way the showcase scripts do, so it uses whatever key, certificate or
agent already works in a terminal. Progress lives in the local run's
``meta.json`` (``status: "remote"`` plus a ``remote`` block), which both front
ends already poll; a viewer restart resumes watching any job still in flight.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import tarfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = ROOT / "config" / "clusters.json"
PLAN = ROOT / "config" / "hpc_plan.json"          # per-demo device and resources
LOCAL = ROOT / "config" / "clusters.local.json"   # dashboard edits; git-ignored
POLL_SECONDS = 15
SSH_TIMEOUT = 60
EDITABLE = {"host", "port", "user", "identity", "root", "runs_root", "python", "account", "qos",
            "partition", "walltime", "cert_email", "default_profile", "cpu_account", "cpu_partition", "cpu_qos"}
PLACEHOLDER_ACCOUNT = "YOUR_ACTIVE_PROJECT_ACCOUNT"
WALLTIME = re.compile(r"^(\d{1,2}-)?\d{1,2}:\d{2}:\d{2}$")
# What a node needs to run any demo. The dashboard itself (app.py, web/) never
# runs there, and saved runs obviously do not travel this way.
CODE_GLOBS = ["run_demo.py", "leonardo_demos/**/*.py", "config/demo_specs.json", "config/profiles.json",
              "tools/*.py", "requirements*.txt"]
DATA_GLOBS = ["data/**/*"]
ACTIVE_STAGES = {"preparing", "syncing", "submitting", "queued", "running", "fetching"}

_jobs: dict[str, threading.Thread] = {}
_jobs_lock = threading.Lock()
_sync_lock = threading.Lock()


# ------------------------------------------------------------ configuration --
def load_clusters() -> dict:
    clusters = json.loads(DEFAULTS.read_text(encoding="utf-8"))
    try:
        local = json.loads(LOCAL.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        local = {}
    for name, overrides in local.items():
        if name in clusters and isinstance(overrides, dict):
            clusters[name].update({k: v for k, v in overrides.items() if k in EDITABLE})
    return clusters


def cluster(name: str) -> dict:
    clusters = load_clusters()
    if name not in clusters:
        raise KeyError(name)
    c = dict(clusters[name])
    c["name"] = name
    return c


def save_overrides(name: str, values: dict) -> dict:
    clusters = load_clusters()
    if name not in clusters:
        raise KeyError(name)
    clean = {}
    for key, value in values.items():
        if key not in EDITABLE:
            raise ValueError(f"{key} cannot be edited here")
        if key == "port":
            value = int(value)
            if not 1 <= value <= 65535:
                raise ValueError("port must be between 1 and 65535")
        else:
            value = str(value).strip()
            if any(ch in value for ch in "\n\r'\"`;|&<>"):
                raise ValueError(f"{key} contains characters that are not allowed")
        if key == "walltime" and not WALLTIME.match(value):
            raise ValueError("walltime must look like HH:MM:SS or D-HH:MM:SS")
        clean[key] = value
    try:
        local = json.loads(LOCAL.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        local = {}
    local.setdefault(name, {}).update(clean)
    LOCAL.write_text(json.dumps(local, indent=2) + "\n", encoding="utf-8")
    return cluster(name)


def public_view(c: dict) -> dict:
    """The configuration as the settings panel shows it."""
    view = {k: c.get(k) for k in sorted(EDITABLE)}
    view.update(label=c.get("label", c["name"]), name=c["name"],
                resources_note=c.get("resources_note", ""),
                account_missing=not c.get("account") or c.get("account") == PLACEHOLDER_ACCOUNT)
    return view


# --------------------------------------------------------------------- ssh --
def _identity(c: dict) -> str | None:
    identity = (c.get("identity") or "").strip()
    return str(Path(os.path.expanduser(identity))) if identity else None


def ssh_program(c: dict | None = None) -> str:
    """Windows' own OpenSSH for clusters marked ``"ssh_agent": true``: it talks
    to the Windows ssh-agent, which holds the passphrase-protected CINECA key
    and certificate after scripts/leonardo_login.ps1 (Git Bash's ssh cannot
    reach that agent). Other clusters keep the default ssh: offering every
    agent certificate first gets Discoverer to disconnect."""
    if os.name == "nt" and (c or {}).get("ssh_agent"):
        native = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"
        if native.exists():
            return str(native)
    return "ssh"


def ssh_base(c: dict) -> list[str]:
    cmd = [ssh_program(c), "-p", str(c.get("port") or 22), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
           "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]
    identity = _identity(c)
    if identity:
        cmd += ["-i", identity, "-o", "IdentitiesOnly=yes"]
    return cmd + [f"{c['user']}@{c['host']}"]


def remote_command(c: dict, script: str) -> str:
    """Wrap a shell snippet so it runs in a login shell with the site setup."""
    setup = (c.get("login_setup") or "").strip()
    return "bash -lc " + shlex.quote(f"{setup}\n{script}" if setup else script)


class RemoteError(RuntimeError):
    pass


def explain_ssh_failure(c: dict, stderr: str) -> str:
    text = stderr.strip()
    low = text.lower()
    label = c.get("label", c["name"])
    if "host identification has changed" in low or "host key verification failed" in low:
        return (f"{label}: the login node's host key is not trusted on this PC yet. Verify it and add it to "
                f"known_hosts (docs/TROUBLESHOOTING.md, 'Host key verification failed'), then try again.")
    if "permission denied" in low or "certificate" in low and "expired" in low:
        if c["name"] == "leonardo":
            return ("Leonardo refused the login: the CINECA SSH certificate is missing or has expired "
                    "(they last 12 hours). Use 'Refresh certificate' in HPC settings, or run "
                    "scripts\\leonardo_login.ps1 -CertOnly, then try again.")
        return f"{label} refused the login (permission denied). Check the user name and SSH key."
    if "timed out" in low or "could not resolve" in low or "network is unreachable" in low:
        return f"{label} could not be reached from this PC ({text.splitlines()[-1] if text else 'timeout'})."
    return f"{label}: {text.splitlines()[-1] if text else 'ssh failed'}"


def run_remote(c: dict, script: str, *, stdin: bytes | None = None, timeout: int = SSH_TIMEOUT) -> str:
    try:
        proc = subprocess.run(ssh_base(c) + [remote_command(c, script)], input=stdin,
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RemoteError(f"{c.get('label')}: no answer within {timeout} s")
    except FileNotFoundError:
        raise RemoteError("The ssh program was not found on this PC (install the OpenSSH client).")
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")
        # 255 is ssh itself failing; anything else is the remote command.
        if proc.returncode == 255:
            raise RemoteError(explain_ssh_failure(c, err))
        out = proc.stdout.decode("utf-8", "replace")
        raise RemoteError(f"{c.get('label')}: remote command failed ({proc.returncode}): "
                          f"{(err or out).strip()[-600:]}")
    return proc.stdout.decode("utf-8", "replace")


def q(path: str) -> str:
    """Double-quote a remote path so $WORK/$FAST still expand but spaces do not split."""
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'


# ------------------------------------------------------ certificate status --
def certificate_status(c: dict) -> dict | None:
    """Validity of an SSH certificate beside the configured key, if any."""
    identity = _identity(c)
    if not identity:
        return None
    cert = Path(identity + "-cert.pub")
    if not cert.exists():
        return {"present": False, "valid": False, "detail": f"No certificate at {cert}"}
    try:
        out = subprocess.run(["ssh-keygen", "-L", "-f", str(cert)], capture_output=True, text=True,
                             timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {"present": True, "valid": None, "detail": "ssh-keygen could not read the certificate"}
    match = re.search(r"Valid:\s+from\s+(\S+)\s+to\s+(\S+)", out)
    if not match:
        return {"present": True, "valid": None, "detail": "Certificate validity not found"}
    try:
        until = time.mktime(time.strptime(match.group(2), "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return {"present": True, "valid": None, "detail": match.group(0)}
    left = until - time.time()
    return {"present": True, "valid": left > 0, "expires": until, "seconds_left": int(left),
            "detail": (f"valid for {int(left // 3600)} h {int(left % 3600 // 60)} min" if left > 0
                       else "expired")}


def open_certificate_login(c: dict) -> None:
    """Open a console running the CINECA certificate script for the presenter."""
    if os.name != "nt":
        raise RemoteError("The certificate helper is a Windows PowerShell script.")
    email = (c.get("cert_email") or "").strip()
    if not email:
        raise RemoteError("Set the e-mail registered with CINECA UserDB in HPC settings first.")
    identity = _identity(c) or str(Path.home() / ".ssh" / "cineca_leonardo")
    script = ROOT / "scripts" / "leonardo_login.ps1"
    subprocess.Popen(["powershell", "-NoExit", "-ExecutionPolicy", "Bypass", "-File", str(script),
                      "-Email", email, "-User", c["user"], "-KeyPath", identity, "-CertOnly"],
                     creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0), cwd=str(ROOT))


# ------------------------------------------------------------- code sync --
def _files(globs: list[str]) -> list[Path]:
    out = set()
    for pattern in globs:
        for path in ROOT.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                out.add(path)
    return sorted(out)


def _bundle(globs: list[str]) -> tuple[str, list[Path]]:
    files = _files(globs)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16], files


def _tar(files: list[Path]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(ROOT).as_posix())
    return buf.getvalue()


def sync_code(c: dict, say=lambda message: None) -> None:
    """Bring the cluster checkout up to date with this PC, only if it differs."""
    root = c["root"]
    with _sync_lock:
        for kind, globs, timeout in (("code", CODE_GLOBS, 300), ("data", DATA_GLOBS, 1800)):
            digest, files = _bundle(globs)
            if not files:
                continue
            marker = f"{root}/.dashboard_{kind}"
            current = run_remote(c, f"cat {q(marker)} 2>/dev/null || true").strip()
            if current == digest:
                continue
            say(f"Uploading {kind} to {c['label']} ({len(files)} files)…")
            run_remote(c, f"mkdir -p {q(root)} && tar -xzf - -C {q(root)} && printf %s {digest} > {q(marker)}",
                       stdin=_tar(files), timeout=timeout)


# ----------------------------------------------------------------- checks --
def check(c: dict) -> dict:
    """Everything a presenter wants to know before pressing Run."""
    result = {"cluster": c["name"], "label": c.get("label"), "ok": False, "items": [],
              "certificate": certificate_status(c)}
    items = result["items"]
    cert = result["certificate"]
    if cert is not None:
        items.append({"name": "SSH certificate", "ok": bool(cert.get("valid")), "detail": cert["detail"]})
    if not c.get("account") or c.get("account") == PLACEHOLDER_ACCOUNT:
        items.append({"name": "Slurm account", "ok": False,
                      "detail": "Not set. On the cluster, 'saldo -b' or 'sacctmgr show associations user=$USER' "
                                "prints it; enter it in HPC settings."})
    python = c.get("python", "")
    script = "\n".join([
        "echo HOST=$(hostname)",
        f"echo ROOT={q(c['root'])}",
        f"echo RUNS={q(c['runs_root'])}",
        f"test -f {q(c['root'] + '/run_demo.py')} && echo ROOT_OK=1",
        f"test -x {q(python)} && echo PY_OK=1",
        f"mkdir -p {q(c['runs_root'])} 2>/dev/null && test -w {q(c['runs_root'])} && echo RUNS_OK=1",
        "command -v sbatch >/dev/null && echo SBATCH_OK=1",
        f"echo CODE=$(cat {q(c['root'] + '/.dashboard_code')} 2>/dev/null)",
        "echo QUEUE=$(squeue -h -u $USER 2>/dev/null | wc -l)",
    ])
    try:
        started = time.time()
        out = run_remote(c, script)
        latency = time.time() - started
    except RemoteError as error:
        items.insert(0, {"name": "Connection", "ok": False, "detail": str(error)})
        return result
    facts = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    digest, _ = _bundle(CODE_GLOBS)
    items.insert(0, {"name": "Connection", "ok": True,
                     "detail": f"{facts.get('HOST', '?')} answered in {latency:.1f} s"})
    items.append({"name": "Slurm", "ok": "SBATCH_OK" in facts,
                  "detail": f"sbatch available · {facts.get('QUEUE', '0')} of your jobs in the queue"
                  if "SBATCH_OK" in facts else "sbatch not found after the login setup"})
    items.append({"name": "Checkout", "ok": True,
                  "detail": (f"{facts.get('ROOT')} · " + ("up to date with this PC" if facts.get("CODE") == digest
                             else "will be updated before the run" if "ROOT_OK" in facts
                             else "missing: the source will be uploaded before the first run"))})
    items.append({"name": "Python environment", "ok": "PY_OK" in facts,
                  "detail": python if "PY_OK" in facts else
                  f"{python} not found. Create it once (docs/DISCOVERER.md or docs/LEONARDO.md)."})
    items.append({"name": "Run folder", "ok": "RUNS_OK" in facts, "detail": facts.get("RUNS", "")})
    result["ok"] = all(item["ok"] for item in items)
    result["remote_root"] = facts.get("ROOT")
    result["remote_runs"] = facts.get("RUNS")
    return result


# ------------------------------------------------------------- job script --
def python_for(c: dict, demo: str, method: str) -> str:
    overrides = c.get("python_overrides") or {}
    return overrides.get(f"{demo}/{method}") or overrides.get(demo) or c["python"]


def load_plan() -> dict:
    try:
        return json.loads(PLAN.read_text(encoding="utf-8")).get("demos", {})
    except (OSError, ValueError):
        return {}


def demo_needs(demo: str, method: str | None = None) -> dict:
    """Device and size one run of this demo needs (config/hpc_plan.json).

    A method entry (e.g. the molecular shuttle, which is CPU-only) overrides the
    demo's defaults. Unknown demos get one GPU and eight cores.
    """
    entry = dict(load_plan().get(demo) or {})
    methods = entry.pop("methods", {}) or {}
    if method and method in methods:
        entry.update(methods[method])
    need = {"device": "gpu", "gpus": 1, "cpus": 8, "mem_gb": 64, "backend": "auto"}
    need.update({k: entry[k] for k in ("device", "gpus", "cpus", "mem_gb", "backend", "why") if k in entry})
    if need["device"] == "cpu":
        need["gpus"] = 0
    return need


def resources(c: dict, walltime: str | None = None, demo: str | None = None, method: str | None = None) -> dict:
    """The Slurm request for one run: only the devices the demo actually uses.

    Every earlier job asked Discoverer for a whole node (4 GPUs, 128 cores) and
    used one GPU: billing counts every allocated GPU and core, busy or not.
    """
    need = demo_needs(demo or "", method)
    node = c.get("gpu_node" if need["device"] == "gpu" else "cpu_node") or c.get("gpu_node") or {}
    gpus = min(int(need["gpus"]), int(node.get("gpus", need["gpus"]) or 0)) if need["device"] == "gpu" else 0
    cpus = min(int(need["cpus"]), int(node.get("cpus", need["cpus"])))
    mem = min(int(need["mem_gb"]), int(node.get("mem_gb", need["mem_gb"])))
    if need["device"] == "cpu":
        account = c.get("cpu_account") or c.get("account")
        partition = c.get("cpu_partition", c.get("partition"))
        qos = c.get("cpu_qos", c.get("qos"))
    else:
        account, partition, qos = c.get("account"), c.get("partition"), c.get("qos")
    flags = ["--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus}", f"--mem={mem}G"]
    if gpus:
        flags.append(f"--gres=gpu:{gpus}")
    flags += list(c.get("sbatch_extra") or [])
    weights = c.get("billing") or {}
    billing = None
    if weights:
        billing = round(gpus * float(weights.get("gpu", 0)) + cpus * float(weights.get("cpu", 0)), 2)
    return {"device": need["device"], "gpus": gpus, "cpus": cpus, "mem_gb": mem, "backend": need["backend"],
            "why": need.get("why", ""), "account": account, "qos": qos, "partition": partition,
            "walltime": walltime or c.get("walltime"), "sbatch": flags,
            "srun": ["--ntasks=1", f"--cpus-per-task={cpus}"] + list(c.get("srun_extra") or []),
            "billing_per_hour": billing,
            "note": (f"{gpus} GPU{'s' if gpus != 1 else ''} + {cpus} cores" if gpus else f"CPU only: {cpus} cores")
                    + f", {mem} GB" + (f" (billing about {billing}/h)" if billing is not None else "")}


def account_missing(c: dict, res: dict) -> bool:
    return not res.get("account") or res.get("account") == PLACEHOLDER_ACCOUNT


def job_script(c: dict, run_dir: str, demo: str, method: str, walltime: str) -> str:
    res = resources(c, walltime, demo, method)
    lines = ["#!/bin/bash", f"#SBATCH --job-name=lvd-{demo[:24]}",
             f"#SBATCH --output={run_dir}/slurm.log", f"#SBATCH --error={run_dir}/slurm.log",
             f"#SBATCH --time={walltime}", f"#SBATCH --account={res['account']}"]
    if res.get("qos"):
        lines.append(f"#SBATCH --qos={res['qos']}")
    if res.get("partition"):
        lines.append(f"#SBATCH --partition={res['partition']}")
    lines += [f"#SBATCH {flag}" for flag in res["sbatch"]]
    lines += ["", "# Written by the Leonardo Visual Demos dashboard (leonardo_demos/remote.py).",
              f"# Device plan: {res['note']}. {res.get('why', '')}".rstrip()]
    lines += list(c.get("job_setup") or [])
    # Frame workers and BLAS threads follow the cores actually allocated.
    lines += ["export LEONARDO_DEMO_CPU_WORKERS=${SLURM_CPUS_PER_TASK:-1} OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
              "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}"]
    lines += ["set -eo pipefail", "export PYTHONUNBUFFERED=1", f"cd {q(c['root'])}",
              "echo \"[$(date +%T)] $(hostname) - job $SLURM_JOB_ID\"",
              "command -v nvidia-smi >/dev/null && nvidia-smi -L || true",
              " ".join(["srun", *res["srun"], q(python_for(c, demo, method)),
                        "tools/run_job.py", q(run_dir + "/job.json")]),
              "echo \"[$(date +%T)] done\""]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------- job lifecycle --
def _read_meta(rd: Path) -> dict:
    try:
        return json.loads((rd / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_meta(rd: Path, meta: dict) -> None:
    tmp = rd / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(rd / "meta.json")


def _update(rd: Path, **remote) -> dict:
    meta = _read_meta(rd)
    block = dict(meta.get("remote") or {})
    block.update(remote, updated=time.time())
    meta["remote"] = block
    _write_meta(rd, meta)
    return meta


def _fail(rd: Path, message: str, **remote) -> None:
    meta = _read_meta(rd)
    meta["remote"] = {**(meta.get("remote") or {}), **remote, "stage": "failed", "updated": time.time()}
    meta.update(status="failed", error=message)
    _write_meta(rd, meta)


def code_version() -> dict:
    """The commit a cluster run was launched from, and whether solver code was edited since."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""
    dirty = git("status", "--porcelain", "--", "run_demo.py", "leonardo_demos", "tools", "config/demo_specs.json",
                "config/profiles.json")
    return {"commit": git("rev-parse", "--short", "HEAD") or None, "solver_code_modified": bool(dirty)}


def create_job(runs: Path, rid: str, c: dict, kwargs: dict, walltime: str, extra_files: dict[str, Path]) -> Path:
    """Write the local placeholder run and its job.json, then start the watcher."""
    rd = runs / rid
    rd.mkdir(parents=True, exist_ok=True)
    job = {k: v for k, v in kwargs.items() if k != "run_dir"}
    job["timings"] = True
    (rd / "job.json").write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
    meta = {"status": "remote", "demo": kwargs["demo"], "profile": kwargs["profile"], "frames": kwargs["frames"],
            "params": {k: v for k, v in kwargs["params"].items() if not k.startswith("_")},
            "method": kwargs["method"], "backend": kwargs["backend"], "precision": kwargs["precision"],
            "created": time.time(), "frame": -1,
            "remote": {"cluster": c["name"], "label": c.get("label"), "stage": "preparing", "walltime": walltime,
                       "code": code_version(),
                       "message": f"Preparing the {c.get('label')} job…", "files": sorted(extra_files)}}
    _write_meta(rd, meta)
    start_watcher(runs, rid)
    return rd


def start_watcher(runs: Path, rid: str) -> None:
    with _jobs_lock:
        thread = _jobs.get(rid)
        if thread and thread.is_alive():
            return
        thread = threading.Thread(target=_lifecycle, args=(runs, rid), name=f"remote-{rid}", daemon=True)
        _jobs[rid] = thread
        thread.start()


def resume_jobs(runs: Path) -> list[str]:
    """Pick up remote runs that were in flight when the viewer last stopped."""
    resumed = []
    for meta_path in runs.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if meta.get("status") == "remote":
            start_watcher(runs, meta_path.parent.name)
            resumed.append(meta_path.parent.name)
    return resumed


def active_jobs(runs: Path) -> list[dict]:
    out = []
    for meta_path in sorted(runs.glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        remote = meta.get("remote")
        if not remote:
            continue
        if meta.get("status") != "remote" and time.time() - float(remote.get("updated") or 0) > 6 * 3600:
            continue
        out.append({"id": meta_path.parent.name, "demo": meta.get("demo"), "status": meta.get("status"),
                    "error": meta.get("error"), **remote})
        if len(out) >= 20:
            break
    return out


def cancel(runs: Path, rid: str) -> None:
    rd = runs / rid
    meta = _read_meta(rd)
    remote = meta.get("remote") or {}
    if meta.get("status") != "remote":
        raise RemoteError("That run is not waiting on a cluster.")
    c = cluster(remote["cluster"])
    if remote.get("job_id"):
        run_remote(c, f"scancel {int(remote['job_id'])}")
    _update(rd, cancel_requested=True, message="Cancelling…")


def _lifecycle(runs: Path, rid: str) -> None:
    rd = runs / rid
    try:
        meta = _read_meta(rd)
        remote = meta.get("remote") or {}
        c = cluster(remote["cluster"])
        if not remote.get("job_id"):
            if remote.get("stage") not in {"preparing", "syncing", "submitting"}:
                return
            if remote.get("stage") != "preparing":
                _fail(rd, "The viewer stopped before this job was submitted. Start it again.")
                return
            _submit(c, rd, rid, remote)
        _watch(c, rd, rid)
    except Exception as error:  # the page must always learn how it ended
        _fail(rd, str(error))


def _submit(c: dict, rd: Path, rid: str, remote: dict) -> None:
    job = json.loads((rd / "job.json").read_text(encoding="utf-8"))
    if account_missing(c, resources(c, None, job["demo"], job.get("method"))):
        raise RemoteError(f"No Slurm account is set for {c['label']} (this demo's partition). Enter it in HPC settings.")
    _update(rd, stage="syncing", message=f"Checking the {c['label']} checkout…")
    sync_code(c, say=lambda message: _update(rd, message=message))
    _update(rd, stage="submitting", message=f"Submitting to {c['label']}…")
    runs_root = run_remote(c, f"mkdir -p {q(c['runs_root'])} && cd {q(c['runs_root'])} && pwd").strip().splitlines()[-1]
    remote_dir = f"{runs_root}/{rid}"
    job = json.loads((rd / "job.json").read_text(encoding="utf-8"))
    script = job_script(c, remote_dir, job["demo"], job.get("method", "default"),
                        remote.get("walltime") or c["walltime"])
    (rd / "job.sbatch").write_text(script, encoding="utf-8", newline="\n")
    files = [rd / "job.json", rd / "job.sbatch"] + [rd / name for name in remote.get("files") or []]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in files:
            tar.add(path, arcname=path.name)
    out = run_remote(c, f"mkdir -p {q(remote_dir)} && tar -xzf - -C {q(remote_dir)} && "
                        f"cd {q(remote_dir)} && sbatch --parsable job.sbatch", stdin=buf.getvalue())
    match = re.search(r"^(\d+)", out.strip().splitlines()[-1] if out.strip() else "")
    if not match:
        raise RemoteError(f"sbatch did not return a job id: {out.strip()[-300:]}")
    _update(rd, stage="queued", job_id=int(match.group(1)), remote_dir=remote_dir, submitted=time.time(),
            message=f"Job {match.group(1)} is waiting in the {c['label']} queue")


def _watch(c: dict, rd: Path, rid: str) -> None:
    unknown_polls = 0
    failures = 0
    while True:
        remote = _read_meta(rd).get("remote") or {}
        job_id, remote_dir = int(remote["job_id"]), remote["remote_dir"]
        script = "\n".join([
            f"echo SQ=$(squeue -h -j {job_id} -o %T 2>/dev/null)",
            f"echo SA=$(sacct -n -X -j {job_id} -o State%30 2>/dev/null | head -1)",
            f"echo REASON=$(squeue -h -j {job_id} -o %r 2>/dev/null)",
            # printf, not echo: an unquoted JSON document would be globbed.
            f"printf 'META=%s\\n' \"$(tr -d '\\n' < {q(remote_dir + '/meta.json')} 2>/dev/null)\"",
        ])
        try:
            out = run_remote(c, script)
            failures = 0
        except RemoteError as error:
            failures += 1
            _update(rd, message=f"Lost contact ({error}); retrying…")
            if failures >= 40:     # ten minutes of silence
                raise
            time.sleep(POLL_SECONDS)
            continue
        facts = {}
        for line in out.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                facts[key] = value.strip()
        queue_state = facts.get("SQ", "").split()[0] if facts.get("SQ") else ""
        acct_state = facts.get("SA", "").split()[0] if facts.get("SA") else ""
        try:
            remote_meta = json.loads(facts.get("META") or "{}")
        except ValueError:
            remote_meta = {}
        frame = remote_meta.get("frame", -1)
        total = remote_meta.get("frames") or _read_meta(rd).get("frames")
        if queue_state in {"PENDING", "CONFIGURING"}:
            reason = facts.get("REASON", "")
            _update(rd, stage="queued", state=queue_state,
                    message=f"Job {job_id} is waiting in the {c['label']} queue" + (f" ({reason})" if reason and reason != "None" else ""))
        elif queue_state:
            if frame is not None and frame >= 0:
                message = f"Running on {c['label']} · frame {frame + 1} of {total}"
            else:
                message = f"Running on {c['label']} · {remote_meta.get('message') or 'starting up'}"
            _update(rd, stage="running", state=queue_state, frame=frame, total=total,
                    started=remote.get("started") or time.time(), message=message,
                    remote_message=remote_meta.get("message"))
        else:
            finished = acct_state or remote_meta.get("status")
            if not finished or finished in {"PENDING", "RUNNING", "starting", "running"}:
                unknown_polls += 1
                if unknown_polls < 8:
                    time.sleep(POLL_SECONDS)
                    continue
            _finish(c, rd, rid, acct_state, remote_meta)
            return
        time.sleep(POLL_SECONDS)


def _finish(c: dict, rd: Path, rid: str, acct_state: str, remote_meta: dict) -> None:
    remote = _read_meta(rd).get("remote") or {}
    status = remote_meta.get("status")
    if status != "complete":
        log = ""
        try:
            log = run_remote(c, f"tail -n 25 {q(remote['remote_dir'] + '/slurm.log')} 2>/dev/null || true")
        except RemoteError:
            pass
        reason = remote_meta.get("error") or (f"Slurm reports {acct_state}" if acct_state else "the job ended early")
        if remote.get("cancel_requested"):
            reason = "Cancelled from the dashboard"
        elif acct_state == "TIMEOUT":
            reason = f"The job hit its {remote.get('walltime')} time limit. Give it more time or fewer frames."
        _fail(rd, f"{c['label']} run did not complete: {reason}", state=acct_state, log_tail=log[-3000:])
        return
    _update(rd, stage="fetching", state=acct_state, message=f"Bringing the run home from {c['label']}…")
    parent, name = remote["remote_dir"].rsplit("/", 1)
    proc = subprocess.Popen(ssh_base(c) + [remote_command(c, f"tar -czf - -C {q(parent)} {shlex.quote(name)}")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    kept = _read_meta(rd).get("remote") or {}
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as archive:
            archive.extractall(rd.parent, filter="data")
    except (tarfile.TarError, OSError) as error:
        proc.kill()
        raise RemoteError(f"Transfer from {c['label']} failed: {error}")
    finally:
        proc.wait()
    if proc.returncode:
        raise RemoteError(f"Transfer from {c['label']} failed: "
                          f"{proc.stderr.read().decode('utf-8', 'replace').strip()[-300:]}")
    meta = _read_meta(rd)
    if meta.get("status") != "complete":
        raise RemoteError("The fetched run is incomplete.")
    kept.update(stage="done", state=acct_state or "COMPLETED", fetched=time.time(),
                message=f"Computed on {c['label']}", updated=time.time())
    kept.pop("log_tail", None)
    meta["remote"] = kept
    _write_meta(rd, meta)
