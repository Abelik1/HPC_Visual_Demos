"""Which demos each demo day shows.

The project is presented twice, once for Discoverer and once for Leonardo, with
a different set of demos each time. ``config/lineups.json`` holds both sets,
the archive, and the non-simulation "extras" (recorded videos such as the
raytracer) and says which machine is active. Both front ends read it, and the
dashboard edits it, so the stand never needs a code change to swap a demo.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINEUPS = ROOT / "config" / "lineups.json"
LOCK = threading.Lock()
EXTRA_ID = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
FOLDER = re.compile(r"^[A-Za-z0-9_\- ]{1,60}$")


def load() -> dict:
    try:
        data = json.loads(LINEUPS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.setdefault("active", "all")
    data.setdefault("machines", {})
    data.setdefault("archived", [])
    data.setdefault("extras", {})
    return data


def validate(data: dict, demo_ids: set[str]) -> dict:
    """Return a clean copy of a lineup document, or raise ValueError."""
    extras = {}
    for key, extra in (data.get("extras") or {}).items():
        if not EXTRA_ID.match(key) or key in demo_ids:
            raise ValueError(f"invalid extra id {key!r}")
        if not isinstance(extra, dict):
            raise ValueError(f"extra {key} must be an object")
        folder = str(extra.get("folder") or key)
        if not FOLDER.match(folder):
            raise ValueError(f"extra {key}: folder may only use letters, digits, spaces, - and _")
        url = str(extra.get("url") or "")
        if url and not url.startswith(("http://", "https://")):
            raise ValueError(f"extra {key}: link must start with http:// or https://")
        extras[key] = {
            "name": str(extra.get("name") or key)[:80],
            "tagline": str(extra.get("tagline") or "")[:200],
            "category": str(extra.get("category") or "Physics")[:40],
            "kind": "video",
            "folder": folder,
            "url": url,
        }
    known = demo_ids | set(extras)
    machines = {}
    for name, machine in (data.get("machines") or {}).items():
        if not EXTRA_ID.match(name):
            raise ValueError(f"invalid machine id {name!r}")
        demos = [d for d in dict.fromkeys(machine.get("demos") or []) if d in known]
        machines[name] = {"label": str(machine.get("label") or name.title())[:40], "demos": demos}
    if not machines:
        raise ValueError("at least one machine is required")
    archived = [d for d in dict.fromkeys(data.get("archived") or []) if d in known]
    active = data.get("active") or "all"
    if active != "all" and active not in machines:
        raise ValueError(f"unknown machine {active!r}")
    return {"active": active, "machines": machines, "archived": archived, "extras": extras}


def save(data: dict) -> None:
    with LOCK:
        tmp = LINEUPS.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(LINEUPS)
