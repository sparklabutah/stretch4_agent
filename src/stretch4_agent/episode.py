"""Per-episode state and logging.

runs/<UTC timestamp>_<id>/events.jsonl   one line per tool call
runs/<...>/images/                       every JPEG the agent was shown
runs/<...>/episode.json                  task, outcome (self-claimed), counters

The gap between one tool result and the next tool call is mostly model inference time, so
each event records `model_wait_s` (that gap) and `exec_s` (time inside the tool) separately.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .imaging import Encoded
from .safety import Budget, SafetyConfig


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, **event: Any) -> None:
        event = {"utc": _now_utc(), **event}
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")


@dataclass
class DepthFrame:
    frame_id: int
    depth: np.ndarray        # full-resolution metres
    enc: Encoded             # how the agent's image maps to `depth`


@dataclass
class Episode:
    task: str
    dir: Path
    budget: Budget
    read_only: bool
    backend_name: str
    id: str = field(default_factory=lambda: secrets.token_hex(3))
    active: bool = True
    outcome: str | None = None
    evidence: str | None = None
    fresh_observation: bool = False      # an observation exists newer than the last motion
    last_motion_end: float | None = None  # epoch seconds, for frame freshness
    frame_counter: int = 0
    current_frame_id: int | None = None  # None after a move: the agent's last image is stale
    depth_frame: DepthFrame | None = None
    depth_calls: int = 0                 # since the last move
    counters: dict[str, int] = field(default_factory=lambda: {
        "moves": 0, "moves_rejected": 0, "observations": 0, "depth_queries": 0,
        "emergency_stops": 0, "images_saved": 0})
    model_wait_s: float = 0.0
    exec_s: float = 0.0
    log: EventLog = field(init=False)
    _t0: float = field(default_factory=time.monotonic)
    _utc0: str = field(default_factory=_now_utc)

    @staticmethod
    def open(runs_dir: Path, task: str, cfg: SafetyConfig, backend_name: str,
             clock: Callable[[], float] = time.monotonic) -> "Episode":
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ep_id = secrets.token_hex(3)
        d = runs_dir / f"{ts}_{ep_id}"
        (d / "images").mkdir(parents=True, exist_ok=True)
        ep = Episode(task=task, dir=d, budget=Budget(cfg, clock), read_only=cfg.read_only,
                     backend_name=backend_name, id=ep_id)
        ep.log = EventLog(d / "events.jsonl")
        ep.write_summary()
        return ep

    def next_frame_id(self) -> int:
        self.frame_counter += 1
        return self.frame_counter

    def save_image(self, frame_id: int, camera: str, jpeg: bytes) -> str:
        rel = f"images/{frame_id:04d}_{camera}.jpg"
        (self.dir / rel).write_bytes(jpeg)
        self.counters["images_saved"] += 1
        return rel

    def write_summary(self) -> None:
        summary = {
            "id": self.id, "task": self.task, "backend": self.backend_name,
            "read_only": self.read_only, "started_utc": self._utc0,
            "active": self.active,
            "outcome_self_claimed": self.outcome,
            "evidence_self_claimed": self.evidence,
            "outcome_verified": False,  # nothing here checks the claim against the world
            "duration_s": round(time.monotonic() - self._t0, 2),
            "model_wait_s": round(self.model_wait_s, 2), "exec_s": round(self.exec_s, 2),
            "counters": self.counters, "budget": self.budget.to_dict(),
        }
        (self.dir / "episode.json").write_text(json.dumps(summary, indent=2))
