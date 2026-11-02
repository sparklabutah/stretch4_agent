"""Vocabulary shared by the server, safety layer and backends.

Everything inside the server uses the agent's units: metres, degrees and gripper
percent. Backends convert to radians at the hardware boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Command keys, in the order they are reported. Value units in UNITS.
JOINTS = ("drive", "slide", "turn", "lift", "arm",
          "wrist_yaw", "wrist_pitch", "wrist_roll", "gripper")
UNITS = {"drive": "m", "slide": "m", "turn": "deg", "lift": "m", "arm": "m",
         "wrist_yaw": "deg", "wrist_pitch": "deg", "wrist_roll": "deg", "gripper": "pct"}
# Command key -> the tool parameter the agent sees (used in error messages).
PARAM = {j: f"{j}_{u}" for j, u in UNITS.items()}
BASE_KEYS = ("drive", "slide", "turn")
# Command key -> absolute state key, for the joints that have an absolute position.
STATE_KEY = {j: j for j in ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "gripper")}

CAMERAS = ("head_left", "head_right", "head_center", "gripper")

COMPLETED, STOPPED_SHORT, TIMEOUT, REJECTED = "completed", "stopped_short", "timeout", "rejected"


@dataclass
class JointResult:
    status: str
    requested: float
    achieved: float | None = None
    in_collision_stop: bool | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        d = {"status": self.status, "requested": round(self.requested, 4),
             "achieved": None if self.achieved is None else round(self.achieved, 4)}
        if self.in_collision_stop is not None:
            d["in_collision_stop"] = self.in_collision_stop
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class Frame:
    rgb: np.ndarray                    # HxWx3 uint8, RGB
    timestamp: float                   # time.time() epoch seconds
    depth: np.ndarray | None = None    # HxW float metres, aligned to rgb (gripper only)


@dataclass
class Readiness:
    homed: bool
    runstopped: bool
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.homed and not self.runstopped and not self.problems

    def reasons(self) -> list[str]:
        r = list(self.problems)
        if self.runstopped:
            r.append("robot is runstopped; a human must clear it")
        if not self.homed:
            r.append("robot is not homed; a human must run stretch_robot_home")
        return r


class FrameTimeout(RuntimeError):
    """No frame newer than the requested time arrived in time."""


class MotionRefused(Exception):
    """Raised by safety checks or the service; the message is shown to the agent."""
