"""Backend contract. Backends are synchronous; the server runs them in threads.

`halt` must be safe to call from another thread while `execute` is running, and
must make that `execute` return promptly.
"""
from __future__ import annotations

from typing import Protocol

from ..schema import Frame, JointResult, Readiness


class Backend(Protocol):
    name: str
    # camera name -> one-line description shown to the agent
    cameras: dict[str, str]

    def start(self) -> None: ...
    def close(self) -> None: ...

    def get_state(self) -> dict[str, float]:
        """Absolute joints in agent units (lift, arm, wrist_*, gripper) plus base odometry
        base_x, base_y (m) and base_theta (deg)."""
        ...

    def get_limits(self) -> dict[str, tuple[float, float]]:
        """(min, max) for each absolute joint in get_state()."""
        ...

    def execute(self, deltas: dict[str, float], timeout: float) -> dict[str, JointResult]:
        """Run a relative move (keys from schema.JOINTS, only non-zero ones) and block until it
        finishes, stalls, is halted, or `timeout` seconds pass. Returns one result per key."""
        ...

    def halt(self, reason: str) -> None: ...

    def get_frames(self, after_time: float | None, timeout: float,
                   cameras: list[str] | None = None) -> dict[str, Frame]:
        """Latest frame per camera, each newer than `after_time` (epoch seconds) or raise
        FrameTimeout after `timeout`."""
        ...

    def readiness(self) -> Readiness: ...

    def doctor(self) -> dict:
        """Read-only diagnostics for `stretch4-agent doctor`. Must not move anything."""
        ...
