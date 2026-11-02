"""Server-side safety layer. The agent cannot change any of this; only `serve` flags can."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .schema import JOINTS, MotionRefused, PARAM, STATE_KEY, UNITS

DEFAULT_PER_MOVE = {
    "drive": 0.30, "slide": 0.30, "turn": 45.0, "lift": 0.15, "arm": 0.15,
    "wrist_yaw": 45.0, "wrist_pitch": 45.0, "wrist_roll": 45.0,
    "gripper": 50.0,  # not in DESIGN.md; a gap filled so every key has a cap
}
LIMIT_TOL = 1e-3  # absorbs encoder noise when a joint sits on its limit


@dataclass
class SafetyConfig:
    per_move: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PER_MOVE))
    max_moves: int = 150
    max_seconds: float = 30 * 60
    move_timeout: float = 15.0
    read_only: bool = False

    def __post_init__(self) -> None:
        """Reject a configuration that would silently weaken a limit (a NaN cap compares False
        with everything, so it would disable the cap)."""
        def ok(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
        problems = [f"per-move cap for {j} is missing" for j in JOINTS if j not in self.per_move]
        problems += [f"unknown per-move key {k!r}" for k in self.per_move if k not in JOINTS]
        problems += [f"per-move cap for {k} must be a finite number > 0, got {v!r}"
                     for k, v in self.per_move.items() if k in JOINTS and not ok(v)]
        problems += [f"{name} must be a finite number > 0, got {v!r}" for name, v in
                     (("max_moves", self.max_moves), ("max_seconds", self.max_seconds),
                      ("move_timeout", self.move_timeout)) if not ok(v)]
        if problems:
            raise ValueError("invalid safety configuration: " + "; ".join(problems))


def validate_command(cmd: Mapping[str, Any], state: Mapping[str, float],
                     limits: Mapping[str, tuple[float, float]], cfg: SafetyConfig) -> dict[str, float]:
    """Return the non-zero deltas, or raise MotionRefused listing every problem.
    One bad key rejects the whole command; nothing is partially accepted."""
    errors: list[str] = []
    deltas: dict[str, float] = {}
    for key, val in cmd.items():
        if key not in JOINTS:
            errors.append(f"unknown key {key!r}; valid keys: {', '.join(JOINTS)}")
            continue
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errors.append(f"{PARAM[key]}: must be a number, got {type(val).__name__}")
            continue
        val = float(val)
        if not math.isfinite(val):
            errors.append(f"{PARAM[key]}: must be finite, got {val}")
            continue
        cap = cfg.per_move[key]
        if abs(val) > cap:
            errors.append(f"{PARAM[key]}: |{val:g}| exceeds the per-move limit {cap:g} {UNITS[key]}")
            continue
        if key in STATE_KEY and val != 0.0:
            if key not in limits or key not in state:
                errors.append(f"{PARAM[key]}: this robot has no readable limits for {key} (tool missing or limits unknown)")
                continue
            lo, hi = limits[key]
            target = state[key] + val
            if not (lo - LIMIT_TOL <= target <= hi + LIMIT_TOL):
                errors.append(f"{PARAM[key]}: target {target:.3f} {UNITS[key]} is outside the joint limits "
                              f"[{lo:.3f}, {hi:.3f}] (currently {state[key]:.3f})")
                continue
        if val != 0.0:
            deltas[key] = val
    if not errors and not deltas:
        errors.append("every delta is zero; nothing to move (use observe to look)")
    if errors:
        raise MotionRefused("; ".join(errors))
    return deltas


class Budget:
    """Move and wall-clock budget for one episode."""

    def __init__(self, cfg: SafetyConfig, clock: Callable[[], float] = time.monotonic):
        self.cfg, self.clock = cfg, clock
        self.started = clock()
        self.moves = 0

    def elapsed(self) -> float:
        return self.clock() - self.started

    def check(self) -> None:
        if self.moves >= self.cfg.max_moves:
            raise MotionRefused(f"move budget exhausted ({self.cfg.max_moves}); call finish_episode")
        if self.elapsed() >= self.cfg.max_seconds:
            raise MotionRefused(f"time budget exhausted ({self.cfg.max_seconds:.0f} s); call finish_episode")

    def to_dict(self) -> dict:
        return {"moves_used": self.moves, "moves_max": self.cfg.max_moves,
                "seconds_used": round(self.elapsed(), 1), "seconds_max": self.cfg.max_seconds}
