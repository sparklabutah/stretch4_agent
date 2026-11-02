"""Pure NumPy/Pillow kinematic stand-in. No MuJoCo, no hardware, no robot libraries.

A small room (four walls, checkered floor, two boxes) is ray-cast from camera poses derived
from the joint state, so images change when the robot moves and the gripper depth image is
exactly consistent with the gripper RGB image (depth is Euclidean distance along the ray,
NaN beyond 4 m). A wall stops base motion, so `stopped_short` is reachable.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from ..schema import (COMPLETED, STOPPED_SHORT, TIMEOUT, REJECTED, Frame, FrameTimeout, JointResult,
                      Readiness)

LIMITS = {"lift": (0.0, 1.1), "arm": (0.0, 0.52), "wrist_yaw": (-90.0, 180.0),
          "wrist_pitch": (-90.0, 20.0), "wrist_roll": (-90.0, 90.0), "gripper": (-100.0, 100.0)}
ROOM = 3.0            # walls at +-3 m
BASE_RADIUS = 0.25
MAX_DEPTH = 4.0
BOXES = [  # (min xyz, max xyz, rgb)
    (np.array([1.6, -0.3, 0.0]), np.array([2.0, 0.1, 0.4]), (200, 40, 40)),
    (np.array([1.2, 0.6, 0.0]), np.array([1.5, 0.9, 0.25]), (40, 170, 70)),
]
WALL_COLORS = {"x+": (120, 130, 160), "x-": (160, 130, 120), "y+": (130, 160, 130), "y-": (160, 160, 110)}


def _rot_about(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1 - math.cos(ang)))


class MockBackend:
    name = "mock"
    cameras = {
        "head_left": "head camera, left of a stereo pair, looking forward and down",
        "head_right": "head camera, right of a stereo pair",
        "head_center": "head camera, centre, wide",
        "gripper": "camera on the wrist looking along the gripper; depth is aligned to this image",
    }

    def __init__(self, latency_s: float = 0.0, size_scale: float = 1.0, homed: bool = True):
        self.latency_s = latency_s
        self.sizes = {c: (max(8, int(w * size_scale)), max(8, int(h * size_scale)))
                      for c, (w, h) in {"head_left": (640, 400), "head_right": (640, 400),
                                        "head_center": (640, 400), "gripper": (640, 480)}.items()}
        self.homed = homed
        self.runstopped = False
        self.frozen_camera_time: float | None = None   # tests: make frames look old
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self.s = {"lift": 0.5, "arm": 0.1, "wrist_yaw": 0.0, "wrist_pitch": -45.0, "wrist_roll": 0.0,
                  "gripper": 0.0, "base_x": 0.0, "base_y": 0.0, "base_theta": 0.0}
        self.halt_reasons: list[str] = []

    # ---- lifecycle / state
    def start(self) -> None: ...
    def close(self) -> None: ...

    def get_state(self) -> dict[str, float]:
        with self._lock:
            return dict(self.s)

    def get_limits(self) -> dict[str, tuple[float, float]]:
        return dict(LIMITS)

    def readiness(self) -> Readiness:
        return Readiness(homed=self.homed, runstopped=self.runstopped)

    def doctor(self) -> dict:
        return {"state": self.get_state(), "limits": LIMITS, "camera_sizes": self.sizes,
                "readiness": self.readiness().reasons()}

    def clear_runstop(self) -> None:   # test helper; the agent has no way to call this
        self.runstopped = False
        self._halt.clear()

    # ---- motion
    def halt(self, reason: str) -> None:
        self.halt_reasons.append(reason)
        self.runstopped = True
        self._halt.set()

    def execute(self, deltas: dict[str, float], timeout: float) -> dict[str, JointResult]:
        if self.runstopped or not self.homed:
            return {k: JointResult(REJECTED, v, 0.0, note="runstopped or not homed") for k, v in deltas.items()}
        self._halt.clear()
        start = self.get_state()
        # Fraction of the move that finishes before `timeout` (a slow motion is halted at timeout).
        frac_time = 1.0 if self.latency_s <= 0 else min(1.0, timeout / self.latency_s)
        steps = 10
        done = 0.0
        for i in range(1, steps + 1):
            if self._halt.is_set() or i / steps > frac_time + 1e-9:
                break
            if self.latency_s > 0:
                time.sleep(self.latency_s / steps)
            done = i / steps
        timed_out = frac_time < 1.0 and not self._halt.is_set()
        halted = self._halt.is_set()
        if self.latency_s <= 0 and not halted:
            done = 1.0
        results: dict[str, JointResult] = {}
        with self._lock:
            for k, v in deltas.items():
                results[k] = self._apply(k, v, done, start, timed_out, halted)
        return results

    def _apply(self, key: str, delta: float, done: float, start: dict, timed_out: bool, halted: bool) -> JointResult:
        s = self.s
        want = delta * done
        if key in ("drive", "slide"):
            th = math.radians(start["base_theta"])
            fwd = np.array([math.cos(th), math.sin(th)])
            left = np.array([-math.sin(th), math.cos(th)])
            vec = (fwd if key == "drive" else left) * want
            pos = np.array([s["base_x"], s["base_y"]]) + vec
            lim = ROOM - BASE_RADIUS
            clipped = np.clip(pos, -lim, lim)
            s["base_x"], s["base_y"] = float(clipped[0]), float(clipped[1])
            achieved = float(np.dot(np.array([s["base_x"], s["base_y"]]) - np.array([start["base_x"], start["base_y"]]),
                                    fwd if key == "drive" else left))
        elif key == "turn":
            s["base_theta"] = (s["base_theta"] + want + 180.0) % 360.0 - 180.0
            achieved = want
        else:
            lo, hi = LIMITS[key]
            s[key] = float(min(max(s[key] + want, lo), hi))
            achieved = s[key] - start[key]
        tol = {"drive": 0.005, "slide": 0.005, "turn": 0.5, "lift": 0.002, "arm": 0.002}.get(key, 0.5)
        if halted:
            return JointResult(STOPPED_SHORT, delta, achieved, note="halted by emergency stop")
        if timed_out:
            return JointResult(TIMEOUT, delta, achieved, note="motion timed out and was halted")
        if abs(achieved - delta) <= tol:
            return JointResult(COMPLETED, delta, achieved)
        return JointResult(STOPPED_SHORT, delta, achieved, in_collision_stop=key in ("drive", "slide"),
                           note="blocked by the wall" if key in ("drive", "slide") else None)

    # ---- cameras
    def get_frames(self, after_time: float | None, timeout: float,
                   cameras: list[str] | None = None) -> dict[str, Frame]:
        ts = self.frozen_camera_time if self.frozen_camera_time is not None else time.time()
        if after_time is not None and ts <= after_time:
            raise FrameTimeout(f"newest frame is {after_time - ts:.2f} s older than the last motion end")
        s = self.get_state()
        out = {}
        for name in cameras or list(self.cameras):
            rgb, depth = self._render(name, s)
            out[name] = Frame(rgb, ts, depth if name == "gripper" else None)
        return out

    def _camera_pose(self, name: str, s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """origin, forward, up, horizontal fov (deg)."""
        th = math.radians(s["base_theta"])
        fwd = np.array([math.cos(th), math.sin(th), 0.0])
        left = np.array([-math.sin(th), math.cos(th), 0.0])
        z = np.array([0.0, 0.0, 1.0])
        base = np.array([s["base_x"], s["base_y"], 0.0])
        if name.startswith("head"):
            off = {"head_left": 0.06, "head_right": -0.06, "head_center": 0.0}[name]
            pitch = math.radians(-25.0)
            f = math.cos(pitch) * fwd + math.sin(pitch) * z
            return base + left * off + z * 1.25 + fwd * 0.05, f, z, 100.0 if name == "head_center" else 90.0
        origin = base + fwd * (0.2 + s["arm"]) + z * (0.25 + s["lift"])
        f = _rot_about(fwd, z, math.radians(s["wrist_yaw"]))
        horiz_axis = np.cross(f, z)   # pitch axis
        f = _rot_about(f, horiz_axis, math.radians(s["wrist_pitch"]))  # positive pitch tilts up
        up = _rot_about(z if abs(f[2]) < 0.99 else fwd, f, math.radians(s["wrist_roll"]))
        return origin, f, up, 80.0

    def _render(self, name: str, s: dict) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.sizes[name]
        o, f, up, fov = self._camera_pose(name, s)
        f = f / np.linalg.norm(f)
        right = np.cross(f, up)
        right /= np.linalg.norm(right)
        down = np.cross(f, right)
        fl = (w / 2) / math.tan(math.radians(fov) / 2)
        u = (np.arange(w) + 0.5 - w / 2) / fl
        v = (np.arange(h) + 0.5 - h / 2) / fl
        uu, vv = np.meshgrid(u, v)
        d = f[None, None] + uu[..., None] * right + vv[..., None] * down
        d = d.reshape(-1, 3)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        t, col = self._cast(o, d)
        rgb = col.reshape(h, w, 3)
        depth = np.where(t < MAX_DEPTH, t, np.nan).astype(np.float32).reshape(h, w)
        return rgb, depth

    def _cast(self, o: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(d)
        best = np.full(n, np.inf)
        col = np.empty((n, 3), np.uint8)
        col[:] = (200, 215, 235)   # background

        def hit(t, mask_ok, colors):
            nonlocal best
            better = mask_ok & (t > 1e-4) & (t < best)
            best = np.where(better, t, best)
            col[better] = colors[better]

        with np.errstate(divide="ignore", invalid="ignore"):
            # floor, checkered
            t = -o[2] / d[:, 2]
            p = o + t[:, None] * d
            chk = ((np.floor(p[:, 0] / 0.25) + np.floor(p[:, 1] / 0.25)) % 2).astype(bool)
            fcol = np.where(chk[:, None], np.array([95, 95, 100], np.uint8), np.array([150, 150, 155], np.uint8))
            hit(t, (d[:, 2] < 0) & (np.abs(p[:, 0]) <= ROOM) & (np.abs(p[:, 1]) <= ROOM), fcol)
            # walls
            for ax, sign, key in ((0, 1, "x+"), (0, -1, "x-"), (1, 1, "y+"), (1, -1, "y-")):
                t = (sign * ROOM - o[ax]) / d[:, ax]
                p = o + t[:, None] * d
                other = 1 - ax
                ok = (p[:, 2] >= 0) & (p[:, 2] <= 2.5) & (np.abs(p[:, other]) <= ROOM)
                stripe = (np.floor(p[:, other] / 0.5) % 2).astype(bool)
                base = np.array(WALL_COLORS[key], np.int16)
                wcol = np.where(stripe[:, None], base, base - 25).astype(np.uint8)
                hit(t, ok, wcol)
            # boxes (slab test)
            for lo, hi, rgb in BOXES:
                t1 = (lo - o) / d
                t2 = (hi - o) / d
                tmin = np.minimum(t1, t2).max(axis=1)
                tmax = np.maximum(t1, t2).min(axis=1)
                ok = (tmax >= tmin) & (tmax > 0)
                tt = np.where(tmin > 0, tmin, tmax)
                p = o + tt[:, None] * d
                top = np.abs(p[:, 2] - hi[2]) < 1e-3
                shade = np.where(top[:, None], 1.0, 0.75)
                bcol = (np.array(rgb, float) * shade).astype(np.uint8)
                hit(tt, ok, bcol)
        return best, col
