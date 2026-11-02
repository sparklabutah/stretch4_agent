"""MuJoCo simulator backend (hello-robot/stretch4_mujoco).

Headless by default. On macOS a viewer needs the child process to be started by `mjpython`
(`--viewer`); headless rendering works with plain python. The simulator uses multiprocessing
'spawn', so entry points need an `if __name__ == "__main__"` guard (the CLI has one).

Differences from the real robot, all approximations: gripper percent is a linear rescale of
the simulated aperture (upper limit = 100), the head cameras are the simulator's fisheye
views, and the stretch backend's lease/runstop behaviour is absent (a halt just freezes the
joints and latches until restart).
"""
from __future__ import annotations

import math
import threading
import time
import numpy as np

from ..schema import (COMPLETED, REJECTED, STOPPED_SHORT, TIMEOUT, Frame, FrameTimeout, JointResult,
                      Readiness)

TOL = {"drive": 0.02, "slide": 0.02, "turn": 2.0, "lift": 0.005, "arm": 0.005,
       "wrist_yaw": 2.0, "wrist_pitch": 2.0, "wrist_roll": 2.0, "gripper": 3.0}
# The simulator's `gripper` actuator (aperture, radians) is missing from pull_joint_limits();
# this is config.robot_settings_se4["gripper_min_max"].
GRIPPER_RAD = (-0.376, 0.56)
ARM_JOINTS = ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "gripper")


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class SimBackend:
    name = "sim"
    cameras = {
        "head_left": "simulated head camera, left",
        "head_right": "simulated head camera, right",
        "head_center": "simulated head camera, centre (low-resolution variant)",
        "gripper": "simulated gripper camera (right imager); depth is aligned to this image",
    }

    def __init__(self, headless: bool = True, camera_hz: float = 30.0, poll_s: float = 0.05):
        self.headless, self.camera_hz, self.poll_s = headless, camera_hz, poll_s
        self.sim = None
        self.cams = None
        self._halt = threading.Event()
        self.runstopped = False
        self._motion_end_sim = 0.0
        self._gr_hi = 1.0

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        from stretch4_mujoco.enums.stretch_cameras import StretchCameras as C
        from stretch4_mujoco.stretch4_mujoco_simulator import Stretch4MujocoSimulator
        self.cams = {"head_left": C.cam_nav_rgb_se4_left, "head_right": C.cam_nav_rgb_se4_right,
                     "head_center": C.cam_nav_rgb_se4_center, "gripper": C.cam_gripper_se4_right_rgb,
                     "_depth": C.cam_gripper_se4_stereo_depth}
        self.sim = Stretch4MujocoSimulator(camera_hz=self.camera_hz, cameras_to_use=list(self.cams.values()))
        self.sim.start(headless=self.headless)     # returns after the first status and camera frames, and homes
        if not self.sim.is_running():
            raise RuntimeError("the MuJoCo process exited during start")
        self._gr_hi = GRIPPER_RAD[1]

    def close(self) -> None:
        if self.sim is not None:
            self.sim.stop()

    # -------------------------------------------------------------------- state
    def _g2pct(self, rad: float) -> float:
        return rad / self._gr_hi * 100.0

    def _pct2g(self, pct: float) -> float:
        return pct / 100.0 * self._gr_hi

    def get_state(self) -> dict[str, float]:
        s = self.sim.pull_status()
        return {"lift": s.lift.pos, "arm": s.arm.pos,
                "wrist_yaw": math.degrees(s.wrist_yaw.pos), "wrist_pitch": math.degrees(s.wrist_pitch.pos),
                "wrist_roll": math.degrees(s.wrist_roll.pos), "gripper": self._g2pct(s.gripper.pos),
                "base_x": s.base.x, "base_y": s.base.y, "base_theta": math.degrees(s.base.theta)}

    def _raw_limits(self) -> dict[str, tuple[float, float]]:
        return {a.name: (float(lo), float(hi)) for a, (lo, hi) in self.sim.pull_joint_limits().items()}

    def get_limits(self) -> dict[str, tuple[float, float]]:
        raw = {**self._raw_limits(), "gripper": GRIPPER_RAD}
        out = {}
        for j in ARM_JOINTS:
            if j not in raw:
                continue
            lo, hi = raw[j]
            if j.startswith("wrist"):
                lo, hi = math.degrees(lo), math.degrees(hi)
            elif j == "gripper":
                lo, hi = self._g2pct(lo), self._g2pct(hi)
            out[j] = (lo, hi)
        return out

    def readiness(self) -> Readiness:
        problems = [] if self.sim.is_running() else ["the MuJoCo process is not running"]
        return Readiness(homed=True, runstopped=self.runstopped, problems=problems)   # sim.start() homes

    def doctor(self) -> dict:
        fr = self.get_frames(None, 10.0)
        return {"state": self.get_state(), "limits": self.get_limits(),
                "cameras": {k: {"rgb_shape": list(f.rgb.shape),
                               **({"depth_shape": list(f.depth.shape)} if f.depth is not None else {})}
                           for k, f in fr.items()}}

    # ------------------------------------------------------------------- motion
    def halt(self, reason: str) -> None:
        self.runstopped = True
        self._halt.set()
        self._freeze()

    def _freeze(self) -> None:
        try:   # command every joint to where it is now and stop the base
            s = self.get_state()
            self.sim.base.set_velocity(0.0, 0.0, 0.0)
            self.sim.lift.move_to(s["lift"])
            self.sim.arm.move_to(s["arm"])
            e = self.sim.end_of_arm
            e.wrist_yaw.move_to(math.radians(s["wrist_yaw"]))
            e.wrist_pitch.move_to(math.radians(s["wrist_pitch"]))
            e.wrist_roll.move_to(math.radians(s["wrist_roll"]))
            e.stretch_gripper.move_to(self._pct2g(s["gripper"]))
        except Exception:
            pass

    def _moving(self) -> bool:
        s = self.sim.pull_status()
        b = s.base
        return bool(s.actuators_in_motion or b.active_translate_x or b.active_translate_y or b.active_rotate)

    def execute(self, deltas: dict[str, float], timeout: float) -> dict[str, JointResult]:
        if self.runstopped:
            return {k: JointResult(REJECTED, v, 0.0, note="emergency stop is latched; restart the server") for k, v in deltas.items()}
        self._halt.clear()
        deadline = time.monotonic() + timeout
        stage_a = {k: v for k, v in deltas.items() if k != "turn"}
        stage_b = {k: v for k, v in deltas.items() if k == "turn"}   # same two-stage rule as the real backend
        results: dict[str, JointResult] = {}
        for stage in (stage_a, stage_b):
            if not stage:
                continue
            if results and any(r.status != COMPLETED for r in results.values()):
                results.update({k: JointResult(REJECTED, v, 0.0, note="not attempted: the earlier part of this move did not complete")
                                for k, v in stage.items()})
                continue
            results.update(self._run_stage(stage, deadline))
        self._motion_end_sim = self.sim.pull_status().time
        return results

    def _run_stage(self, stage: dict[str, float], deadline: float) -> dict[str, JointResult]:
        before = self.get_state()
        sim = self.sim
        if "drive" in stage or "slide" in stage:
            sim.base.translate_by(stage.get("drive", 0.0), stage.get("slide", 0.0))
        if "turn" in stage:
            sim.base.rotate_by(math.radians(stage["turn"]))
        if "lift" in stage:
            sim.lift.move_by(stage["lift"])
        if "arm" in stage:
            sim.arm.move_by(stage["arm"])
        for j in ("wrist_yaw", "wrist_pitch", "wrist_roll"):
            if j in stage:
                getattr(sim.end_of_arm, j).move_by(math.radians(stage[j]))
        if "gripper" in stage:
            sim.end_of_arm.stretch_gripper.move_by(self._pct2g(stage["gripper"]))
        t0, started, idle, outcome = time.monotonic(), False, 0, "done"
        while True:
            if self._halt.is_set():
                outcome = "halted"
                break
            if time.monotonic() > deadline:
                outcome = "timeout"
                break
            time.sleep(self.poll_s)
            if self._moving():
                started, idle = True, 0
            else:
                idle += 1
                if (started and idle >= 2) or (not started and time.monotonic() - t0 > 0.7):
                    break
        if outcome == "timeout":
            self._freeze()
        after = self.get_state()
        res = {}
        for k, want in stage.items():
            if k in ("drive", "slide"):
                th = math.radians(before["base_theta"])
                dx, dy = after["base_x"] - before["base_x"], after["base_y"] - before["base_y"]
                got = dx * math.cos(th) + dy * math.sin(th) if k == "drive" else -dx * math.sin(th) + dy * math.cos(th)
            elif k == "turn":
                got = _wrap(after["base_theta"] - before["base_theta"])
            else:
                got = after[k] - before[k]
            if abs(got - want) <= TOL[k]:
                res[k] = JointResult(COMPLETED, want, got)
            elif outcome == "timeout":
                res[k] = JointResult(TIMEOUT, want, got, note="motion exceeded the timeout and was halted")
            elif outcome == "halted":
                res[k] = JointResult(STOPPED_SHORT, want, got, note="halted by emergency stop")
            else:
                res[k] = JointResult(STOPPED_SHORT, want, got, self.sim.pull_status().is_self_colliding or None,
                                     "stopped short: contact, a joint limit, or the simulator's guard")
        return res

    # ------------------------------------------------------------------ cameras
    def get_frames(self, after_time: float | None, timeout: float,
                   cameras: list[str] | None = None) -> dict[str, Frame]:
        names = cameras or list(self.cameras)
        min_sim = 0.0 if after_time is None else self._motion_end_sim   # simulation clock, not wall clock
        t_end = time.monotonic() + timeout
        while True:
            data = self.sim.pull_camera_data()
            want = [self.cams[n] for n in names] + ([self.cams["_depth"]] if "gripper" in names else [])
            if all(self._has(data, c) and data.get_camera_time(c) > min_sim for c in want):
                break
            if time.monotonic() > t_end:
                raise FrameTimeout("no camera frame rendered after the last motion within %.1f s" % timeout)
            time.sleep(0.02)
        now = time.time()
        out = {}
        for n in names:
            rgb = np.ascontiguousarray(data.get_camera_data(self.cams[n], auto_rotate=True, auto_correct_rgb=False))
            depth = None
            if n == "gripper":
                d = np.array(data.get_camera_data(self.cams["_depth"]), dtype=np.float32)
                if d.shape != rgb.shape[:2]:   # nearest-neighbour onto the RGB grid
                    ys = (np.arange(rgb.shape[0]) * d.shape[0] / rgb.shape[0]).astype(int)
                    xs = (np.arange(rgb.shape[1]) * d.shape[1] / rgb.shape[1]).astype(int)
                    d = d[ys][:, xs]
                d[~np.isfinite(d) | (d <= 0)] = np.nan
                depth = d
            out[n] = Frame(rgb, now, depth)
        return out

    @staticmethod
    def _has(data, cam) -> bool:
        try:
            data.get_camera_data(cam)
            return True
        except ValueError:
            return False
