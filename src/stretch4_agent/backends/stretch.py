"""Real Stretch 4, via RobotClient (local IPC on the NUC) and the camera generators.

NOT exercised against hardware: the tests drive it with a fake RobotClient and fake camera
generators shaped from the stretch4_body source. See the README bring-up checklist.

`hello-robot-stretch4-body` is imported lazily in start(), so other backends and the tests
never need it.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Iterable

import numpy as np

from ..schema import (COMPLETED, REJECTED, STOPPED_SHORT, TIMEOUT, Frame, FrameTimeout, JointResult,
                      Readiness)

WRIST = ("wrist_yaw", "wrist_pitch", "wrist_roll")
# How close achieved must be to requested to call it completed (agent units).
TOL = {"drive": 0.02, "slide": 0.02, "turn": 2.0, "lift": 0.005, "arm": 0.005,
       "wrist_yaw": 2.0, "wrist_pitch": 2.0, "wrist_roll": 2.0, "gripper": 3.0}
# Used only if the gripper's command range cannot be read from the tool metadata.
GRIPPER_FALLBACK = (-100.0, 50.0)
LEASE_NOTE = ("no motion was observed. If the command was valid, another RobotClient may hold the "
              "control lease (ROS driver or gamepad teleop running?): commands from a non-holder are "
              "dropped silently")


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class StretchBackend:
    name = "stretch"
    cameras = {
        "head_left": "head camera, left, rotated upright by the camera pipeline",
        "head_right": "head camera, right",
        "head_center": "head camera, centre, high resolution wide view",
        "gripper": "gripper RGBD camera (right imager); depth is aligned to this image",
    }

    def __init__(self, robot_factory: Callable[[], Any] | None = None,
                 head_stream: Callable[[], Iterable] | None = None,
                 gripper_stream: Callable[[], Iterable] | None = None,
                 poll_s: float = 0.05, start_grace_s: float = 0.6,
                 depth_scale_to_m: float = 0.001, ip_address: str | None = None):
        self._robot_factory, self._head_stream, self._gripper_stream = robot_factory, head_stream, gripper_stream
        self.poll_s, self.start_grace_s = poll_s, start_grace_s
        self.doctor_window_s = 3.0
        self.depth_scale_to_m = depth_scale_to_m  # ASSUMED millimetres; see README "Unverified"
        self.ip_address = ip_address
        self.robot: Any = None
        self._lock = threading.RLock()       # RobotClient is not thread-safe
        self._halt = threading.Event()
        self._stop = threading.Event()
        self._latest: dict[str, tuple[float, Any]] = {}
        self._frame_counts = {"head": 0, "gripper": 0}
        self._cam_err: dict[str, str] = {}
        self._cam_lock = threading.Lock()
        self.joints: list[str] = []
        self.gripper_joint: str | None = None
        self._limits: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._robot_factory is None:
            try:
                from stretch4_body.robot.robot_client import RobotClient   # Linux + hardware libraries
            except ModuleNotFoundError as e:
                raise RuntimeError("the stretch backend needs hello-robot-stretch4-body (Linux, on the robot's NUC): "
                                   'pip install -e ".[robot]"') from e
            ip = self.ip_address
            self._robot_factory = (lambda: RobotClient(ip_address=ip)) if ip else RobotClient
        if self._head_stream is None or self._gripper_stream is None:
            # Camera generators open the DepthAI devices in this process, which is why this
            # backend runs on the NUC and RobotClient alone cannot see. The *_compressed
            # variants decode on the host by default, so frames arrive as ordinary BGR.
            from stretch4_body.subsystem import cameras as cam
            self._head_stream = self._head_stream or cam.stream_left_right_center_camera_compressed
            self._gripper_stream = self._gripper_stream or cam.stream_gripper_camera_compressed
        self.robot = self._robot_factory()
        if not self.robot.startup():           # returns False on failure; the `with` form would yield None
            raise RuntimeError("RobotClient.startup() failed: is `stretch_body_server` running on this machine?")
        self.joints = list(self.robot.end_of_arm.joints)    # depends on the attached tool: never hardcode
        self.gripper_joint = next((j for j in self.joints if j not in WRIST), None)
        self._pull()
        self._limits = self._discover_limits()
        for key, factory in (("head", self._head_stream), ("gripper", self._gripper_stream)):
            threading.Thread(target=self._camera_loop, args=(key, factory), daemon=True, name=f"cam-{key}").start()

    def close(self) -> None:
        self._stop.set()
        if self.robot is not None:
            with self._lock:
                try:
                    self.robot.stop()    # closes the connection; it is NOT a motion halt
                except (Exception, SystemExit):
                    pass

    def _camera_loop(self, key: str, factory: Callable[[], Iterable]) -> None:
        try:
            for fr in factory():
                if self._stop.is_set():
                    return
                if fr is None:
                    continue
                with self._cam_lock:
                    self._latest[key] = (time.time(), fr)   # our clock, so freshness compares with motion end
                    self._frame_counts[key] += 1
        except Exception as e:
            self._cam_err[key] = f"{type(e).__name__}: {e}"

    # -------------------------------------------------------------------- status
    def _pull(self, timeout: float = 1.0) -> None:
        """Fresh status or an error. Non-blocking pulls in a loop: a dead server must not hang us."""
        t_end = time.monotonic() + timeout
        with self._lock:
            while not self.robot.pull_status(blocking=False):
                if time.monotonic() > t_end:
                    raise RuntimeError("no status from stretch_body_server within %.1f s" % timeout)
                time.sleep(0.005)

    def _status(self) -> dict:
        return self.robot.status

    def _state_locked(self) -> dict[str, float]:
        st = self._status()
        eoa = st["end_of_arm"]
        out = {"lift": st["lift"]["pos"], "arm": st["arm"]["pos"],
               "base_x": st["omnibase"]["x"], "base_y": st["omnibase"]["y"],
               "base_theta": math.degrees(st["omnibase"]["theta"])}
        for j in WRIST:
            if j in eoa:
                out[j] = math.degrees(eoa[j]["pos"])
        if self.gripper_joint and self.gripper_joint in eoa:
            g = eoa[self.gripper_joint]
            out["gripper"] = g.get("pos_pct", g["pos"])   # the gripper reports pct; other tools only pos
        return out

    def get_state(self) -> dict[str, float]:
        with self._lock:
            self._pull()
            return self._state_locked()

    def _discover_limits(self) -> dict[str, tuple[float, float]]:
        st = self._status()
        lim = {}
        for j in ("lift", "arm"):
            lo, hi = st[j]["soft_motion_limits"]["current"]
            lim[j] = (float(lo), float(hi))
        params = getattr(self.robot, "robot_params", {}) or {}
        for j in WRIST:
            if j in self.joints:
                rng = (params.get(j) or {}).get("range_deg")
                if rng:   # a joint without a readable range is left out, so moving it is refused
                    lim[j] = (float(rng[0]), float(rng[1]))
        if self.gripper_joint:
            lim["gripper"] = self._gripper_range()
        return lim

    def _gripper_range(self) -> tuple[float, float]:
        try:
            from stretch4_body.utils.tool_metadata import get_tool_metadata
            lo, hi = get_tool_metadata(self.gripper_joint).command_range
            return float(lo), float(hi)
        except Exception:
            return GRIPPER_FALLBACK

    def get_limits(self) -> dict[str, tuple[float, float]]:
        with self._lock:
            self._pull()
            lim = dict(self._limits)   # lift/arm soft limits can change at runtime: re-read
            for j in ("lift", "arm"):
                a, b = self._status()[j]["soft_motion_limits"]["current"]
                lim[j] = (float(a), float(b))
            return lim

    def readiness(self) -> Readiness:
        problems = [f"camera stream {k} stopped: {e}" for k, e in self._cam_err.items()]
        for key, ok in (("head", "head_left"), ("gripper", "gripper")):
            if key not in self._latest and key not in self._cam_err and not self._frame_counts[key]:
                problems.append(f"no frames yet from the {ok} camera stream")
        try:
            with self._lock:
                self._pull()
                return Readiness(homed=bool(self.robot.is_homed()), runstopped=bool(self.robot.is_runstopped()),
                                 problems=problems)
        except Exception as e:
            return Readiness(homed=False, runstopped=False, problems=problems + [f"status unavailable: {e}"])

    # -------------------------------------------------------------------- motion
    def halt(self, reason: str) -> None:
        self._halt.set()
        got = self._lock.acquire(timeout=1.0)   # best effort: do not deadlock if a poll is stuck
        try:
            self.robot.trigger_runstop()        # only queues a command ...
            self.robot.push_command(ignore_control_lock=True)   # ... so it must be pushed (unverified on hardware)
        except (Exception, SystemExit):
            pass
        finally:
            if got:
                self._lock.release()

    def execute(self, deltas: dict[str, float], timeout: float) -> dict[str, JointResult]:
        self._halt.clear()
        deadline = time.monotonic() + timeout
        # RobotClient keys queued commands by subsystem, so translate and rotate (both 'omnibase')
        # would overwrite each other in one push. Run translate with the joints first, then rotate.
        stage_a = {k: v for k, v in deltas.items() if k != "turn"}
        stage_b = {k: v for k, v in deltas.items() if k == "turn"}
        results: dict[str, JointResult] = {}
        for stage in (stage_a, stage_b):
            if not stage:
                continue
            if results and any(r.status != COMPLETED for r in results.values()):
                results.update({k: JointResult(REJECTED, v, 0.0, note="not attempted: the earlier part of this move did not complete")
                                for k, v in stage.items()})
                continue
            results.update(self._run_stage(stage, deadline))
        return results

    def _queue(self, key: str, val: float) -> bool:
        r, eoa = self.robot, self.robot.end_of_arm
        if key == "lift":
            return r.lift.move_by(val) is not False
        if key == "arm":
            return r.arm.move_by(val) is not False
        if key in WRIST:
            return eoa.move_by(key, math.radians(val)) is not False
        if key == "gripper":
            return eoa.move_by(self.gripper_joint, val) is not False   # passed through as pct (unverified)
        raise KeyError(key)

    def _moving(self, keys: Iterable[str]) -> bool:
        st, r = self._status(), self.robot
        for k in keys:
            if k in ("drive", "slide", "turn"):
                # OmniBaseClient.is_moving ANDs the three wheels, which reads False while the base moves
                # along a wheel's axis (that wheel is still). Use OR.
                if any(st["omnibase"][f"wheel_{i}"]["is_mg_moving"] for i in range(3)):
                    return True
            elif k in ("lift", "arm"):
                if getattr(r, k).is_moving():
                    return True
            else:
                j = k if k in WRIST else self.gripper_joint
                if r.end_of_arm.is_moving([j]):
                    return True
        return False

    def _run_stage(self, stage: dict[str, float], deadline: float) -> dict[str, JointResult]:
        with self._lock:
            self._pull()
            if self.robot.is_runstopped() or not self.robot.is_homed():
                return {k: JointResult(REJECTED, v, 0.0, note="runstopped or not homed") for k, v in stage.items()}
            before = self._state_locked()
            res: dict[str, JointResult] = {}
            queued: list[str] = []
            if "drive" in stage or "slide" in stage:
                self.robot.omnibase.translate_by(stage.get("drive", 0.0), stage.get("slide", 0.0))
                queued += [k for k in ("drive", "slide") if k in stage]
            if "turn" in stage:
                self.robot.omnibase.rotate_by(math.radians(stage["turn"]))
                queued.append("turn")
            for k, v in stage.items():
                if k in ("drive", "slide", "turn"):
                    continue
                if self._queue(k, v):
                    queued.append(k)
                else:   # the client returns False for a joint that is not homed
                    res[k] = JointResult(REJECTED, v, 0.0, note="client refused the command (joint not homed?)")
            if queued:
                try:
                    self.robot.push_command()
                except SystemExit:
                    # push_command calls sys.exit(1) when another process holds the control lock.
                    # The queued commands were NOT sent and stay in cmd_dict: clear them, or the next
                    # push would send stale motion.
                    self.robot.cmd_dict.clear()
                    res.update({k: JointResult(REJECTED, stage[k], 0.0,
                                               note="another process holds the robot control lock") for k in queued})
                    return res
                except Exception:
                    self.robot.cmd_dict.clear()
                    raise
        outcome = self._wait(queued, deadline) if queued else "done"
        with self._lock:
            if outcome == "timeout":
                self._stop_motion(queued)
            self._pull()
            after, st = self._state_locked(), self._status()
            for k in queued:
                res[k] = self._classify(k, stage[k], before, after, st, outcome)
        return res

    def _wait(self, keys: list[str], deadline: float) -> str:
        """Poll until the motion ends. wait_on_motion_finish() cannot be used: it drops the timeout result."""
        t_push = time.monotonic()
        started, idle = False, 0
        while True:
            if self._halt.is_set():
                return "halted"
            if time.monotonic() > deadline:
                return "timeout"
            time.sleep(self.poll_s)
            with self._lock:
                self._pull()
                moving = self._moving(keys)
            if moving:
                started, idle = True, 0
            else:
                idle += 1
                if started and idle >= 2:
                    return "done"
                if not started and time.monotonic() - t_push > self.start_grace_s:
                    return "done"   # never moved: classified from positions (possibly a dropped lease)

    def _stop_motion(self, keys: list[str]) -> None:
        r = self.robot
        try:
            for k in keys:
                if k in ("drive", "slide", "turn"):
                    r.omnibase.hard_stop()
                elif k in ("lift", "arm"):
                    getattr(r, k).set_velocity(0.0)
                else:
                    r.end_of_arm.quick_stop(k if k in WRIST else self.gripper_joint)
            r.push_command()
        except (Exception, SystemExit):
            self.robot.cmd_dict.clear()

    def _classify(self, key: str, want: float, before: dict, after: dict, st: dict, outcome: str) -> JointResult:
        if key in ("drive", "slide"):
            th = math.radians(before["base_theta"])
            dx, dy = after["base_x"] - before["base_x"], after["base_y"] - before["base_y"]
            # Assumes odometry x/y are in the odom frame and +x is robot forward, +y left (unverified).
            got = dx * math.cos(th) + dy * math.sin(th) if key == "drive" else -dx * math.sin(th) + dy * math.cos(th)
        elif key == "turn":
            got = _wrap_deg(after["base_theta"] - before["base_theta"])
        else:
            got = after[key] - before[key]
        if abs(got - want) <= TOL[key]:
            return JointResult(COMPLETED, want, got)
        coll = self._in_collision(key, st)
        if outcome == "timeout":
            return JointResult(TIMEOUT, want, got, coll, "motion exceeded the timeout and was halted")
        if outcome == "halted":
            return JointResult(STOPPED_SHORT, want, got, coll, "halted by emergency stop")
        note = LEASE_NOTE if abs(got) <= TOL[key] / 2 else None
        if key == "gripper" and note is None:
            note = "gripper stopped short: usually a grasp"
        return JointResult(STOPPED_SHORT, want, got, coll, note)

    def _in_collision(self, key: str, st: dict) -> bool | None:
        node = None
        if key in ("lift", "arm"):
            node = st[key].get("in_collision_stop")
        elif key in WRIST:
            node = st["end_of_arm"].get(key, {}).get("in_collision_stop")
        elif key == "gripper" and self.gripper_joint:
            node = st["end_of_arm"].get(self.gripper_joint, {}).get("in_collision_stop")
        if node is None:
            return None
        return bool(node.get("pos") or node.get("neg")) if isinstance(node, dict) else bool(node)

    # ------------------------------------------------------------------- cameras
    def get_frames(self, after_time: float | None, timeout: float,
                   cameras: list[str] | None = None) -> dict[str, Frame]:
        names = cameras or list(self.cameras)
        need = {"head" if n.startswith("head") else "gripper" for n in names}
        t_end = time.monotonic() + timeout
        while True:
            with self._cam_lock:
                snap = {k: self._latest.get(k) for k in need}
            errs = [self._cam_err[k] for k in need if k in self._cam_err]
            if errs and any(snap[k] is None for k in need if k in self._cam_err):
                raise FrameTimeout("camera stream died: " + "; ".join(errs))
            if all(v is not None and (after_time is None or v[0] > after_time) for v in snap.values()):
                break
            if time.monotonic() > t_end:
                raise FrameTimeout("no frame newer than the last motion within %.1f s" % timeout)
            time.sleep(0.02)
        out: dict[str, Frame] = {}
        for n in names:
            ts, fr = snap["head" if n.startswith("head") else "gripper"]
            if n == "gripper":
                right = fr.right
                rgb = self._rgb(right.image)
                out[n] = Frame(rgb, ts, self._depth_m(fr.depth, rgb.shape[:2]))
            else:
                sub = getattr(fr, n.split("_", 1)[1])
                if sub is None:
                    raise FrameTimeout(f"{n} is not present in the head stream frame")
                out[n] = Frame(self._rgb(sub.image), ts)
        return out

    @staticmethod
    def _rgb(bgr: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(bgr)[..., ::-1])   # the pipeline decodes to BGR

    def _depth_m(self, depth: Any, shape: tuple[int, int]) -> np.ndarray | None:
        if depth is None:
            return None
        d = np.asarray(depth, dtype=np.float32) * self.depth_scale_to_m
        if d.shape != shape:   # nearest-neighbour onto the RGB grid; only if the pipeline did not align sizes
            ys = (np.arange(shape[0]) * d.shape[0] / shape[0]).astype(int)
            xs = (np.arange(shape[1]) * d.shape[1] / shape[1]).astype(int)
            d = d[ys][:, xs]
        d[d <= 0] = np.nan   # 0 means no stereo match
        return d

    # -------------------------------------------------------------------- doctor
    def doctor(self) -> dict:
        """Status keys, limits and camera shapes/rates. Sends no commands."""
        out: dict[str, Any] = {}
        with self._lock:
            self._pull()
            st = self._status()
            out["status_keys"] = sorted(st.keys())
            out["end_of_arm_joints"] = self.joints
            out["gripper_joint"] = self.gripper_joint
            out["state"] = self._state_locked()
            out["omnibase_status_keys"] = sorted(st["omnibase"].keys())
            out["is_homed"], out["is_runstopped"] = self.robot.is_homed(), self.robot.is_runstopped()
        out["limits"] = self.get_limits()
        out["limits_missing"] = [k for k in ("wrist_yaw", "wrist_pitch", "wrist_roll", "gripper") if k not in out["limits"]]
        c0, t0 = dict(self._frame_counts), time.monotonic()
        time.sleep(self.doctor_window_s)
        dt = time.monotonic() - t0
        cams: dict[str, Any] = {"errors": dict(self._cam_err)}
        for key in ("head", "gripper"):
            cams[f"{key}_fps"] = round((self._frame_counts[key] - c0[key]) / dt, 1)
        try:
            fr = self.get_frames(None, 2.0)
            for n, f in fr.items():
                cams[n] = {"rgb_shape": list(f.rgb.shape)}
                if f.depth is not None:
                    v = f.depth[np.isfinite(f.depth)]
                    cams[n]["depth"] = {"shape": list(f.depth.shape), "valid_fraction": round(float(v.size / f.depth.size), 3),
                                        "min_m": float(v.min()) if v.size else None, "max_m": float(v.max()) if v.size else None,
                                        "assumed_unit": "millimetres, scaled by %g" % self.depth_scale_to_m}
        except FrameTimeout as e:
            cams["frames"] = f"unavailable: {e}"
        out["cameras"] = cams
        return out
