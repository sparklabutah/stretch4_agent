"""Fake RobotClient and camera generators, shaped from stretch4_body (robot_client.py,
subsystem_client.py, subsystem/cameras/models/image_frame.py).

Mirrored on purpose: commands are queued in `cmd_dict` keyed by subsystem (a second omnibase
command overwrites the first), nothing moves until `push_command()`, `push_command()` calls
sys.exit(1) when the control lock is held elsewhere, and `move_by` returns False for an
unhomed joint. Motion plays out over a few `pull_status()` calls.
"""
import math
import sys
import time
from types import SimpleNamespace

import numpy as np


class _Joint:
    def __init__(self, robot, key):
        self.robot, self.key = robot, key


class FakeRobot:
    def __init__(self, *, homed=True, runstopped=False, tool_joints=("wrist_yaw", "wrist_pitch", "wrist_roll", "stretch_gripper"),
                 ticks_per_move=3, lease_dropped=False, push_exits=False, never_finishes=False, stall=None):
        self.cmd_dict = {}
        self.push_log = []           # snapshot of cmd_dict at every successful push
        self.pushes_attempted = 0
        self.homed, self._runstopped = homed, runstopped
        self.ticks_per_move, self.lease_dropped = ticks_per_move, lease_dropped
        self.push_exits, self.never_finishes = push_exits, never_finishes
        self.stall = stall or {}     # key -> fraction of the move that is achieved before stalling
        self.started = False
        self.stopped = False
        self.warnings = []
        self.robot_params = {"wrist_yaw": {"range_deg": [-90, 180]}, "wrist_pitch": {"range_deg": [-90, 20]},
                             "wrist_roll": {"range_deg": [-90, 90]}}
        self.status = {
            "lift": {"pos": 0.5, "motor": {"pos_calibrated": True, "is_mg_moving": False},
                     "in_collision_stop": {"pos": False, "neg": False},
                     "soft_motion_limits": {"current": [0.0, 1.1]}},
            "arm": {"pos": 0.1, "motor": {"pos_calibrated": True, "is_mg_moving": False},
                    "in_collision_stop": {"pos": False, "neg": False},
                    "soft_motion_limits": {"current": [0.0, 0.52]}},
            "omnibase": {"x": 0.0, "y": 0.0, "theta": 0.0,
                         **{f"wheel_{i}": {"is_mg_moving": False} for i in range(3)}},
            "end_of_arm": {j: {"pos": 0.0, "is_moving": False, "pos_calibrated": homed} for j in tool_joints},
            "power_periph": {"runstop_event": runstopped},
        }
        if "stretch_gripper" in tool_joints:
            self.status["end_of_arm"]["stretch_gripper"]["pos_pct"] = 0.0
        self.end_of_arm = SimpleNamespace(joints=list(tool_joints), move_by=self._eoa_move_by,
                                          is_moving=self._eoa_is_moving, quick_stop=lambda j: self._queue(f"{j}.end_of_arm", "quick_stop", j))
        self.lift = SimpleNamespace(move_by=lambda m: self._pj_move_by("lift", m), is_moving=lambda: self.status["lift"]["motor"]["is_mg_moving"],
                                    set_velocity=lambda v: self._queue("lift", "set_velocity", v))
        self.arm = SimpleNamespace(move_by=lambda m: self._pj_move_by("arm", m), is_moving=lambda: self.status["arm"]["motor"]["is_mg_moving"],
                                   set_velocity=lambda v: self._queue("arm", "set_velocity", v))
        self.omnibase = SimpleNamespace(translate_by=lambda x, y: self._queue("omnibase", "translate_by", x, y),
                                        rotate_by=lambda w: self._queue("omnibase", "rotate_by", w),
                                        hard_stop=lambda: self._queue("omnibase", "hard_stop"))
        self._motion = None   # (ticks_left, plan)

    # --- RobotClient surface -------------------------------------------------------
    def startup(self, **kw):
        self.started = True
        return True

    def stop(self):
        self.stopped = True

    def is_homed(self):
        return self.homed

    def is_runstopped(self):
        return self.status["power_periph"]["runstop_event"]

    def trigger_runstop(self):
        self._queue("power_periph", "trigger_runstop")

    def pull_status(self, blocking=True):
        self._tick()
        return True

    def push_command(self, ignore_control_lock=False, priority=0):
        self.pushes_attempted += 1
        if self.push_exits and not ignore_control_lock:
            sys.exit(1)      # real behaviour; cmd_dict is NOT cleared on this path
        snap = dict(self.cmd_dict)
        self.cmd_dict.clear()
        self.push_log.append(snap)
        for key, cmd in snap.items():
            if key == "power_periph" and cmd[0] == "trigger_runstop":
                self.status["power_periph"]["runstop_event"] = True
                self._motion = None
                self._set_moving(False)
            elif cmd[0] in ("hard_stop", "set_velocity", "quick_stop"):
                self._motion = None
                self._set_moving(False)
        plan = {k: c for k, c in snap.items() if c[0] in ("move_by", "translate_by", "rotate_by")}
        if plan and not self.lease_dropped:
            self._motion = [self.ticks_per_move, plan]
            self._set_moving(True)
        return True

    # --- command queue --------------------------------------------------------------
    def _queue(self, subsystem, command, *args):
        if subsystem in self.cmd_dict:
            self.warnings.append(f"overwriting previous command for {subsystem}")
        self.cmd_dict[subsystem] = (command, *args)

    def _pj_move_by(self, name, m):
        if not self.status[name]["motor"]["pos_calibrated"]:
            return False
        self._queue(name, "move_by", m)
        return True

    def _eoa_move_by(self, joint, x):
        if joint not in self.end_of_arm.joints or not self.status["end_of_arm"][joint]["pos_calibrated"]:
            return False
        self._queue(f"{joint}.end_of_arm", "move_by", joint, x)
        return True

    def _eoa_is_moving(self, names=None):
        return any(self.status["end_of_arm"][j]["is_moving"] for j in (names or self.end_of_arm.joints))

    # --- motion model ---------------------------------------------------------------
    def _set_moving(self, on):
        for n in ("lift", "arm"):
            self.status[n]["motor"]["is_mg_moving"] = on and n in self._moving_keys()
        for j in self.status["end_of_arm"]:
            self.status["end_of_arm"][j]["is_moving"] = on and f"{j}.end_of_arm" in self._moving_keys()
        # only two wheels turn for a pure translation along a wheel axis: the OR in the backend must cope
        for i in range(3):
            self.status["omnibase"][f"wheel_{i}"]["is_mg_moving"] = on and "omnibase" in self._moving_keys() and i != 0

    def _moving_keys(self):
        return self._motion[1].keys() if self._motion else ()

    def _tick(self):
        if not self._motion or self.never_finishes:
            return
        self._motion[0] -= 1
        if self._motion[0] > 0:
            return
        plan = self._motion[1]
        st = self.status
        for key, cmd in plan.items():
            frac = lambda k: self.stall.get(k, 1.0)
            if key in ("lift", "arm"):
                st[key]["pos"] += cmd[1] * frac(key)
                if frac(key) < 1.0:
                    st[key]["in_collision_stop"] = {"pos": cmd[1] > 0, "neg": cmd[1] < 0}
            elif key == "omnibase" and cmd[0] == "translate_by":
                th = st["omnibase"]["theta"]
                x, y = cmd[1] * frac("drive"), cmd[2] * frac("slide")
                st["omnibase"]["x"] += x * math.cos(th) - y * math.sin(th)
                st["omnibase"]["y"] += x * math.sin(th) + y * math.cos(th)
            elif key == "omnibase" and cmd[0] == "rotate_by":
                st["omnibase"]["theta"] += cmd[1] * frac("turn")
            else:
                joint = cmd[1]
                e = st["end_of_arm"][joint]
                if joint == "stretch_gripper":
                    e["pos_pct"] += cmd[2] * frac("gripper")
                    e["pos"] = e["pos_pct"] / 100.0
                else:
                    e["pos"] += cmd[2] * frac(joint)
        self._motion = None
        self._set_moving(False)


# --- camera generators ---------------------------------------------------------------

def _img(h, w, bgr):
    a = np.zeros((h, w, 3), np.uint8)
    a[:] = bgr
    return a


def head_stream(period=0.01, fail_after=None):
    """Yields SyncedImageFrame-shaped objects: .left/.right/.center, each with .image in BGR."""
    def gen():
        n = 0
        while True:
            n += 1
            if fail_after is not None and n > fail_after:
                raise RuntimeError("OAK device disconnected")
            mk = lambda c: SimpleNamespace(image=_img(40, 64, c), timestamp=time.time(), frame_number=n)
            yield SimpleNamespace(timestamp=time.time(), left=mk((255, 0, 0)),   # pure BLUE in BGR
                                  right=mk((0, 255, 0)), center=mk((0, 0, 255)))
            time.sleep(period)
    return gen


def gripper_stream(period=0.01, depth_shape=(24, 32), depth_mm=640):
    """Yields frames with .right.image (BGR) and .depth (uint16 millimetres, aligned to the right image)."""
    def gen():
        while True:
            depth = np.full(depth_shape, depth_mm, np.uint16)
            depth[0, 0] = 0   # no stereo match
            right = SimpleNamespace(image=_img(48, 64, (0, 128, 255)), timestamp=time.time(), frame_number=1)
            yield SimpleNamespace(timestamp=time.time(), left=right, right=right, center=None, depth=depth)
            time.sleep(period)
    return gen
