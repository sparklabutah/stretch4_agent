"""StretchBackend against a FAKE RobotClient and fake camera generators.
Nothing here touches hardware; it proves the backend's logic, not the robot's behaviour."""
import json
import math
import time

import numpy as np
import pytest

from conftest import make_service
from fakes import FakeRobot, gripper_stream, head_stream
from stretch4_agent.backends.stretch import StretchBackend
from stretch4_agent.schema import FrameTimeout, MotionRefused


def make(robot=None, **kw):
    robot = robot or FakeRobot()
    be = StretchBackend(robot_factory=lambda: robot, head_stream=head_stream(), gripper_stream=gripper_stream(),
                        poll_s=0.005, start_grace_s=0.15, **kw)
    be.start()
    t_end = time.time() + 2
    while not be.readiness().ok and time.time() < t_end:
        time.sleep(0.01)
    return be, robot


@pytest.fixture
def be():
    b, r = make()
    yield b, r
    b.close()


def test_start_discovers_tool_joints_and_limits(be):
    b, r = be
    assert b.joints == ["wrist_yaw", "wrist_pitch", "wrist_roll", "stretch_gripper"]
    assert b.gripper_joint == "stretch_gripper"
    lim = b.get_limits()
    assert lim["lift"] == (0.0, 1.1) and lim["arm"] == (0.0, 0.52)
    assert lim["wrist_yaw"] == (-90.0, 180.0)
    assert "gripper" in lim


def test_start_fails_clearly_when_client_cannot_connect():
    class Bad(FakeRobot):
        def startup(self, **kw):
            return False
    b = StretchBackend(robot_factory=Bad, head_stream=head_stream(), gripper_stream=gripper_stream())
    with pytest.raises(RuntimeError, match="stretch_body_server"):
        b.start()


def test_missing_tool_means_no_wrist_or_gripper_limits():
    b, r = make(FakeRobot(tool_joints=()))
    try:
        assert b.gripper_joint is None
        assert set(b.get_limits()) == {"lift", "arm"}
    finally:
        b.close()


def test_state_is_in_agent_units(be):
    b, r = be
    r.status["end_of_arm"]["wrist_yaw"]["pos"] = math.radians(30)
    r.status["omnibase"]["theta"] = math.radians(90)
    s = b.get_state()
    assert s["wrist_yaw"] == pytest.approx(30) and s["base_theta"] == pytest.approx(90)
    assert s["gripper"] == 0.0


def test_joint_move_completes_and_commands_are_in_robot_units(be):
    b, r = be
    res = b.execute({"arm": 0.05, "wrist_yaw": 10.0, "gripper": 20.0}, timeout=2)
    assert {k: v.status for k, v in res.items()} == {"arm": "completed", "wrist_yaw": "completed", "gripper": "completed"}
    assert len(r.push_log) == 1
    push = r.push_log[0]
    assert push["arm"] == ("move_by", 0.05)
    assert push["wrist_yaw.end_of_arm"][2] == pytest.approx(math.radians(10))   # wrist joints: radians
    assert push["stretch_gripper.end_of_arm"][2] == 20.0                        # gripper: pct, untouched
    assert b.get_state()["gripper"] == pytest.approx(20.0)


def test_translate_and_rotate_are_never_queued_in_the_same_push(be):
    b, r = be
    res = b.execute({"drive": 0.2, "turn": 30.0, "arm": 0.05}, timeout=2)
    assert all(v.status == "completed" for v in res.values()), res
    assert len(r.push_log) == 2
    kinds = [{c[0] for k, c in p.items() if k == "omnibase"} for p in r.push_log]
    assert kinds == [{"translate_by"}, {"rotate_by"}]
    assert "arm" in r.push_log[0]
    assert r.warnings == []          # nothing was overwritten in cmd_dict
    s = b.get_state()
    assert s["base_x"] == pytest.approx(0.2) and s["base_theta"] == pytest.approx(30)


def test_drive_and_slide_share_one_translate(be):
    b, r = be
    res = b.execute({"drive": 0.1, "slide": -0.1}, timeout=2)
    assert all(v.status == "completed" for v in res.values())
    assert r.push_log[0]["omnibase"] == ("translate_by", 0.1, -0.1)


def test_base_displacement_is_measured_in_the_robot_frame(be):
    b, r = be
    r.status["omnibase"]["theta"] = math.radians(90)      # facing +y in the odom frame
    res = b.execute({"drive": 0.2}, timeout=2)
    assert res["drive"].status == "completed" and res["drive"].achieved == pytest.approx(0.2, abs=1e-3)
    assert r.status["omnibase"]["y"] == pytest.approx(0.2)


def test_turn_is_skipped_if_the_first_stage_did_not_complete():
    b, r = make(FakeRobot(stall={"drive": 0.2}))
    try:
        res = b.execute({"drive": 0.2, "turn": 30.0}, timeout=2)
        assert res["drive"].status == "stopped_short"
        assert res["turn"].status == "rejected" and "not attempted" in res["turn"].note
        assert len(r.push_log) == 1
    finally:
        b.close()


def test_wheel_motion_uses_any_not_all_wheels(be):
    b, r = be
    r.omnibase.translate_by(0.1, 0.0)
    r.push_command()
    r.pull_status()
    assert r.status["omnibase"]["wheel_0"]["is_mg_moving"] is False   # one wheel is still ...
    assert b._moving(["drive"]) is True                               # ... but the base is moving


def test_systemexit_from_push_command_is_caught_and_cmd_dict_cleared():
    b, r = make(FakeRobot(push_exits=True))
    try:
        res = b.execute({"arm": 0.05, "lift": 0.05}, timeout=2)        # would kill the server if uncaught
        assert {v.status for v in res.values()} == {"rejected"}
        assert all("control lock" in v.note for v in res.values())
        assert r.cmd_dict == {}      # stale commands must not survive to the next push
        r.push_exits = False
        res = b.execute({"lift": 0.02}, timeout=2)
        assert res["lift"].status == "completed"
        assert list(r.push_log[-1]) == ["lift"]       # only the new command was sent
    finally:
        b.close()


def test_systemexit_through_the_service_does_not_kill_the_server(tmp_path):
    b, r = make(FakeRobot(push_exits=True))
    try:
        svc = make_service(tmp_path, backend=b)
        svc.start_episode("t")
        d = json.loads(svc.move("r", {"arm": 0.05}, return_observation=False)[0])
        assert d["results"]["arm"]["status"] == "rejected" and d["all_completed"] is False
    finally:
        b.close()


def test_no_motion_is_stopped_short_with_the_lease_note():
    b, r = make(FakeRobot(lease_dropped=True))
    try:
        res = b.execute({"arm": 0.05}, timeout=2)
        assert res["arm"].status == "stopped_short"
        assert "lease" in res["arm"].note and "gamepad" in res["arm"].note
        assert res["arm"].achieved == pytest.approx(0.0)
    finally:
        b.close()


def test_stall_reports_stopped_short_with_collision_flag():
    b, r = make(FakeRobot(stall={"arm": 0.5}))
    try:
        res = b.execute({"arm": 0.1}, timeout=2)
        assert res["arm"].status == "stopped_short" and res["arm"].in_collision_stop is True
        assert res["arm"].achieved == pytest.approx(0.05)
    finally:
        b.close()


def test_gripper_stopping_short_is_flagged_as_a_grasp():
    b, r = make(FakeRobot(stall={"gripper": 0.4}))
    try:
        res = b.execute({"gripper": -50.0}, timeout=2)
        assert res["gripper"].status == "stopped_short" and "grasp" in res["gripper"].note
    finally:
        b.close()


def test_runstopped_or_not_homed_refuses_motion():
    for kw in ({"runstopped": True}, {"homed": False}):
        b, r = make(FakeRobot(**kw))
        try:
            res = b.execute({"arm": 0.05}, timeout=1)
            assert res["arm"].status == "rejected" and "runstopped or not homed" in res["arm"].note
            assert r.push_log == []
            assert not b.readiness().ok
        finally:
            b.close()


def test_service_refuses_motion_when_runstopped(tmp_path):
    b, r = make(FakeRobot(runstopped=True))
    try:
        svc = make_service(tmp_path, backend=b)
        svc.start_episode("t")
        with pytest.raises(MotionRefused, match="runstopped"):
            svc.move("r", {"arm": 0.05})
        assert r.pushes_attempted == 0
    finally:
        b.close()


def test_unhomed_joint_client_refusal_is_reported_as_rejected():
    r = FakeRobot()
    r.status["arm"]["motor"]["pos_calibrated"] = False      # is_homed() on the robot says True, arm says no
    b, _ = make(r)
    try:
        res = b.execute({"arm": 0.05, "lift": 0.05}, timeout=2)
        assert res["arm"].status == "rejected" and res["lift"].status == "completed"
    finally:
        b.close()


def test_timeout_halts_the_motion_and_reports_timeout():
    b, r = make(FakeRobot(never_finishes=True))
    try:
        t0 = time.monotonic()
        res = b.execute({"arm": 0.05, "drive": 0.1}, timeout=0.3)
        assert time.monotonic() - t0 < 2
        assert res["arm"].status == "timeout" and res["drive"].status == "timeout"
        stop_push = r.push_log[-1]
        assert stop_push["arm"][0] == "set_velocity" and stop_push["omnibase"][0] == "hard_stop"
    finally:
        b.close()


def test_halt_triggers_runstop_pushes_it_and_interrupts_a_running_move():
    import threading
    b, r = make(FakeRobot(never_finishes=True))
    try:
        box = {}
        th = threading.Thread(target=lambda: box.update(res=b.execute({"arm": 0.05}, timeout=5)))
        th.start()
        time.sleep(0.1)
        b.halt("test")
        th.join(timeout=2)
        assert not th.is_alive()
        assert box["res"]["arm"].status == "stopped_short" and "emergency" in box["res"]["arm"].note
        assert r.push_log[-1]["power_periph"][0] == "trigger_runstop"
        assert b.readiness().runstopped
    finally:
        b.close()


def test_halt_survives_systemexit_and_errors():
    class Hostile(FakeRobot):
        def push_command(self, **kw):
            raise SystemExit(1)
    b, r = make(Hostile())
    try:
        b.halt("x")      # must not raise or exit
    finally:
        b.close()


def test_close_calls_stop_not_a_halt(be):
    b, r = be
    b.close()
    assert r.stopped and r.push_log == []


# ---- cameras ----------------------------------------------------------------------

def test_frames_are_converted_from_bgr_to_rgb(be):
    b, _ = be
    fr = b.get_frames(None, 2.0)
    assert set(fr) == {"head_left", "head_right", "head_center", "gripper"}
    left = fr["head_left"].rgb                      # fake pushes pure blue in BGR order
    assert left.dtype == np.uint8 and tuple(left[0, 0]) == (0, 0, 255)
    assert tuple(fr["head_center"].rgb[0, 0]) == (255, 0, 0)
    assert tuple(fr["gripper"].rgb[0, 0]) == (255, 128, 0)


def test_gripper_depth_is_metres_with_invalid_as_nan_and_on_the_rgb_grid(be):
    b, _ = be
    g = b.get_frames(None, 2.0, ["gripper"])["gripper"]
    assert g.depth.shape == g.rgb.shape[:2] == (48, 64)         # fake depth is 24x32: resampled up
    assert np.isnan(g.depth[0, 0]) and g.depth[10, 10] == pytest.approx(0.64)


def test_stale_frames_time_out(be):
    b, _ = be
    with pytest.raises(FrameTimeout, match="newer than the last motion"):
        b.get_frames(time.time() + 100, 0.2)


def test_frames_must_be_newer_than_motion_end(be):
    b, _ = be
    t = time.time()
    time.sleep(0.05)
    fr = b.get_frames(t, 2.0)
    assert all(f.timestamp > t for f in fr.values())


def test_dead_camera_stream_is_reported_and_blocks_motion():
    r = FakeRobot()
    b = StretchBackend(robot_factory=lambda: r, head_stream=head_stream(fail_after=2), gripper_stream=gripper_stream(),
                       poll_s=0.005)
    b.start()
    try:
        time.sleep(0.3)
        rd = b.readiness()
        assert not rd.ok and any("OAK device disconnected" in p for p in rd.problems)
    finally:
        b.close()


def test_doctor_reports_without_commanding_anything():
    b, r = make()
    try:
        b.doctor_window_s = 0.3
        d = b.doctor()
        assert "lift" in d["status_keys"] and d["gripper_joint"] == "stretch_gripper"
        assert d["cameras"]["head_fps"] > 0 and d["cameras"]["gripper"]["depth"]["valid_fraction"] > 0.9
        assert r.pushes_attempted == 0
    finally:
        b.close()
