import json
import threading
import time

import pytest

from conftest import PatternBackend, make_service
from stretch4_agent.backends.mock import MockBackend
from stretch4_agent.safety import SafetyConfig
from stretch4_agent.schema import MotionRefused


def doc(result):
    return json.loads(result[0])


# ---- motion gating -----------------------------------------------------------

def test_motion_rejected_before_start_episode(svc):
    with pytest.raises(MotionRefused, match="no active episode"):
        svc.move("r", {"arm": 0.05})
    assert svc.backend.get_state()["arm"] == 0.1
    # the rejection is still logged, to the server-level log
    logs = list((svc.cfg.runs_dir).glob("_server_*.jsonl"))
    assert logs and "no active episode" in logs[0].read_text()


def test_motion_rejected_after_finish_episode(started):
    started.observe()
    started.finish_episode("success", "arm is where I wanted it")
    with pytest.raises(MotionRefused, match="no active episode"):
        started.move("r", {"arm": 0.05})
    assert started.backend.get_state()["arm"] == 0.1


def test_motion_rejected_in_read_only_mode_but_everything_else_works(tmp_path):
    svc = make_service(tmp_path, safety=SafetyConfig(read_only=True), image_max_dim=80)
    info = json.loads(svc.start_episode("look")[1])
    assert info["read_only"] and not info["ready_for_motion"]
    assert svc.observe()
    with pytest.raises(MotionRefused, match="read-only"):
        svc.move("r", {"arm": 0.05})
    assert svc.backend.get_state()["arm"] == 0.1
    svc.finish_episode("blocked", "read-only mode, saw the scene")
    events = [json.loads(l) for l in (svc.ep.dir / "events.jsonl").read_text().splitlines()]
    assert any(e["tool"] == "move" and "read-only" in (e["error"] or "") for e in events)


def test_new_episode_after_finish_allows_motion_again(started):
    started.observe()
    started.finish_episode("failure", "x")
    started.start_episode("again")
    assert doc(started.move("r", {"arm": 0.05}, return_observation=False))["all_completed"]


def test_unready_robot_refuses_motion(tmp_path):
    be = MockBackend(size_scale=0.25, homed=False)
    svc = make_service(tmp_path, backend=be)
    svc.start_episode("t")
    with pytest.raises(MotionRefused, match="not homed"):
        svc.move("r", {"arm": 0.05})
    be.homed, be.runstopped = True, True
    with pytest.raises(MotionRefused, match="runstopped"):
        svc.move("r", {"arm": 0.05})


def test_invalid_command_moves_nothing_and_does_not_spend_budget(started):
    before = started.backend.get_state()
    with pytest.raises(MotionRefused):
        started.move("r", {"arm": 0.05, "lift": 5.0})     # one bad key, one good
    assert started.backend.get_state() == before
    assert started.ep.budget.moves == 0 and started.ep.counters["moves_rejected"] == 1


# ---- budgets -----------------------------------------------------------------

def test_move_budget(tmp_path):
    svc = make_service(tmp_path, safety=SafetyConfig(max_moves=2))
    svc.start_episode("t")
    for _ in range(2):
        svc.move("r", {"lift": 0.01}, return_observation=False)
    with pytest.raises(MotionRefused, match="move budget"):
        svc.move("r", {"lift": 0.01})
    svc.observe()   # looking is still allowed
    svc.finish_episode("blocked", "budget")


def test_time_budget(tmp_path):
    t = [0.0]
    svc = make_service(tmp_path, safety=SafetyConfig(max_seconds=60))
    svc.clock = lambda: t[0]
    svc.start_episode("t")
    t[0] = 61.0
    with pytest.raises(MotionRefused, match="time budget"):
        svc.move("r", {"lift": 0.01})


# ---- move semantics -----------------------------------------------------------

def test_move_reports_per_joint_results_and_fresh_observation(started):
    out = started.move("r", {"arm": 0.05, "lift": -0.1})
    d = doc(out)
    assert d["results"]["arm"]["status"] == "completed"
    assert d["results"]["lift"]["status"] == "completed"
    assert json.loads(out[1])["frame_id"] == 1   # observation attached
    assert len(out) == 2 + 2 * 4                 # 2 json docs + (label, image) x 4 cameras


def test_move_into_wall_is_stopped_short(started):
    started.backend.s["base_x"] = 2.6     # wall at 3.0, radius 0.25
    d = doc(started.move("r", {"drive": 0.3}, return_observation=False))
    r = d["results"]["drive"]
    assert r["status"] == "stopped_short" and r["in_collision_stop"] is True
    assert r["achieved"] < 0.3


def test_slow_motion_times_out_and_is_halted(tmp_path):
    svc = make_service(tmp_path, backend=MockBackend(size_scale=0.25, latency_s=2.0),
                       safety=SafetyConfig(move_timeout=0.2))
    svc.start_episode("t")
    t0 = time.monotonic()
    d = doc(svc.move("r", {"arm": 0.1}, return_observation=False))
    assert time.monotonic() - t0 < 1.5
    assert d["results"]["arm"]["status"] == "timeout"


def test_emergency_stop_halts_a_running_move_and_latches(tmp_path):
    be = MockBackend(size_scale=0.25, latency_s=3.0)
    svc = make_service(tmp_path, backend=be)
    svc.start_episode("t")
    box = {}
    th = threading.Thread(target=lambda: box.update(d=doc(svc.move("r", {"arm": 0.1}, return_observation=False))))
    th.start()
    time.sleep(0.3)
    svc.emergency_stop("test")
    th.join(timeout=2)
    assert not th.is_alive()
    assert box["d"]["results"]["arm"]["status"] == "stopped_short"
    with pytest.raises(MotionRefused, match="emergency_stop was called"):
        svc.move("r", {"arm": 0.01})
    assert svc.ep.counters["emergency_stops"] == 1


def test_backend_exception_halts_robot_and_is_reported(started, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("link lost")
    monkeypatch.setattr(started.backend, "execute", boom)
    with pytest.raises(MotionRefused, match="robot halted: RuntimeError: link lost"):
        started.move("r", {"arm": 0.01})
    assert started.backend.runstopped


# ---- finish_episode -------------------------------------------------------------

def test_finish_requires_a_fresh_observation(started):
    with pytest.raises(MotionRefused, match="observation"):
        started.finish_episode("success", "I think so")          # never observed
    started.observe()
    started.move("r", {"arm": 0.01}, return_observation=False)
    with pytest.raises(MotionRefused, match="observation"):
        started.finish_episode("success", "I moved")             # stale: moved since
    started.observe()
    assert doc(started.finish_episode("success", "arm visible at new place"))["finished"]


def test_finish_after_move_with_returned_observation_is_fine(started):
    started.move("r", {"arm": 0.01})
    started.finish_episode("success", "see image")


def test_finish_validates_outcome_and_evidence(started):
    started.observe()
    with pytest.raises(MotionRefused, match="outcome"):
        started.finish_episode("done", "x")
    with pytest.raises(MotionRefused, match="evidence"):
        started.finish_episode("success", "   ")


def test_finish_records_a_self_claim_not_a_verified_outcome(started):
    started.observe()
    started.finish_episode("success", "looks right")
    ep = json.loads((started.ep.dir / "episode.json").read_text())
    assert ep["outcome_self_claimed"] == "success" and ep["outcome_verified"] is False
    assert ep["active"] is False


# ---- observation / logging -------------------------------------------------------

def test_observe_validates_cameras_and_saves_images(started):
    with pytest.raises(MotionRefused, match="unknown camera"):
        started.observe(["nope"])
    out = started.observe(["gripper"])
    assert json.loads(out[0])["depth_available"] is True
    assert len(out) == 3
    assert (started.ep.dir / "images" / "0001_gripper.jpg").exists()


def test_stale_camera_frame_is_an_error(started):
    started.backend.frozen_camera_time = time.time() - 100
    started.move("r", {"arm": 0.01}, return_observation=False)
    with pytest.raises(MotionRefused, match="no fresh camera frame"):
        started.observe()


def test_events_log_has_tool_calls_and_timing_split(started):
    started.observe()
    started.move("careful", {"arm": 0.02}, return_observation=False)
    events = [json.loads(l) for l in (started.ep.dir / "events.jsonl").read_text().splitlines()]
    assert [e["tool"] for e in events] == ["start_episode", "observe", "move"]
    mv = events[-1]
    assert mv["args"]["reasoning"] == "careful" and mv["args"]["arm"] == 0.02
    assert mv["model_wait_s"] is not None and mv["exec_s"] >= 0
    assert mv["results"]["arm"]["status"] == "completed"


# ---- depth -----------------------------------------------------------------------

@pytest.fixture
def depth_svc(tmp_path):
    # full gripper image 160x120, shown at 80x60 (scale 0.5)
    svc = make_service(tmp_path, backend=PatternBackend(size_scale=0.25), image_max_dim=80)
    svc.start_episode("depth")
    return svc


def observe_gripper(svc):
    out = svc.observe(["gripper"])
    d = json.loads(out[0])
    assert d["cameras"]["gripper"]["image_size"] == [80, 60]
    assert d["cameras"]["gripper"]["full_size"] == [160, 120]
    return d["frame_id"]


def test_depth_maps_downscaled_pixels_to_full_resolution(depth_svc):
    fid = observe_gripper(depth_svc)
    d = doc(depth_svc.get_gripper_depth([[10, 20], [0, 0], [79, 59]], fid))
    # shown (10, 20) -> full (20, 40): 0.5 + 0.001*20 + 0.002*40 = 0.6 m
    assert d["depth_cm"] == [60.0, 50.0, 0.5 * 100 + 0.1 * 158 + 0.2 * 118]


def test_depth_nan_and_out_of_range_points(depth_svc):
    fid = observe_gripper(depth_svc)
    for bad in ([80, 0], [0, 60], [-1, 5]):
        with pytest.raises(MotionRefused, match="outside the 80x60"):
            depth_svc.get_gripper_depth([[1, 1], bad], fid)
    depth_svc.ep.depth_frame.depth[:] = float("nan")
    assert doc(depth_svc.get_gripper_depth([[3, 3]], fid))["depth_cm"] == [None]


def test_depth_point_count_limit(depth_svc):
    fid = observe_gripper(depth_svc)
    assert len(doc(depth_svc.get_gripper_depth([[1, 1]] * 64, fid))["depth_cm"]) == 64
    with pytest.raises(MotionRefused, match="1 to 64"):
        depth_svc.get_gripper_depth([[1, 1]] * 65, fid)
    with pytest.raises(MotionRefused, match="1 to 64"):
        depth_svc.get_gripper_depth([], fid)
    with pytest.raises(MotionRefused, match="finite"):
        depth_svc.get_gripper_depth([[float("nan"), 1]], fid)


def test_depth_call_limit_resets_on_move(depth_svc):
    fid = observe_gripper(depth_svc)
    for _ in range(31):
        depth_svc.get_gripper_depth([[1, 1]], fid)
    d = doc(depth_svc.get_gripper_depth([[1, 1]], fid))     # the 32nd
    assert d["depth_calls_left_before_next_move"] == 0
    with pytest.raises(MotionRefused, match="at most 32"):
        depth_svc.get_gripper_depth([[1, 1]], fid)
    out = depth_svc.move("r", {"lift": 0.01})                # a move, observation attached
    new_fid = json.loads(out[1])["frame_id"]
    assert doc(depth_svc.get_gripper_depth([[1, 1]], new_fid))["depth_cm"]


def test_stale_frame_id_is_an_error(depth_svc):
    fid = observe_gripper(depth_svc)
    with pytest.raises(MotionRefused, match="stale frame_id 99"):
        depth_svc.get_gripper_depth([[1, 1]], 99)
    depth_svc.move("r", {"lift": 0.01}, return_observation=False)
    with pytest.raises(MotionRefused, match="stale frame_id"):
        depth_svc.get_gripper_depth([[1, 1]], fid)           # old id after a move
    fid2 = observe_gripper(depth_svc)
    assert fid2 != fid
    with pytest.raises(MotionRefused, match="stale frame_id"):
        depth_svc.get_gripper_depth([[1, 1]], fid)           # old id after a new observation


def test_depth_needs_a_gripper_image(depth_svc):
    out = depth_svc.observe(["head_left"])
    fid = json.loads(out[0])["frame_id"]
    with pytest.raises(MotionRefused, match="no gripper image"):
        depth_svc.get_gripper_depth([[1, 1]], fid)


def test_start_episode_keys_move_arguments_by_their_unit_suffixed_names(svc):
    """The reply must not present bare joint names (`arm`) as move keys: the agent copies them."""
    from stretch4_agent.schema import PARAM, UNITS
    info = json.loads(svc.start_episode("t")[1])
    args = info["move_arguments"]
    assert list(args) == list(PARAM.values())
    assert args["arm_m"] == {"unit": "metres", "max_abs_per_move": 0.15, "reported_as": "arm"}
    assert args["turn_deg"]["unit"] == "degrees" and args["gripper_pct"]["unit"] == "percent"
    assert "units" not in info and "per_move_limits" not in info
    assert set(info["joint_limits"]) <= set(info["state_units"])
    assert set(info["state"]) == set(info["state_units"])


# ---- review of service.py / safety.py (2026-11-02) ---------------------------------------

def test_emergency_stop_latches_in_the_service_even_if_the_backend_does_not(started, monkeypatch):
    """Whether the real robot's runstop takes effect is unverified; motion must stay refused anyway."""
    monkeypatch.setattr(started.backend, "halt", lambda reason: None)      # a halt that latches nothing
    started.emergency_stop("person too close")
    assert started.backend.readiness().ok
    with pytest.raises(MotionRefused, match="emergency_stop was called .*person too close"):
        started.move("r", {"arm": 0.01})
    info = json.loads(started.start_episode("again")[1])
    assert info["ready_for_motion"] is False and any("emergency_stop" in p for p in info["readiness_problems"])
    with pytest.raises(MotionRefused, match="emergency_stop was called"):
        started.move("r", {"arm": 0.01})
    started.observe()                                                       # looking and finishing still work
    assert doc(started.finish_episode("blocked", "stopped for safety"))["finished"]


def _run_slow_move(svc, box, **kw):
    th = threading.Thread(target=lambda: box.update(d=doc(svc.move("r", {"arm": 0.1}, **kw))))
    th.start()
    time.sleep(0.2)
    return th


def test_observe_during_a_move_waits_and_shows_the_result(tmp_path):
    svc = make_service(tmp_path, backend=MockBackend(size_scale=0.25, latency_s=1.0))
    svc.start_episode("t")
    box = {}
    th = _run_slow_move(svc, box, return_observation=False)
    head = doc(svc.observe())                      # sent mid-move: must wait, not show mid-motion images
    th.join(timeout=3)
    assert box["d"]["all_completed"]
    assert head["state"]["arm"] == pytest.approx(box["d"]["state_after"]["arm"])
    assert svc.ep.fresh_observation and svc.ep.current_frame_id == head["frame_id"]


def test_finish_during_a_move_waits_and_then_needs_a_new_observation(tmp_path):
    svc = make_service(tmp_path, backend=MockBackend(size_scale=0.25, latency_s=1.0))
    svc.start_episode("t")
    svc.observe()                                  # fresh before the move ...
    box = {}
    th = _run_slow_move(svc, box, return_observation=False)
    with pytest.raises(MotionRefused, match="take an observation after your last move"):
        svc.finish_episode("success", "it moved")  # ... but stale once the move it waited for ends
    th.join(timeout=3)


def test_a_second_move_during_a_move_is_refused_not_queued(tmp_path):
    svc = make_service(tmp_path, backend=MockBackend(size_scale=0.25, latency_s=1.0))
    svc.start_episode("t")
    box = {}
    th = _run_slow_move(svc, box, return_observation=False)
    with pytest.raises(MotionRefused, match="still running"):
        svc.move("r", {"arm": 0.05})
    th.join(timeout=3)
    assert svc.ep.counters["moves"] == 1
    assert svc.backend.get_state()["arm"] == pytest.approx(0.2)
