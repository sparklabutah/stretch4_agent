import math

import pytest

from stretch4_agent.safety import DEFAULT_PER_MOVE, Budget, SafetyConfig, validate_command
from stretch4_agent.schema import MotionRefused

STATE = {"lift": 0.5, "arm": 0.1, "wrist_yaw": 0.0, "wrist_pitch": -45.0, "wrist_roll": 0.0, "gripper": 0.0}
LIMITS = {"lift": (0.0, 1.1), "arm": (0.0, 0.52), "wrist_yaw": (-90.0, 180.0),
          "wrist_pitch": (-90.0, 20.0), "wrist_roll": (-90.0, 90.0), "gripper": (-100.0, 100.0)}
CFG = SafetyConfig()


def check(cmd):
    return validate_command(cmd, STATE, LIMITS, CFG)


def test_valid_command_returns_only_nonzero_deltas():
    assert check({"arm": 0.1, "lift": 0.0, "turn": -30}) == {"arm": 0.1, "turn": -30.0}


def test_one_unknown_key_rejects_whole_command():
    with pytest.raises(MotionRefused, match="unknown key 'arm_cm'"):
        check({"arm": 0.05, "arm_cm": 5})


def test_all_problems_are_reported_together():
    with pytest.raises(MotionRefused) as e:
        check({"arm": 9, "turn": 400, "bogus": 1})
    msg = str(e.value)
    assert "arm_m" in msg and "turn_deg" in msg and "bogus" in msg


@pytest.mark.parametrize("key,cap", [("drive", 0.30), ("slide", 0.30), ("turn", 45), ("lift", 0.15),
                                     ("arm", 0.15), ("wrist_yaw", 45), ("wrist_pitch", 45), ("wrist_roll", 45)])
def test_per_move_limits_at_and_over_the_cap(key, cap):
    state = dict(STATE, arm=0.2, wrist_pitch=-30.0, wrist_yaw=10.0)
    ok = validate_command({key: cap}, state, LIMITS, CFG)
    assert ok[key] == cap
    with pytest.raises(MotionRefused, match="per-move limit"):
        validate_command({key: cap * 1.01}, state, LIMITS, CFG)
    with pytest.raises(MotionRefused, match="per-move limit"):
        validate_command({key: -cap * 1.01}, state, LIMITS, CFG)


def test_target_joint_limits():
    with pytest.raises(MotionRefused, match="outside the joint limits"):
        check({"arm": -0.15})          # 0.1 - 0.15 < 0
    with pytest.raises(MotionRefused, match="outside the joint limits"):
        validate_command({"lift": 0.15}, dict(STATE, lift=1.0), LIMITS, CFG)   # 1.15 > 1.1
    with pytest.raises(MotionRefused, match="outside the joint limits"):
        validate_command({"wrist_pitch": 45}, dict(STATE, wrist_pitch=0.0), LIMITS, CFG)  # 45 > 20
    assert check({"arm": -0.1})["arm"] == -0.1   # exactly on the limit is allowed


def test_base_has_no_joint_limit_only_per_move_cap():
    assert check({"drive": -0.3, "slide": 0.3})["drive"] == -0.3


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_rejected(bad):
    with pytest.raises(MotionRefused, match="finite"):
        check({"arm": bad})
    with pytest.raises(MotionRefused, match="finite"):
        check({"drive": 0.1, "lift": bad})


@pytest.mark.parametrize("bad", ["0.1", None, True, [0.1]])
def test_non_numbers_rejected(bad):
    with pytest.raises(MotionRefused, match="must be a number"):
        check({"arm": bad})


def test_all_zero_is_rejected():
    with pytest.raises(MotionRefused, match="every delta is zero"):
        check({"arm": 0.0})
    with pytest.raises(MotionRefused):
        check({})


def test_move_budget_and_time_budget():
    t = [0.0]
    b = Budget(SafetyConfig(max_moves=2, max_seconds=10), clock=lambda: t[0])
    b.check()
    b.moves = 2
    with pytest.raises(MotionRefused, match="move budget"):
        b.check()
    b.moves = 0
    t[0] = 10.0
    with pytest.raises(MotionRefused, match="time budget"):
        b.check()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -0.1, True, "0.1"])
def test_a_cap_that_would_weaken_the_limit_is_refused(bad):
    per_move = dict(DEFAULT_PER_MOVE, arm=bad)
    with pytest.raises(ValueError, match="per-move cap for arm"):
        SafetyConfig(per_move=per_move)


def test_missing_or_unknown_caps_and_bad_budgets_are_refused():
    with pytest.raises(ValueError, match="missing"):
        SafetyConfig(per_move={k: v for k, v in DEFAULT_PER_MOVE.items() if k != "turn"})
    with pytest.raises(ValueError, match="unknown per-move key 'arm_m'"):
        SafetyConfig(per_move=dict(DEFAULT_PER_MOVE, arm_m=0.1))
    for kw in ({"max_moves": 0}, {"max_seconds": float("nan")}, {"move_timeout": -1}):
        with pytest.raises(ValueError):
            SafetyConfig(**kw)


def test_serve_refuses_a_nan_cap_cleanly(monkeypatch, capsys):
    from stretch4_agent import cli
    monkeypatch.setenv("STRETCH4_AGENT_TOKEN", "x")
    assert cli.main(["serve", "--backend", "mock", "--max-arm", "nan"]) == 2
    assert "per-move cap for arm" in capsys.readouterr().err


def test_limits_without_a_state_value_refuse_instead_of_crashing():
    state = {k: v for k, v in STATE.items() if k != "gripper"}
    with pytest.raises(MotionRefused, match="no readable limits for gripper"):
        validate_command({"gripper": 5.0}, state, LIMITS, CFG)
