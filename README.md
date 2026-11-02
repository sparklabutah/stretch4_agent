# stretch4_agent

An MCP server that lets a coding agent (Claude Code, Codex) drive a Hello Robot Stretch 4 through
bounded, relative moves: look with `observe`, act with `move`, measure with `get_gripper_depth`,
stop with `emergency_stop`, and end with `finish_episode`. One transport (Streamable HTTP with a
bearer token), three interchangeable backends. Design and rationale: [DESIGN.md](DESIGN.md).

| Backend | What it is | Status |
|---|---|---|
| `mock` | NumPy ray-cast room, kinematic robot, consistent depth | Tested. Runs anywhere |
| `sim` | `stretch4_mujoco`, headless MuJoCo | Run end to end on macOS (see Quick start) |
| `stretch` | Real robot via `RobotClient` + camera generators | **Only ever exercised against a fake `RobotClient` and fake camera generators written from the source. No hardware has touched it.** |

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
export STRETCH4_AGENT_TOKEN=$(.venv/bin/stretch4-agent token)    # the server refuses to start without one

.venv/bin/stretch4-agent serve --backend mock                    # http://127.0.0.1:8765/mcp
```

Runs are written to `runs/<UTC timestamp>_<id>/` (`events.jsonl`, `images/`, `episode.json`).
`events.jsonl` splits each call into `model_wait_s` (gap since the previous result, mostly model
inference) and `exec_s`. The outcome in `episode.json` is the agent's own claim, marked unverified.

**Sim.** Install `stretch4_mujoco` from a clone (it brings `mujoco`), then:

```bash
git clone https://github.com/hello-robot/stretch4_mujoco --recurse-submodules
.venv/bin/pip install -e ./stretch4_mujoco -e ".[sim]"
.venv/bin/stretch4-agent serve --backend sim                     # first start takes tens of seconds
```

Headless is the default and works with plain `python` on macOS. A viewer (`--viewer`) needs the
child process started by `mjpython` on macOS (`.venv/bin/mjpython -m stretch4_agent.cli serve
--backend sim --viewer`); that path is untested here. On Linux set `MUJOCO_GL=egl` for GPU rendering.

## Using it from an agent

Start a server (above), keep `STRETCH4_AGENT_TOKEN` exported in the agent's shell too.

**Claude Code, interactive:** `cd` to this directory and run `claude`. `.mcp.json` declares the
`stretch` server (`${STRETCH4_AGENT_URL:-http://127.0.0.1:8765/mcp}`, bearer token from the env);
approve it when prompted. Then ask for a task. `MAX_MCP_OUTPUT_TOKENS=50000` is advisable: four
images in one result exceed the default 25,000.

**Codex, interactive:** `.codex/config.toml` declares the server but Codex reads project config only
for trusted projects (`codex mcp list` here shows nothing until trusted). Either trust the project or
pass the same keys: `codex -c 'mcp_servers.stretch.url="http://127.0.0.1:8765/mcp"' -c
'mcp_servers.stretch.bearer_token_env_var="STRETCH4_AGENT_TOKEN"' -c 'mcp_servers.stretch.required=true'`.

**Headless, either agent:**

```bash
stretch4-agent run --agent claude --model haiku --max-turns 8 --task "Look, extend the arm 5 cm, finish."
stretch4-agent run --agent codex --model gpt-6.1-sol --effort medium --task "..."   # defaults: gpt-6-astra, high
stretch4-agent run --agent claude --task "..." --dry-run                              # print the command
```

The runner finds `claude`/`codex` on `PATH` or in `STRETCH4_AGENT_CLAUDE_BIN` / `STRETCH4_AGENT_CODEX_BIN`.
Claude runs with `--strict-mcp-config --tools "" --allowedTools "mcp__stretch__*"`, so the robot is its
only tool. Codex runs with `--ignore-user-config --sandbox read-only` and the config keys that can be
removed; see Known limitations for what remains.

What was observed on this Mac against `mock` (one run each, tiny task): Claude Code 2.1.309 with
`--model haiku` called `start_episode`, `observe`, `move`, `finish_episode`, and its answer named the
coloured boxes, which appear only in the images. Codex 0.163 with `gpt-6.1-sol` did the same, and also
used `get_gripper_depth`. In both, the tool results contained four JPEG image blocks.

## Moving to a workstation

Copy the directory without `.venv` and `.deps`, create a venv, `pip install -e ".[dev]"` (plus the sim
extras if wanted), run `pytest`. Generate a fresh token; do not copy tokens or `runs/`. Nothing in the
repo contains an absolute path. Nothing about the server changes: run it with `--backend sim`, point
the agent at it.

## NUC deployment (real robot)

Topology: `stretch_body_server` and the OAK cameras live on the robot's **NUC** (the Jetson is a GPU
expansion). Camera generators open the devices in their own process, so the server must run on the
NUC, next to `RobotClient`. The agent runs wherever you like and reaches the server through an SSH
tunnel.

```bash
# on the NUC, in the environment that has hello-robot-stretch4-body
pip install -e ".[robot]"
stretch_body_server --daemon                    # if not already running
stretch4-agent token --file .tokens/agent.token
STRETCH4_AGENT_TOKEN_FILE=.tokens/agent.token stretch4-agent serve --backend stretch --read-only

# on the agent machine
ssh -N -L 8765:127.0.0.1:8765 stretch
export STRETCH4_AGENT_TOKEN=...                 # the same token
```

The server binds `127.0.0.1` only and refuses another address without `--allow-remote-bind`. Do not use
that flag; the tunnel is the intended way. Do not run the ROS driver or gamepad teleop while an agent
drives: commands from a client without the control lease are dropped silently (`stretch_gamepad_teleop`
outranks default clients by design; it is also the human override, along with the head runstop).

## Bring-up checklist (real robot)

Everything below is unverified on hardware. Stop at the first surprise. A person stays within reach of the
head runstop button for every stage after 1.

1. **Preflight.** A human homes the robot (`stretch_robot_home`; this server never homes). Clear runstop.
   Free space around the robot. `stretch_body_server` running, no other RobotClient or ROS driver.
2. **Doctor, no motion.** `stretch4-agent doctor --backend stretch`. Check: status keys exist as assumed;
   `limits` has lift, arm and, if present, wrist and gripper (`limits_missing` should be empty or
   understood: a joint without a readable limit is refused, not guessed); camera shapes and measured fps for
   head and gripper streams running together; depth min/max and valid fraction.
3. **Read-only episodes.** `serve --backend stretch --read-only`; run an agent episode. Check image
   orientation and left/right, `observe` after a hand-wave shows fresh frames, depth readings against a
   tape measure, and that `move` is rejected and logged.
4. **One joint at a time**, restart without `--read-only` and with tiny caps
   (`--max-lift 0.02 --max-arm 0.02 --max-wrist-yaw 5 ...`), hand on the runstop. Order: lift, arm, wrist
   yaw, pitch, roll, gripper, then base drive, slide, turn. For each, confirm direction, that
   `achieved` matches a ruler or protractor, and what `stopped_short` and `timeout` look like. Gripper:
   confirm what a given `gripper_pct` does and what closing on an object reports.
5. **Failure paths, still tiny caps.** Start `stretch_gamepad_teleop` or another RobotClient: expect
   `rejected` ("control lock") or `stopped_short` with the lease note, and the server still alive. Set
   `--move-timeout 1` and command a slow move: expect `timeout` and the motion halted. Call
   `emergency_stop` mid-move: expect the robot to stop and refuse motion until a person clears the runstop
   (and re-homes if the robot requires it).
6. **Combined moves, then full episodes** with default caps, a person watching, a task that cannot hurt
   anything. Watch `events.jsonl` timings.

### Unverified until on hardware

From DESIGN.md:

1. Sign of base Y. `robot_client.py` says Y is left; `omnibase.py:move_by` says "Y is right".
2. What base "forward" means. `omnibase.py` says "X is forward (direction of arm)".
3. Depth units in `SyncedImageFrame.depth` (assumed mm, from `stereo_max_range_mm`; the backend scales by 0.001).
4. Whether `trigger_runstop` takes effect without `push_command`. The backend pushes it with
   `ignore_control_lock=True`; whether that works while another process holds the lease is also unknown.
5. Whether the three head streams and the gripper stream can run at once at acceptable rates.
6. `robot.end_of_arm.move_by` on the gripper joint (units through the client; passed as pct).

Added while implementing:

7. Wrist joint limits are read from `robot.robot_params[joint]['range_deg']`; the path and the frame those
   degrees are in are guesses. A joint where this fails has no limits and is refused.
8. Gripper limits come from `get_tool_metadata(joint).command_range`, falling back to (-100, 50) pct.
9. Timeout stop primitives: `lift/arm.set_velocity(0)`, `omnibase.hard_stop()`, `end_of_arm.quick_stop(joint)`.
10. Status paths used for motion detection (`status['omnibase']['wheel_N']['is_mg_moving']`,
    `status['end_of_arm'][joint]['is_moving']`), and `in_collision_stop` on wrist and gripper. The base uses
    OR over wheels because `OmniBaseClient.is_moving` ANDs them and would read "stopped" while the base moves
    along one wheel's axis.
11. Base odometry frame: achieved drive/slide are odometry deltas projected on the starting heading.
12. DESIGN.md says `get_gripper_depth` returns Euclidean depth. The backends return what the camera pipeline
    provides (assumed depth along the optical axis for real hardware and the simulator; the mock is
    Euclidean). Converting needs intrinsics, which this prototype does not use.
13. `RobotClient.pull_status(blocking=False)` returning False means "no new status"; polled for up to 1 s.
14. Frame freshness uses the time our thread received the frame, not the camera timestamp.

## Known limitations

- **The `stretch` backend is untested against hardware.** Its tests prove its logic against a fake shaped
  from the stretch4_body source, not that the robot behaves that way.
- **Codex is not fully locked down.** Asked to list its tools (codex 0.163, the runner's flags), the model
  named `exec`, `apply_patch`, `request_user_input`, `sleep`, the `collaboration.*` sub-agent tools and MCP
  resource readers, plus the stretch tools through tool discovery. `--disable` removed web search, image
  generation, `view_image` and goals, but not `exec`, `apply_patch` or the sub-agent tools; no config key
  found removes them. The model's self-report of its tools is the only evidence here. The runner adds
  `--sandbox read-only` and `--ignore-user-config` (otherwise the user's computer-use plugins load). If that
  matters, run agents as an unprivileged user in a container. Claude Code is restricted to the MCP server
  by `--tools ""` and `--allowedTools`.
- Single client, single episode at a time. `start_episode` abandons an active one.
- `emergency_stop` is latched by the backend (mock/sim: until the server restarts; real: until a person
  clears the runstop). There is no tool to clear it, by design.
- The mock is geometry, not physics. The sim's head cameras are fisheye and rotated, its wrist limits are the
  simulator's generic ones, and its gripper percent is a linear rescale of the aperture. Neither reproduces
  the real lease, runstop or contact behaviour.
- Depth is a single pixel, no filtering. Base `drive`/`slide`/`turn` use odometry only: no obstacle avoidance
  beyond the robot's own guards. No absolute end-effector goals, no Nav2.
- Invalid-argument rejections (unknown keys, wrong types) happen in the MCP layer and are logged by a small
  middleware; extra keys are forbidden by editing the SDK's argument models, which relies on a private
  attribute of `MCPServer`. A test covers it.
- HTTP without TLS; the loopback bind plus SSH tunnel is the security model. Images are stored in `runs/`
  and Claude Code also keeps copies under `~/.claude/projects/`.
- The agent's `success` is a claim, never checked.
