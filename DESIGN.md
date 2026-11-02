# stretch4_agent: design

Agent-agnostic control of a Hello Robot Stretch 4. One MCP server owns the robot; Claude
Code and Codex are interchangeable clients. Evidence and source facts are cited so each
decision can be revisited.

## Topology (the important correction)

`stretch_body_server` and the cameras are on the robot's **Intel NUC**, not the Jetson. The
Jetson Orin (`192.168.1.101`, user `jetson1`) is a GPU expansion on the internal switch; the
NUC is `192.168.1.100` internally and on Wi-Fi externally.

Cameras are not available through `RobotClient` (`stream_*` generators open DepthAI devices
locally), so a remote `RobotClient` cannot see. Hence:

```
Claude Code / Codex  (Mac, workstation, or Jetson)
        |  MCP over Streamable HTTP + bearer token, through an SSH tunnel
        v
stretch4-agent serve --backend stretch        (on the NUC, 127.0.0.1:8765)
        |  RobotClient over local IPC     |  camera generators in threads
        v                                 v
stretch_body_server (100 Hz, guards)     OAK cameras
```

Same code with `--backend sim` or `--backend mock` runs anywhere. There is exactly one
transport (HTTP) everywhere. No stdio: `RobotClient`, MuJoCo child processes and libraries
print to stdout, which corrupts an MCP stdio stream.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Agent interface | MCP tools, one server | Both CLIs consume MCP. Claude Code passes MCP images inline (docs); Codex converts MCP image content to `input_image` when the model supports images (`codex-rs/protocol/src/models.rs`, `convert_mcp_content_to_items`) |
| Action space | One relative 9-key waypoint per call | GT-RAIL's working Stretch 4 configuration. Astranomy found absolute EEF much better than joint in Direct mode (72.7 vs 48.7) but nearly equal in Agentic mode (86.0 vs 84.0), and this is an agentic loop. Absolute EEF is a later experiment |
| Depth | Point-query tool, never a depth image | GT-RAIL default 3/5, colour-mapped depth image 1/5 (n=5 each, indicative only) |
| Control style | Bounded position moves only; no velocity tool | A dropped link cannot leave the robot driving |
| Termination | `finish_episode` requires evidence and a fresh observation | GT-RAIL false "done" 3/5 at ultra, 4/5 at low. Logged as a self-claim, never as ground truth |
| Prompt | Lives in the package, delivered by `start_episode` | Same text to any client. `AGENTS.md` is canonical for repo instructions, `CLAUDE.md` is `@AGENTS.md` |
| Base motion | Translate then rotate, sequentially | `_queue_command` keys on subsystem, so a second `omnibase` command overwrites the first (`core/subsystem_client.py`) |

## MCP tools

`mcp` is 2.x: `from mcp.server.mcpserver import MCPServer` (FastMCP no longer exists). Pin
`mcp>=2,<3`. The SDK's DNS-rebinding protection checks Host headers; a tunnel presents
`127.0.0.1:8765`, keep the protection on.

- `start_episode(task: str)`: opens a log directory, resets budgets, returns the operating
  prompt, move arguments with units and per-move limits, joint limits, camera descriptions,
  budgets, read-only flag.
  Motion is rejected before this.
- `observe(cameras: list[str] | None = None)`: images plus state text, plus `frame_id`.
- `move(reasoning, drive_m=0, slide_m=0, turn_deg=0, lift_m=0, arm_m=0, wrist_yaw_deg=0,
  wrist_pitch_deg=0, wrist_roll_deg=0, gripper_pct=0, return_observation=True)`: blocks until
  done; returns per-joint result and, by default, a fresh observation. All values are relative.
  The unit is part of every argument name and the schema repeats it in each argument's
  `description` (degrees say "not radians"), because a metres/degrees mix-up moves hardware.
  Results, `state` and `joint_limits` use the bare joint names (`arm`); `start_episode`
  lists the arguments as `move_arguments: {arm_m: {unit, max_abs_per_move, reported_as}}`
  so a bare name is never shown as a key.
- `get_gripper_depth(points: list[[x, y]], frame_id)`: up to 64 pixel coordinates **in the
  downscaled gripper image the agent saw**; returns Euclidean depth in cm or null. At most 32
  calls between moves. A stale `frame_id` is an error, since the depth would not match the image.
- `emergency_stop(reason)`: halts now. On the real robot this latches the runstop, and only a
  human clears it. The service also latches it: motion stays refused until the server is
  restarted, so a runstop that did not take effect cannot let motion resume. There is
  deliberately no tool to clear a runstop or to home.
- `finish_episode(outcome: "success"|"failure"|"blocked", evidence: str)`: rejected if no
  observation has been taken since the last motion. Ends motion until the next `start_episode`.

Per-joint status: `completed`, `stopped_short` (with `in_collision_stop` when known; for the
gripper this usually means a grasp), `timeout`, `rejected`. Any invalid key rejects the
**whole** command with reasons and moves nothing. Every tool's schema has
`additionalProperties: false`; an unknown name (`arm`, `arm_cm`) is rejected before the call
runs, with the declared name and its unit in the message. Any subset of declared names is valid.

Images: RGB JPEG quality 80, longest side default 768 (`--image-max-dim`). Remember the scale
for each camera so depth queries map back to full resolution.

## Safety layer (server side, the agent cannot change it)

Defaults, overridable by serve flags only:

- Per-move limits: drive/slide 0.30 m, turn 45 deg, lift 0.15 m, arm 0.15 m, wrist 45 deg.
- Target joint limits checked before sending.
- Budgets: 150 moves, 30 min per episode, per-move motion timeout 15 s.
- Non-finite numbers rejected. Motion arguments are strict numbers: a boolean or a string is
  rejected, not coerced (`true` would otherwise become 1.0).
- One tool call at a time (except `emergency_stop`). A move sent while another call runs is
  refused; `observe`, `get_gripper_depth`, `start_episode` and `finish_episode` wait for a running
  move and then see its result. Limits are checked inside that lock, against the state the move
  starts from.
- Serve flags are validated: a cap or budget that is missing, non-finite or not positive stops
  the server from starting (a NaN cap would otherwise disable the cap).
- Real robot: refuses motion if not homed or runstopped. Never homes on its own.
- `--read-only`: everything runs except motion, which is rejected and logged. For bring-up.
- Motion timeout: halt, report `timeout`.

Network: always require a bearer token (`STRETCH4_AGENT_TOKEN`, constant-time compare, refuse
to start if unset). Default bind `127.0.0.1`. Refuse a non-loopback bind unless
`--allow-remote-bind` is passed. Remote access is an SSH tunnel:
`ssh -N -L 8765:127.0.0.1:8765 stretch`.

Human override stays physical: head runstop button, and `stretch_gamepad_teleop` outranks
default clients by lease priority.

## Backends (`Protocol` in `backends/base.py`)

Methods: `start()`, `close()`, `get_state()`, `get_limits()`, `execute(deltas, timeout) ->
per-joint results`, `halt(reason)`, `get_frames(after_time, timeout) -> {camera: Frame}`,
`readiness()` (homed, runstopped, problems). Frames carry RGB array, timestamp, and for the
gripper a depth array in metres aligned to the gripper image.

### `mock`
Pure NumPy/Pillow kinematic robot with a synthetic scene whose rendered images change with
state and whose depth is consistent. For tests and client wiring. No MuJoCo import.

### `sim`
`Stretch4MujocoSimulator(cameras_to_use=...)`, `start(headless=True)`. Install extra `sim`.
Read `.deps/stretch4_mujoco/stretch4_mujoco/stretch_mujoco_simulator.py`,
`stretch4_mujoco.py`, `datamodels/status_stretch_camera.py`, `examples/camera_feeds.py`,
`examples/base_motion.py`. Uses multiprocessing, so entry points need a `__main__` guard.

### `stretch` (real robot; cannot be run in this environment)
Install extra `robot` (`hello-robot-stretch4-body`, Linux, on the NUC). Import lazily so other
backends never need it. Source facts to honour:

- `RobotClient()`; `startup()` returns False on failure; `with` yields None on failure.
- Status: `robot.pull_status()`; `robot.status['lift'|'arm']['pos']`,
  `['in_collision_stop']`, `['soft_motion_limits']` (a plain `[min, max]` list, the joint's
  `soft_motion_limits['current']`; wrist joints publish the same in radians, `[None, None]`
  until homed);
  `robot.status['omnibase']['x','y','theta']`;
  `robot.status['end_of_arm'][joint]['pos']`; gripper uses `pos_pct`.
- Commands: `robot.lift.move_by(m)`, `robot.arm.move_by(m)`,
  `robot.end_of_arm.move_by(joint, value)` (radians for wrist joints; pct for the
  gripper, passed through to the server's `StretchGripper.move_by(delta_pct)`),
  `robot.omnibase.translate_by(x, y)`, `robot.omnibase.rotate_by(rad)`; all queued until
  `robot.push_command()`.
- Moves return False when a joint is not homed; treat as rejected.
- `push_command` calls `sys.exit(1)` when another process holds the control lock. **Catch
  `SystemExit`** and report rejected, or the MCP server dies.
- Lease (1.1 s): commands from a non-holder are dropped silently. A move producing no motion
  must therefore be reported as `stopped_short` with a note about a possible lease conflict
  (ROS driver or gamepad running).
- `robot.is_homed()`, `robot.is_runstopped()`, `robot.trigger_runstop()`. `RobotClient.stop()`
  closes the connection; it is not a motion halt.
- `robot.wait_on_motion_finish(names, timeout)` does not report whether it timed out (the
  bool from `_wait_on_status` is dropped), so poll `is_moving` yourself.
- Tool may vary: check `robot.end_of_arm.joints` / `is_tool_present`; do not hardcode.
  Only the `stretch_gripper` joint takes percent; `parallel_gripper` takes aperture metres
  (`ParallelGripper.move_by(x_m)`). The backend exposes `gripper` only for `stretch_gripper`
  and refuses gripper moves on any other tool rather than send a value in the wrong unit.
- `RobotClient.stop()` calls each subsystem's `stop()`: the base goes to freewheel and lift/arm
  to `enable_safety`, pushed with `ignore_control_lock=True`.
- Cameras: run generators in daemon threads keeping the latest frame.
  `stream_left_right_center_camera_compressed()` gives `.left/.right/.center`, BGR after the
  default decode. `stream_gripper_camera_compressed()` gives `.right` and `.depth`; depth is
  aligned to the **right** gripper camera (`setDepthAlign`), so that is the gripper image.
  The pipeline rotates and crops only colour images (head cameras rotate by default, the
  gripper does not), never depth; depth is resampled only for a pure scale difference. A
  missing right image arrives as a black `ImageFrame(timestamp=0)` and is skipped.
- Convert BGR to RGB. Freshness: wait for a frame newer than motion end, up to 5 s, else error.

`stretch4-agent doctor --backend stretch` prints status keys, limits, camera shapes and rates
with no motion.

### Unverified until on hardware (put in the bring-up checklist, do not guess)

1. Sign of base Y. `robot_client.py` says Y is left; `omnibase.py:move_by` says "Y is right".
2. What base "forward" means. `omnibase.py` says "X is forward (direction of arm)".
3. Depth units in `SyncedImageFrame.depth` (assumed mm, from `stereo_max_range_mm`).
4. Whether `trigger_runstop` takes effect without `push_command`.
5. Whether the three head streams and the gripper stream can run at once at acceptable rates.
6. `robot.end_of_arm.move_by` on the gripper joint (units through the client).

## Client wiring

- `.mcp.json`: `type: http`, `url: ${STRETCH4_AGENT_URL:-http://127.0.0.1:8765/mcp}`,
  header `Authorization: Bearer ${STRETCH4_AGENT_TOKEN}`, server name `stretch`.
- `.codex/config.toml`: `[mcp_servers.stretch]`, `url`, `bearer_token_env_var`,
  `required = true`, `tool_timeout_sec = 120`, `default_tools_approval_mode = "approve"`.
  Project config applies to trusted projects only; the runner should pass overrides with `-c`
  so it does not depend on trust.
- `AGENTS.md` (canonical), `CLAUDE.md` containing `@AGENTS.md`.

`stretch4-agent run --agent claude|codex --task "..." [--url] [--model] [--effort]
[--max-turns]` builds the headless command:

- Claude: `claude -p PROMPT --strict-mcp-config --mcp-config .mcp.json --tools ""
  --allowedTools "mcp__stretch__*" --permission-mode dontAsk --output-format stream-json
  --verbose --max-turns N`, with env `MAX_MCP_OUTPUT_TOKENS=50000` (four images per call hit
  the 25,000 default) and `MCP_TOOL_TIMEOUT=120000`.
- Codex: `codex exec --json --sandbox read-only ...`. Confirm the flags with
  `codex exec --help`. Default model `gpt-6-astra`, effort `high` (the GT-RAIL default).
  Disable shell access if a config key for it exists; if not, state that in the README.

Both CLIs resolved from `PATH`, or `STRETCH4_AGENT_CLAUDE_BIN` / `STRETCH4_AGENT_CODEX_BIN`.

## Logging

`runs/<UTC timestamp>_<id>/`: `events.jsonl` (every tool call, arguments, results, timings,
split into model wait vs execution), `images/`, `episode.json` (task, outcome, self-claim
marked as unverified, counters). The time gap between a tool result and the next tool call
approximates inference latency, the quantity GT-RAIL (82% of episode) and Astranomy (95-98%)
both identify as dominant. Gitignore `runs/`.

## Layout

```
pyproject.toml            extras: sim, robot, dev
README.md                 Quick start, workstation move, NUC deployment, bring-up checklist
AGENTS.md, CLAUDE.md, .mcp.json, .codex/config.toml, .gitignore
src/stretch4_agent/
  __init__.py  cli.py  server.py  safety.py  schema.py  episode.py  imaging.py  runner.py
  prompts/operating_prompt.md
  backends/  __init__.py  base.py  mock.py  sim.py  stretch.py
tests/
```

No absolute paths anywhere: the project moves to a workstation next.

## Out of scope for this prototype

Absolute EEF/IK, Nav2, ROS 2 backend, prompt-supplied camera geometry (Astranomy's biggest
lever in Direct mode, smaller in Agentic), fast local policy under a slow agent, multi-robot.
