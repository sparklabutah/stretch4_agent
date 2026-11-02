# Handoff

State as of 2026-11-02, branch `claude/bold-pasteur-f3wo40` (not merged). Read `DESIGN.md` for settled
decisions, `README.md` for usage and the hardware bring-up checklist. Reviewed line by line so far: the
MCP argument handling (`server.py`), `backends/stretch.py` against the `stretch4_body` source, and
`service.py` and `safety.py`. Not reviewed: `sim.py`, `mock.py`, `runner.py`, `episode.py`.

## What has been observed

| Item | Result | Evidence |
|---|---|---|
| Tests before any change (cloud Linux, Python 3.13.18, mcp 2.3.0) | 96 passed | `pytest`, matching the previous handoff |
| Tests now | 120 passed | `pytest`, mock backend and a fake `RobotClient` only |
| Issue 1 reproduced with an MCP client | `{"arm": 0.05, "reasoning": ...}` rejected, `{"arm_m": 0.05, ...}` accepted | Real `mcp.Client` over HTTP against the committed code; see below |
| `sim` backend, headless Linux, no GPU | Works with Mesa software EGL | `doctor --backend sim` returned 4 camera shapes; served one agent episode |
| Claude Code 2.1.316, `--model haiku`, `sim` | Completed, 1 move, 13.3 s, about $0.004 | `events.jsonl`: `start_episode`, `observe`, `move {"arm_m": 0.05}` (achieved 0.0494 m), `finish_episode`, no errors. I compared the saved head-centre images by eye: the gripper is further out after the move |
| Claude Code 2.1.309, `--model haiku`, `mock` (Mac, earlier session) | Completed, 1 move, 4.3 s | Previous handoff; logs not in the repo |
| Codex 0.163, `gpt-6.1-sol`, `mock` (Mac, earlier session) | Completed, 1 move, 19.7 s; first `move` rejected (issue 1) | Previous handoff; logs not in the repo |
| Codex in this cloud session | Not run | No `codex` binary and no `~/.codex` here; not worked around |
| `stretch` backend | Never run on hardware | Fake only, now corrected to the source's status shapes |

All `success` outcomes are the agent's own claims. Run logs stay in `runs/` (gitignored), not in the repo.

## Resolved this session

1. **Issue 1, `move` rejected `{"arm": ...}`.** Reproduced with an MCP client. The strict-argument patch
   was working: in the committed code the declared key is `arm_m`, not `arm`, and a subset of declared
   keys (`{"arm_m": 0.05}`) was already accepted. The previous handoff's premise that `arm` was declared
   did not match the code. *Assumed, not observed:* Codex used `arm` because the server showed it that
   name everywhere else. `start_episode` keyed `units` and `per_move_limits` by `arm`, results and state
   use `arm`, and the prompt listed the bare names next to the arguments. The Codex run's logs are gone,
   so the schema it saw cannot be checked. Changes:
   - every tool schema advertises `additionalProperties: false`;
   - an unknown name is rejected before the call runs, naming the right key and its unit ("did you mean
     'arm_m'? Its value is in metres"), and logged with `stage: "schema"`;
   - the server refuses to start if an SDK change stops the private-attribute patch from forbidding
     extras;
   - `start_episode` returns `move_arguments` keyed by argument name (`arm_m: {unit, max_abs_per_move,
     reported_as}`) and `state_units`; the old `units` and `per_move_limits` are gone.
   Regression tests: a subset of declared keys passes; one unknown key rejects the whole call and nothing
   moves; the exact Codex call gets the hint.
2. **Issue 2, units in the schema.** The code already used suffixed names matching `DESIGN.md`. Each
   `move` argument now also has a description stating its unit (degrees say "not radians"). The prompt,
   `DESIGN.md`, `README.md` and tests agree, and a test checks every key's description.
3. **`backends/stretch.py` review** against `hello-robot/stretch4_body` at 282e31c (2026-10-30). Fixed:
   - `status[joint]['soft_motion_limits']` is a plain `[min, max]` list. The backend read `['current']`
     from it, which raises `TypeError` in `start()`. The fake had the same wrong shape, so tests passed.
     With a source-accurate fake the old backend fails 32 of 33 backend tests.
   - Wrist limits now come from each joint's published `soft_motion_limits` (radians, padded), falling
     back to `range_deg` minus `range_pad_deg`. Before, the padding was ignored.
   - Only `stretch_gripper` is commandable as `gripper`. `parallel_gripper.move_by` takes aperture metres,
     and the old code would have sent percent values to it as metres. State no longer falls back to the
     gripper's radian `pos`.
   - The gripper adapter's placeholder frame (black, `timestamp=0`) is skipped.
   - Depth is resampled only for a pure scale difference; otherwise withheld.
   Consistent with the source and unchanged: per-joint end-of-arm queue keys (`{joint}.end_of_arm`, so
   wrist joints do not overwrite each other), `omnibase` translate/rotate overwrite, `push_command`'s
   `sys.exit(1)`, OR over wheels for base motion, gripper percent passed through to
   `StretchGripper.move_by(delta_pct)`, odometry integrated in the odom frame, depth aligned to the right
   gripper imager. Added to the README's unverified list as items 15 to 18, including that
   `RobotClient.stop()` (called on server close) leaves the base in **freewheel**.
4. **`service.py` and `safety.py` review.** Fixed, each with a test that fails on the old code:
   - Limits were checked before taking the move lock, so a move could be validated against a state
     that changed before it started. Validation now happens inside the lock.
   - `observe` (and depth, finish, start) could run beside a move and mark mid-motion images as the
     fresh observation that `finish_episode` and depth queries rely on. Now one tool call runs at a
     time (except `emergency_stop`): a second move is refused; the others wait for the move.
   - Pydantic's lax mode turned `true` into 1.0 and `"0.05"` into 0.05 before the safety layer, so
     its bool check never ran over MCP. Motion arguments are now strict (integers still accepted).
   - `emergency_stop` now also latches in the service until the server restarts, so a runstop that
     does not take effect on hardware (README unverified item 4) cannot let motion resume. Bring-up
     step 5 and the README describe the restart. This changes recovery on the robot; revert it if a
     restart is unacceptable.
   - `SafetyConfig` accepted a NaN cap (which compares False and disables the cap), zero, negative or
     missing caps and budgets. Now refused; `serve` exits with a config error.
   - `validate_command` raised `KeyError` for a joint with limits but no state; now refuses it.
   Read and judged fine: per-move caps before joint limits, whole-command rejection, zero-move
   rejection, readiness, read-only and budget checks before every move, halt and reporting on a
   backend exception, frame freshness after motion, depth pixel mapping, finish requiring a fresh
   observation.

## Open issues, in priority order

1. **Codex keeps `exec`, `apply_patch` and sub-agent tools**, per the model's self-report. The runner
   uses a read-only sandbox, but on real hardware run Codex as an unprivileged user or in a container.
2. **Stretch's ZMQ ports 23114-23116 bind on all interfaces without authentication**
   (`core/client_server.py` in `stretch4_body`). Anyone on the robot's network can send commands,
   independent of this project. Firewall them on the NUC.
3. **No Codex run against `sim`**, and Claude has driven the sim once, with Haiku, on a one-move task.
4. **`serve --backend sim` exits with code 1 on SIGINT.** The log shows a clean shutdown and no
   traceback. Not investigated; harmless so far.

## Next steps

1. Run Codex against `--backend sim` on a machine where it is logged in. Watch for `stage: "schema"`
   events in `events.jsonl` to see whether the new rejection message is enough.
2. Review the rest of the code (`service.py`, `safety.py`, `sim.py`) the way `stretch.py` was reviewed.
3. On the robot: follow the README bring-up checklist in order, from `doctor` and `--read-only`, with
   a person at the runstop. `doctor` should now show `limits` for lift, arm and the wrists, and an empty
   `unsupported_tool_joints` with the SG4 gripper. Resolve every "Unverified until on hardware" item
   before a full episode.
4. Reproduce GT-RAIL's ablation shape (reasoning effort, depth on/off) with Claude and Astra on the
   same tasks. No head-to-head has been run.

## Resume on any machine

`.venv`, `.deps`, `runs/` and tokens are gitignored and must be recreated.

```bash
git clone <repo-url> stretch4_agent
```

```bash
cd stretch4_agent && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
.venv/bin/pytest -q
```

Expect 120 passed. For the simulator (needs Python 3.12 or newer; on headless Linux also
`export MUJOCO_GL=egl`, and without a GPU install Mesa's EGL first):

```bash
sudo apt-get install -y libegl1 libegl-mesa0 libgl1-mesa-dri
```

```bash
git clone --recurse-submodules https://github.com/hello-robot/stretch4_mujoco .deps/stretch4_mujoco
```

```bash
.venv/bin/pip install -e .deps/stretch4_mujoco -e ".[sim]"
```

To review the robot backend, clone the SDK source next to it (not installed; Linux-only hardware
libraries):

```bash
git clone https://github.com/hello-robot/stretch4_body .deps/stretch4_body
```

Start a server, then run an agent from a second shell:

```bash
.venv/bin/stretch4-agent token --file .tokens/agent.token
```

```bash
STRETCH4_AGENT_TOKEN_FILE=.tokens/agent.token .venv/bin/stretch4-agent serve --backend sim
```

```bash
export STRETCH4_AGENT_TOKEN="$(cat .tokens/agent.token)"
```

```bash
.venv/bin/stretch4-agent run --agent claude --model haiku --max-turns 8 --task "Observe once, extend the arm 5 cm, finish."
```

## Constraints for whoever continues

- Never add a tool that homes the robot, clears a runstop, or commands velocity.
- Never commit tokens or `runs/` (images may show people and rooms).
- A cloud session has no robot and may have no Codex login. Report that rather than working around it.
- Report what was observed separately from what is assumed.
