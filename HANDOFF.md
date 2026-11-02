# Handoff

State as of 2026-11-01. Read `DESIGN.md` for settled decisions, `README.md` for usage and the
hardware bring-up checklist. Implementation was stopped on request, so its final self-review
never finished: treat the code as unreviewed.

## What has been observed

| Item | Result | Evidence |
|---|---|---|
| Unit and integration tests | 96 passed | `pytest`, mock backend and a fake `RobotClient` only |
| `sim` backend, Mac, headless | Works when scripted | MCP session: 4 cameras returned rendered frames, 5 cm arm move achieved 4.97 cm, stale `frame_id` refused. Two of the frames were inspected by eye. No agent has driven the sim yet |
| Claude Code 2.1.309, `--model haiku`, `mock` | Completed, 1 move, 4.3 s | Named the green and red cubes, which exist only in the images |
| Codex 0.163, `gpt-6.1-sol`, `mock` | Completed, 1 move, 19.7 s | Named the same cubes; used `get_gripper_depth`; first `move` failed, see issue 1 |
| `stretch` backend | Never run on hardware | Fake only |

Those run logs were in a temporary directory and are not in this repo. All `success` outcomes are
the agent's own claims.

## Open issues, in priority order

1. **Rejected `move` from Codex, unexplained.** Codex sent `{"arm": 0.05, "reasoning": ...}` and got
   `Extra inputs are not permitted` on `arm`. Resending with all nine keys succeeded. Sending a subset
   of keys must work, and `arm` is a declared key, so this looks like a bug in the strict-argument
   patch in `server.py`, which relies on a private `MCPServer` attribute. Reproduce with an MCP client
   and add a regression test.
2. **Parameter names differ from `DESIGN.md`.** Code uses `drive`, `arm`, `wrist_yaw`, `gripper`; the
   design has `drive_m`, `arm_m`, `wrist_yaw_deg`, `gripper_pct`. Units must be unmistakable in the
   tool schema, since a metres/degrees mix-up is a hardware risk. Either restore unit suffixes or
   confirm the schema states units for every key, then update `DESIGN.md`.
3. **Codex keeps `exec`, `apply_patch` and sub-agent tools**, per the model's self-report. The runner
   uses a read-only sandbox, but on real hardware run Codex as an unprivileged user or in a container.
4. **Stretch's ZMQ ports 23114-23116 bind on all interfaces without authentication**
   (`core/client_server.py` in `stretch4_body`). Anyone on the robot's network can send commands,
   independent of this project. Firewall them on the NUC.

## Next steps

1. Fix issues 1 and 2.
2. Run Claude and Codex against `--backend sim` with a small task, keeping cost low.
3. Review `src/stretch4_agent/backends/stretch.py` against the `stretch4_body` source.
4. On the robot: follow the README bring-up checklist in order, from `doctor` and `--read-only`, with
   a person at the runstop. Resolve every "Unverified until on hardware" item before a full episode.
5. Reproduce GT-RAIL's ablation shape (reasoning effort, depth on/off) with Claude and Astra on the
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

Expect 96 passed. For the simulator (needs Python 3.12 or newer; on headless Linux also
`export MUJOCO_GL=egl`):

```bash
git clone --recurse-submodules https://github.com/hello-robot/stretch4_mujoco .deps/stretch4_mujoco
```

```bash
.venv/bin/pip install -e .deps/stretch4_mujoco -e ".[sim]"
```

Start a server, then run an agent from a second shell:

```bash
.venv/bin/stretch4-agent token --file .tokens/agent.token
```

```bash
STRETCH4_AGENT_TOKEN_FILE=.tokens/agent.token .venv/bin/stretch4-agent serve --backend mock
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
