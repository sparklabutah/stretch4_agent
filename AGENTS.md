# stretch4_agent

An MCP server that lets a coding agent control a Hello Robot Stretch 4 through bounded relative
moves. Design and rationale: `DESIGN.md`. Usage: `README.md`.

## If you are here to operate the robot

Use only the `stretch` MCP tools (`start_episode`, `observe`, `move`, `get_gripper_depth`,
`emergency_stop`, `finish_episode`). Call `start_episode` first: it returns the operating
instructions, limits and budgets. You do not need a shell, and the server rejects anything outside
its safety limits whatever you ask for. If anything looks unsafe, call `emergency_stop`.

## If you are here to change the code

- Python 3.10+, `pip install -e ".[dev]"`, tests with `pytest`. `mcp` is 2.x
  (`from mcp.server.mcpserver import MCPServer`; there is no FastMCP).
- Layout: `service.py` is the tool logic, `server.py` the HTTP/auth wrapper, `safety.py` the limits,
  `backends/` the three robots (`mock`, `sim`, `stretch`).
- On the robot, agent CLIs run as `stretch-agent` (`scripts/nuc/setup_agent_user.sh`, `--run-as`), checked by
  `isolation.py`. Do not weaken that user's isolation; Codex keeps a shell.
- The safety layer lives server-side and must not become agent-configurable. Do not add a velocity
  tool, a tool that clears a runstop, or one that homes.
- `backends/stretch.py` has only ever run against the fake in `tests/fakes.py`. Do not describe
  hardware behaviour as verified; add to the "Unverified until on hardware" list in the README instead.
- No absolute paths in the repo. `runs/`, tokens, `.venv/` and `.deps/` are gitignored.
- stdout is not a channel: there is no stdio transport, and libraries here print freely.
