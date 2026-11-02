"""Build and run headless Claude Code / Codex sessions against a running server."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .server import TOKEN_ENV

CODEX_DEFAULT_MODEL = "gpt-6-astra"
CODEX_DEFAULT_EFFORT = "high"


@dataclass
class CommandSpec:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)   # extra env, never the token
    cwd: Path = Path(".")


def _binary(agent: str) -> str:
    env_name = f"STRETCH4_AGENT_{agent.upper()}_BIN"
    path = os.environ.get(env_name) or shutil.which(agent)
    if not path:
        raise SystemExit(f"cannot find `{agent}` on PATH; set {env_name} to its full path")
    return path


def _prompt(task: str) -> str:
    return (f"Use the `stretch` MCP tools to carry out this task on the robot: {task}\n"
            "Begin by calling start_episode with the task and follow the operating instructions it returns. "
            "You have no other tools. Finish with finish_episode.")


def build_command(agent: str, task: str, *, url: str, model: str | None = None, effort: str | None = None,
                  max_turns: int = 40, project_dir: Path = Path(".")) -> CommandSpec:
    project_dir = Path(project_dir)
    prompt = _prompt(task)
    if agent == "claude":
        argv = [_binary("claude"), "-p", prompt,
                "--strict-mcp-config", "--mcp-config", str((project_dir / ".mcp.json").resolve()),
                "--tools", "", "--allowedTools", "mcp__stretch__*",
                "--permission-mode", "dontAsk",
                "--output-format", "stream-json", "--verbose", "--max-turns", str(max_turns)]
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        # Four images per call exceed Claude Code's default 25,000-token MCP output cap.
        env = {"STRETCH4_AGENT_URL": url, "MAX_MCP_OUTPUT_TOKENS": "50000", "MCP_TOOL_TIMEOUT": "120000"}
        return CommandSpec(argv, env, project_dir)
    if agent == "codex":
        model = model or CODEX_DEFAULT_MODEL
        effort = effort or CODEX_DEFAULT_EFFORT
        # Overrides on the command line so nothing depends on the project being trusted.
        # --ignore-user-config: the user's ~/.codex/config.toml may enable shell/computer-use plugins;
        # a robot session should see only this server. Auth still comes from CODEX_HOME.
        argv = [_binary("codex"), "exec", "--json", "--ignore-user-config", "--sandbox", "read-only",
                "--skip-git-repo-check", "-C", str(project_dir.resolve()),
                "-c", f'mcp_servers.stretch.url="{url}"',
                "-c", f'mcp_servers.stretch.bearer_token_env_var="{TOKEN_ENV}"',
                "-c", "mcp_servers.stretch.required=true",
                "-c", "mcp_servers.stretch.tool_timeout_sec=120",
                "-c", 'mcp_servers.stretch.default_tools_approval_mode="approve"',
                # Strip what config can strip. Observed with codex 0.163: this removes web search, image
                # generation, view_image, goals and sleep, but `exec`, `apply_patch` and the sub-agent
                # tools remain. The read-only sandbox is the only backstop (see README).
                "--disable", "shell_tool", "--disable", "unified_exec", "--disable", "image_generation",
                "--disable", "view_image", "--disable", "goals", "--disable", "multi_agent",
                "-c", 'web_search="disabled"']
        argv += ["-m", model, "-c", f'model_reasoning_effort="{effort}"']
        argv.append(prompt)
        return CommandSpec(argv, {}, project_dir)
    raise ValueError(agent)


def run_agent(spec: CommandSpec, token: str) -> int:
    env = {**os.environ, **spec.env, TOKEN_ENV: token}
    return subprocess.run(spec.argv, env=env, cwd=spec.cwd, stdin=subprocess.DEVNULL).returncode
