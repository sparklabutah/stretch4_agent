from pathlib import Path

import pytest

from stretch4_agent import cli
from stretch4_agent.runner import build_command


@pytest.fixture(autouse=True)
def bins(monkeypatch):
    monkeypatch.setenv("STRETCH4_AGENT_CODEX_BIN", "/usr/local/bin/codex")
    monkeypatch.setenv("STRETCH4_AGENT_CLAUDE_BIN", "/usr/local/bin/claude")


def test_codex_defaults_are_astra_high_with_a_read_only_sandbox():
    spec = build_command("codex", "t", url="http://127.0.0.1:8765/mcp", project_dir=Path("/srv/w"))
    a = spec.argv
    assert a[0] == "/usr/local/bin/codex" and a[1] == "exec"
    assert a[a.index("-m") + 1] == "gpt-6-astra" and 'model_reasoning_effort="high"' in a
    assert a[a.index("--sandbox") + 1] == "read-only" and a[a.index("-C") + 1] == "/srv/w"
    assert spec.env == {}


def test_run_as_wraps_in_non_interactive_sudo_and_passes_only_the_token_and_settings():
    spec = build_command("codex", "t", url="u", run_as="stretch-agent")
    assert spec.argv[:7] == ["sudo", "-n", "-u", "stretch-agent", "-H", "--preserve-env=STRETCH4_AGENT_TOKEN", "--"]
    assert spec.argv[7] == "/usr/local/bin/codex"
    spec = build_command("claude", "t", url="u", run_as="stretch-agent")
    keep = spec.argv[5].split("=", 1)[1].split(",")
    assert keep == ["STRETCH4_AGENT_TOKEN", "STRETCH4_AGENT_URL", "MAX_MCP_OUTPUT_TOKENS", "MCP_TOOL_TIMEOUT"]
    assert spec.argv[7] == "/usr/local/bin/claude"


def test_codex_without_run_as_warns(monkeypatch, capsys):
    monkeypatch.setenv("STRETCH4_AGENT_TOKEN", "x")
    assert cli.main(["run", "--agent", "codex", "--task", "t", "--dry-run"]) == 0
    assert "warning: Codex keeps a shell" in capsys.readouterr().err
    assert cli.main(["run", "--agent", "codex", "--task", "t", "--dry-run", "--run-as", "stretch-agent"]) == 0
    out = capsys.readouterr()
    assert "warning" not in out.err and '"sudo"' in out.out
