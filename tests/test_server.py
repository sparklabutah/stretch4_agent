import base64
import io
import json

import httpx
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from PIL import Image as PILImage
from pydantic import ValidationError

from conftest import TOKEN
from stretch4_agent import cli
from stretch4_agent.schema import PARAM, UNIT_WORD, UNITS
from stretch4_agent.server import ConfigError, build_mcp, check_bind, create_app, require_token, token_from_env

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "0"}}}
HDRS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


# ---- startup refusals ---------------------------------------------------------

@pytest.mark.parametrize("tok", [None, "", "   "])
def test_require_token_refuses_missing_or_blank(tok):
    with pytest.raises(ConfigError, match="STRETCH4_AGENT_TOKEN"):
        require_token(tok)


def test_create_app_refuses_without_token(svc):
    with pytest.raises(ConfigError):
        create_app(svc, "")


def test_serve_cli_refuses_to_start_without_token(monkeypatch, capsys):
    monkeypatch.delenv("STRETCH4_AGENT_TOKEN", raising=False)
    monkeypatch.delenv("STRETCH4_AGENT_TOKEN_FILE", raising=False)
    assert cli.main(["serve", "--backend", "mock"]) == 2
    assert "refusing to start without a bearer token" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.100", "::", "example.com"])
def test_non_loopback_bind_refused_without_flag(host):
    with pytest.raises(ConfigError, match="--allow-remote-bind"):
        check_bind(host, allow_remote=False)
    check_bind(host, allow_remote=True)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
def test_loopback_bind_allowed(host):
    check_bind(host, allow_remote=False)


def test_serve_cli_refuses_remote_bind_without_flag(monkeypatch, capsys):
    monkeypatch.setenv("STRETCH4_AGENT_TOKEN", "x")
    assert cli.main(["serve", "--backend", "mock", "--host", "0.0.0.0"]) == 2
    assert "not a loopback address" in capsys.readouterr().err


def test_token_file_fallback(monkeypatch, tmp_path):
    f = tmp_path / "t.token"
    f.write_text("abc\n")
    monkeypatch.delenv("STRETCH4_AGENT_TOKEN", raising=False)
    monkeypatch.setenv("STRETCH4_AGENT_TOKEN_FILE", str(f))
    assert token_from_env().strip() == "abc"


# ---- bearer token over real HTTP -------------------------------------------------

def test_missing_and_wrong_token_are_401_and_correct_passes(live):
    _, url = live
    r = httpx.post(url, json=INIT, headers=HDRS)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    for bad in ("Bearer wrong", "Bearer", "Basic " + TOKEN, TOKEN, f"Bearer {TOKEN}x", f"Bearer {TOKEN[:-1]}"):
        assert httpx.post(url, json=INIT, headers={**HDRS, "Authorization": bad}).status_code == 401, bad
    ok = httpx.post(url, json=INIT, headers={**HDRS, "Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 200


def test_dns_rebinding_protection_stays_on(live):
    _, url = live
    r = httpx.post(url, json=INIT, headers={**HDRS, "Authorization": f"Bearer {TOKEN}", "Host": "evil.example.com"})
    assert r.status_code in (400, 403, 421)


# ---- end to end with a real MCP client over HTTP -----------------------------------

def client_for(url, token=TOKEN):
    http = create_mcp_http_client(headers={"Authorization": f"Bearer {token}"})
    return Client(streamable_http_client(url, http_client=http))


def jpeg_size(content) -> tuple[int, int]:
    img = PILImage.open(io.BytesIO(base64.b64decode(content.data)))
    img.load()
    return img.size


async def test_end_to_end_episode_with_real_images(live):
    service, url = live
    async with client_for(url) as c:
        names = {t.name for t in (await c.list_tools()).tools}
        assert names == {"start_episode", "observe", "move", "get_gripper_depth", "emergency_stop", "finish_episode"}

        # motion before start_episode is an error result, not a crash
        r = await c.call_tool("move", {"reasoning": "x", "arm_m": 0.05})
        assert r.is_error and "start_episode" in r.content[0].text

        r = await c.call_tool("start_episode", {"task": "touch the red box"})
        assert not r.is_error
        assert "Operating a Stretch 4" in r.content[0].text
        info = json.loads(r.content[1].text)
        assert set(info["cameras"]) == {"head_left", "head_right", "head_center", "gripper"}

        r = await c.call_tool("observe", {})
        assert not r.is_error
        head = json.loads(r.content[0].text)
        images = [b for b in r.content if b.type == "image"]
        assert len(images) == 4 and all(b.mime_type == "image/jpeg" for b in images)
        sizes = [jpeg_size(b) for b in images]
        assert all(0 < max(s) <= 96 for s in sizes)           # image_max_dim=96 in the fixture
        assert head["cameras"]["gripper"]["image_size"] == list(sizes[3])
        frame_id = head["frame_id"]

        r = await c.call_tool("get_gripper_depth", {"points": [[10, 10], [20, 30]], "frame_id": frame_id})
        assert not r.is_error and len(json.loads(r.content[0].text)["depth_cm"]) == 2

        r = await c.call_tool("move", {"reasoning": "extend a bit", "arm_m": 0.05})
        assert not r.is_error
        assert json.loads(r.content[0].text)["results"]["arm"]["status"] == "completed"
        new = [b for b in r.content if b.type == "image"]
        assert len(new) == 4                                   # fresh observation came back with the move

        r = await c.call_tool("get_gripper_depth", {"points": [[1, 1]], "frame_id": frame_id})
        assert r.is_error and "stale" in r.content[0].text

        r = await c.call_tool("move", {"reasoning": "typo", "arm_cm": 5})
        assert r.is_error and "arm_cm" in r.content[0].text
        r = await c.call_tool("move", {"reasoning": "too far", "arm_m": 1.0})
        assert r.is_error and "per-move limit" in r.content[0].text

        r = await c.call_tool("finish_episode", {"outcome": "success", "evidence": "arm extended in the last image"})
        assert not r.is_error
        r = await c.call_tool("move", {"reasoning": "after", "arm_m": 0.01})
        assert r.is_error and "start_episode" in r.content[0].text

    ev = [json.loads(l) for l in (service.ep.dir / "events.jsonl").read_text().splitlines()]
    assert [e["tool"] for e in ev] == ["start_episode", "observe", "get_gripper_depth", "move",
                                      "get_gripper_depth", "move", "move", "finish_episode", "move"]
    assert ev[5]["stage"] == "schema" and "arm_cm" in ev[5]["error"]
    assert len(list((service.ep.dir / "images").glob("*.jpg"))) == 8


async def test_emergency_stop_over_http_latches(live):
    service, url = live
    async with client_for(url) as c:
        await c.call_tool("start_episode", {"task": "t"})
        r = await c.call_tool("emergency_stop", {"reason": "test"})
        assert not r.is_error and json.loads(r.content[0].text)["runstopped"] is True
        r = await c.call_tool("move", {"reasoning": "x", "arm_m": 0.01})
        assert r.is_error and "runstopped" in r.content[0].text


async def test_client_with_wrong_token_cannot_connect(live):
    _, url = live
    with pytest.raises(Exception):
        async with client_for(url, "nope") as c:
            await c.list_tools()


# ---- move arguments: names, units, strictness (HANDOFF open issues 1 and 2) ----------

async def test_move_schema_names_the_unit_of_every_key_and_forbids_extras(live):
    _, url = live
    async with client_for(url) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    for t in tools.values():
        assert t.input_schema.get("additionalProperties") is False, t.name
    props = tools["move"].input_schema["properties"]
    assert set(props) == {"reasoning", "return_observation", *PARAM.values()}
    for joint, param in PARAM.items():
        assert param.endswith("_" + UNITS[joint])
        assert UNIT_WORD[UNITS[joint]] in props[param]["description"], param
    for param in ("turn_deg", "wrist_yaw_deg", "wrist_pitch_deg", "wrist_roll_deg"):
        assert "not radians" in props[param]["description"]


async def test_subset_of_declared_keys_is_accepted_and_unknown_key_rejects_whole_call(live):
    """Codex sent {"arm": 0.05, "reasoning": ...} and was rejected. `arm` is not a declared key
    (`arm_m` is); any subset of declared keys must pass, and one unknown key must stop everything."""
    service, url = live
    async with client_for(url) as c:
        await c.call_tool("start_episode", {"task": "t"})
        state = lambda: service.backend.get_state()

        # the exact call from the Codex run: rejected, with the right name in the message
        before = state()
        r = await c.call_tool("move", {"arm": 0.05, "reasoning": "extend"})
        assert r.is_error
        msg = r.content[0].text
        assert "'arm' is not an argument; did you mean 'arm_m'? Its value is in metres." in msg
        assert "nothing was done" in msg and state() == before

        # wrong unit suffix: the hint names the unit so the agent converts the value
        r = await c.call_tool("move", {"arm_cm": 5, "reasoning": "x"})
        assert r.is_error and "did you mean 'arm_m'? Its value is in metres" in r.content[0].text
        r = await c.call_tool("move", {"turn_rad": 0.1, "reasoning": "x"})
        assert r.is_error and "did you mean 'turn_deg'? Its value is in degrees" in r.content[0].text

        # one valid key plus one unknown key: the valid one must not move either
        r = await c.call_tool("move", {"arm_m": 0.05, "lift": 0.05, "reasoning": "mixed"})
        assert r.is_error and "'lift'" in r.content[0].text and state() == before

        # subsets of declared keys pass: one key, then two keys plus a flag
        r = await c.call_tool("move", {"arm_m": 0.05, "reasoning": "one key"})
        assert not r.is_error, r.content[0].text
        assert abs(state()["arm"] - before["arm"] - 0.05) < 1e-6
        r = await c.call_tool("move", {"reasoning": "two keys", "lift_m": 0.02, "wrist_yaw_deg": 5.0,
                                       "return_observation": False})
        assert not r.is_error, r.content[0].text
        assert set(json.loads(r.content[0].text)["results"]) == {"lift", "wrist_yaw"}

        # all nine keys, as Codex resent them, also pass
        nine = {p: 0.0 for p in PARAM.values()} | {"arm_m": -0.05}
        r = await c.call_tool("move", {"reasoning": "nine keys", **nine, "return_observation": False})
        assert not r.is_error, r.content[0].text

        # other tools are strict too
        r = await c.call_tool("observe", {"camera": ["gripper"]})
        assert r.is_error and "did you mean 'cameras'" in r.content[0].text

    ev = [json.loads(l) for l in (service.ep.dir / "events.jsonl").read_text().splitlines()]
    schema = [e for e in ev if e.get("stage") == "schema"]
    assert [sorted(e["args"]) for e in schema] == [["arm", "reasoning"], ["arm_cm", "reasoning"],
                                                    ["reasoning", "turn_rad"], ["arm_m", "lift", "reasoning"],
                                                    ["camera"]]
    assert service.ep.counters["moves"] == 3


def test_argument_models_forbid_extras_even_without_the_middleware(svc):
    mcp = build_mcp(svc)
    model = mcp._tool_manager.get_tool("move").fn_metadata.arg_model
    with pytest.raises(ValidationError, match="extra_forbidden"):
        model.model_validate({"reasoning": "x", "arm": 0.05})
    assert model.model_validate({"reasoning": "x", "arm_m": 0.05}).arm_m == 0.05


def test_server_refuses_to_start_if_extras_stop_being_forbidden(svc, monkeypatch):
    """If an SDK change made the private-attribute patch a no-op, the server must not start."""
    from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase
    monkeypatch.setattr(ArgModelBase, "model_rebuild", classmethod(lambda cls, **kw: None))
    with pytest.raises(RuntimeError, match="would accept unknown arguments"):
        build_mcp(svc)
