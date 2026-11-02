"""MCP server (Streamable HTTP only) with bearer-token auth and loopback-by-default binding.

No stdio transport on purpose: RobotClient, MuJoCo child processes and the camera libraries
print to stdout, which would corrupt an MCP stdio stream.
"""
from __future__ import annotations

import difflib
import functools
import hmac
import ipaddress
import os
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from pydantic import Field, ValidationError
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .schema import JOINTS, PARAM, PARAM_DOC, UNIT_WORD, UNITS, MotionRefused
from .service import AgentService

TOKEN_ENV = "STRETCH4_AGENT_TOKEN"


class ConfigError(Exception):
    pass


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_bind(host: str, allow_remote: bool) -> None:
    if not is_loopback(host) and not allow_remote:
        raise ConfigError(f"refusing to bind to {host!r}: not a loopback address. "
                          "Use an SSH tunnel to a loopback bind, or pass --allow-remote-bind "
                          "if you accept anyone on that network reaching the robot (the token is the only gate).")


def require_token(token: str | None) -> str:
    if not token or not token.strip():
        raise ConfigError(f"{TOKEN_ENV} is not set (or empty); refusing to start without a bearer token. "
                          "Generate one with `stretch4-agent token`.")
    return token.strip()


class BearerAuth:
    """Pure ASGI middleware: every HTTP request needs `Authorization: Bearer <token>`."""

    def __init__(self, app: ASGIApp, token: str):
        self.app, self._token = app, token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            auth = dict(scope["headers"]).get(b"authorization", b"")
            scheme, _, supplied = auth.partition(b" ")
            if scheme.lower() != b"bearer" or not hmac.compare_digest(supplied.strip(), self._token):
                resp = JSONResponse({"error": "missing or invalid bearer token"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
                return await resp(scope, receive, send)
        await self.app(scope, receive, send)


def _refuse(fn):
    """Turn MotionRefused into a tool error the agent can read."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except MotionRefused as e:
            raise ToolError(str(e)) from None
    return wrapper


def build_mcp(service: AgentService) -> MCPServer:
    mcp = MCPServer("stretch4-agent", instructions=(
        "Control a Hello Robot Stretch 4 with bounded relative moves. Call start_episode first: "
        "it returns the operating instructions."))

    @mcp.tool(structured_output=False)
    @_refuse
    def start_episode(task: str) -> list:
        """Begin an episode. Returns the operating prompt, limits, budgets and cameras. Motion is rejected until this is called."""
        return service.start_episode(task)

    @mcp.tool(structured_output=False)
    @_refuse
    def observe(cameras: list[str] | None = None) -> list:
        """Images from the cameras (all by default: head_left, head_right, head_center, gripper) plus joint state and a frame_id."""
        return service.observe(cameras)

    # strict=True: lax mode would turn `true` into 1.0 and "0.05" into 0.05 before the safety
    # layer sees the value. Integers are still accepted as numbers.
    @mcp.tool(structured_output=False)
    @_refuse
    def move(reasoning: Annotated[str, Field(description="One sentence: why this move.")],
             drive_m: Annotated[float, Field(strict=True, description=PARAM_DOC["drive"])] = 0.0,
             slide_m: Annotated[float, Field(strict=True, description=PARAM_DOC["slide"])] = 0.0,
             turn_deg: Annotated[float, Field(strict=True, description=PARAM_DOC["turn"])] = 0.0,
             lift_m: Annotated[float, Field(strict=True, description=PARAM_DOC["lift"])] = 0.0,
             arm_m: Annotated[float, Field(strict=True, description=PARAM_DOC["arm"])] = 0.0,
             wrist_yaw_deg: Annotated[float, Field(strict=True, description=PARAM_DOC["wrist_yaw"])] = 0.0,
             wrist_pitch_deg: Annotated[float, Field(strict=True, description=PARAM_DOC["wrist_pitch"])] = 0.0,
             wrist_roll_deg: Annotated[float, Field(strict=True, description=PARAM_DOC["wrist_roll"])] = 0.0,
             gripper_pct: Annotated[float, Field(strict=True, description=PARAM_DOC["gripper"])] = 0.0,
             return_observation: Annotated[bool, Field(strict=True, description="Return a fresh observation after the move.")] = True) -> list:
        """One relative move; every value is a delta from now, in the unit named by its suffix (_m metres, _deg degrees, _pct percent). Pass only the keys you need; omitted keys are 0. A key without its unit suffix (e.g. `arm`) or with another unit (e.g. `arm_cm`) is rejected. Blocks until done. Any invalid value or unknown key rejects the whole command and nothing moves. Returns per-joint results (completed / stopped_short / timeout / rejected) keyed by joint name (drive, arm, ...) and by default a fresh observation."""
        vals = dict(zip(JOINTS, (drive_m, slide_m, turn_deg, lift_m, arm_m, wrist_yaw_deg,
                                 wrist_pitch_deg, wrist_roll_deg, gripper_pct)))
        return service.move(reasoning, vals, return_observation)

    @mcp.tool(structured_output=False)
    @_refuse
    def get_gripper_depth(points: Annotated[list[list[float]], Field(description="[[x, y], ...] up to 64 pixel coordinates in the gripper image as shown to you")],
                          frame_id: int) -> list:
        """Depth in cm at pixels of the gripper image from your latest observation (null where there is no reading). frame_id must be the latest; a move makes it stale. At most 32 calls between moves."""
        return service.get_gripper_depth(points, frame_id)

    @mcp.tool(structured_output=False)
    @_refuse
    def emergency_stop(reason: str) -> list:
        """Halt all motion now. On the real robot this latches the runstop and only a person can clear it. Use if anything looks unsafe."""
        return service.emergency_stop(reason)

    @mcp.tool(structured_output=False)
    @_refuse
    def finish_episode(outcome: Annotated[str, Field(description="success, failure or blocked")], evidence: str) -> list:
        """End the episode. Needs an observation taken after your last move. Your outcome is recorded as your own claim, unverified. Motion is rejected afterwards."""
        return service.finish_episode(outcome, evidence)

    declared = _forbid_extra_arguments(mcp)

    async def strict_arguments(ctx, call_next):
        """Reject unknown argument names before the SDK sees them, with a message that names
        the right key. Calls rejected by argument validation never reach the service, so both
        kinds of schema rejection are logged here."""
        if ctx.method == "tools/call" and ctx.params:
            name, args = ctx.params.get("name"), ctx.params.get("arguments") or {}
            if name in declared and isinstance(args, dict):
                unknown = sorted(set(args) - declared[name])
                if unknown:
                    text = _unknown_args_message(name, unknown, declared[name])
                    service._log(tool=name, args=args, error=text, stage="schema")
                    return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
        result = await call_next(ctx)
        if ctx.method == "tools/call" and ctx.params:
            blocks = getattr(result, "content", None) or (result or {}).get("content") or []
            text = " ".join(str(getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else "")) for b in blocks)
            if "validation error" in text:
                service._log(tool=ctx.params.get("name"), args=ctx.params.get("arguments"),
                             error=text[:600], stage="schema")
        return result
    mcp.middleware.append(strict_arguments)
    return mcp


def _forbid_extra_arguments(mcp: MCPServer) -> dict[str, set[str]]:
    """Make every tool reject unknown argument names, and say so in its advertised schema.

    Pydantic ignores unknown keys by default, which would silently drop a typo such as
    `arm_cm` and move the rest. The SDK has no public switch for this, so the argument models
    are edited through the private `_tool_manager`; the self-check below makes the server
    refuse to start if an SDK change stops that from working. Returns the declared names.
    """
    declared: dict[str, set[str]] = {}
    for tool in mcp._tool_manager.list_tools():
        model = tool.fn_metadata.arg_model
        model.model_config["extra"] = "forbid"
        model.model_rebuild(force=True)
        tool.parameters = model.model_json_schema(by_alias=True)   # now has additionalProperties: false
        probe = "__stretch4_agent_probe__"
        try:
            model.model_validate({probe: 0})
            rejected = False
        except ValidationError as e:
            rejected = any(err["type"] == "extra_forbidden" for err in e.errors())
        if not rejected or tool.parameters.get("additionalProperties") is not False:
            raise RuntimeError(f"tool {tool.name!r} would accept unknown arguments; the mcp SDK changed. "
                               "Refusing to start.")
        declared[tool.name] = set(tool.parameters.get("properties", {}))
    return declared


def _suggest(key: str, declared: set[str]) -> str | None:
    k = key.strip().lower()
    for joint in sorted(JOINTS, key=len, reverse=True):     # `arm`, `arm_cm`, `turn_rad`, ...
        if (k == joint or k.startswith(joint + "_")) and PARAM[joint] in declared:
            return PARAM[joint]
    close = difflib.get_close_matches(k, sorted(declared), n=1, cutoff=0.7)
    return close[0] if close else None


def _unknown_args_message(tool: str, unknown: list[str], declared: set[str]) -> str:
    unit_of = {PARAM[j]: UNIT_WORD[UNITS[j]] for j in JOINTS}
    parts = []
    for key in unknown:
        hint = _suggest(key, declared)
        if hint in unit_of:
            parts.append(f"{key!r} is not an argument; did you mean {hint!r}? Its value is in {unit_of[hint]}.")
        elif hint:
            parts.append(f"{key!r} is not an argument; did you mean {hint!r}?")
        else:
            parts.append(f"{key!r} is not an argument.")
    order = [p for p in ("reasoning", *PARAM.values()) if p in declared] + sorted(declared - {"reasoning", *PARAM.values()})
    return (f"{tool} rejected, nothing was done: " + " ".join(parts) +
            f" Unknown argument names reject the whole call. Accepted: {', '.join(order)}.")


def create_app(service: AgentService, token: str, host: str = "127.0.0.1") -> ASGIApp:
    """ASGI app: /mcp behind bearer auth. DNS-rebinding protection stays on for loopback hosts."""
    token = require_token(token)
    mcp = build_mcp(service)
    return BearerAuth(mcp.streamable_http_app(host=host), token)


def serve(service: AgentService, host: str, port: int, token: str | None, allow_remote: bool = False) -> None:
    import uvicorn
    token = require_token(token)
    check_bind(host, allow_remote)
    app = create_app(service, token, host)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        service.close()


def token_from_env() -> str | None:
    t = os.environ.get(TOKEN_ENV)
    if t:
        return t
    f = os.environ.get(TOKEN_ENV + "_FILE")
    if f and os.path.exists(f):
        return open(f).read()
    return None
