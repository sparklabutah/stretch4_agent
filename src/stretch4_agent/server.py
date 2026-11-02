"""MCP server (Streamable HTTP only) with bearer-token auth and loopback-by-default binding.

No stdio transport on purpose: RobotClient, MuJoCo child processes and the camera libraries
print to stdout, which would corrupt an MCP stdio stream.
"""
from __future__ import annotations

import functools
import hmac
import ipaddress
import os
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .schema import JOINTS, MotionRefused
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

    @mcp.tool(structured_output=False)
    @_refuse
    def move(reasoning: str,
             drive_m: float = 0.0, slide_m: float = 0.0, turn_deg: float = 0.0,
             lift_m: float = 0.0, arm_m: float = 0.0,
             wrist_yaw_deg: float = 0.0, wrist_pitch_deg: float = 0.0, wrist_roll_deg: float = 0.0,
             gripper_pct: float = 0.0, return_observation: bool = True) -> list:
        """One relative move; all values are deltas from now. drive_m forward, slide_m sideways (left), turn_deg counter-clockwise, lift_m up, arm_m extend, wrist_*_deg, gripper_pct (positive opens). Blocks until done. Any invalid value rejects the whole command and nothing moves. Returns per-joint results (completed / stopped_short / timeout / rejected) and by default a fresh observation."""
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

    async def log_schema_rejections(ctx, call_next):
        """Calls rejected by argument validation never reach the service; log them here."""
        result = await call_next(ctx)
        if ctx.method == "tools/call" and ctx.params:
            blocks = getattr(result, "content", None) or (result or {}).get("content") or []
            text = " ".join(str(getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else "")) for b in blocks)
            if "validation error" in text:
                service._log(tool=ctx.params.get("name"), args=ctx.params.get("arguments"),
                             error=text[:600], stage="schema")
        return result
    mcp.middleware.append(log_schema_rejections)

    # Pydantic ignores unknown keys by default, which would silently drop a typo such as
    # `arm_cm` and move something else. Forbid extras so the whole call fails instead.
    for tool in mcp._tool_manager.list_tools():
        model = tool.fn_metadata.arg_model
        model.model_config["extra"] = "forbid"
        model.model_rebuild(force=True)
    return mcp


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
