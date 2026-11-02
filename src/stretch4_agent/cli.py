"""stretch4-agent serve | doctor | token | run"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

from .safety import DEFAULT_PER_MOVE, SafetyConfig
from .server import ConfigError, TOKEN_ENV, check_bind, require_token, serve, token_from_env


def _build_service(args):
    from .backends import make_backend
    from .service import AgentService, ServiceConfig
    kw = {}
    if args.backend == "mock":
        kw["latency_s"] = args.mock_latency
    if args.backend == "sim":
        kw["headless"] = not args.viewer
    per_move = dict(DEFAULT_PER_MOVE)
    for key in per_move:
        v = getattr(args, f"max_{key}", None)
        if v is not None:
            per_move[key] = v
    safety = SafetyConfig(per_move=per_move, max_moves=args.max_moves, max_seconds=args.max_minutes * 60,
                          move_timeout=args.move_timeout, read_only=args.read_only)
    cfg = ServiceConfig(safety=safety, runs_dir=Path(args.runs_dir), image_max_dim=args.image_max_dim)
    backend = make_backend(args.backend, **kw)
    return AgentService(backend, cfg), backend


def cmd_serve(args) -> int:
    try:
        token = require_token(token_from_env())
        check_bind(args.host, args.allow_remote_bind)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    service, backend = _build_service(args)
    backend.start()
    print(f"stretch4-agent: backend={args.backend} read_only={args.read_only} "
          f"http://{args.host}:{args.port}/mcp", file=sys.stderr, flush=True)
    serve(service, args.host, args.port, token, args.allow_remote_bind)
    return 0


def cmd_doctor(args) -> int:
    args.read_only = True
    service, backend = _build_service(args)
    backend.start()
    try:
        print(json.dumps(backend.doctor(), indent=2, default=str))
    finally:
        backend.close()
    return 0


def cmd_token(args) -> int:
    tok = secrets.token_urlsafe(32)
    if args.file:
        p = Path(args.file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(tok)
        print(f"wrote token to {p} (mode 600). export {TOKEN_ENV}=$(cat {p})")
    else:
        print(tok)
    return 0


def cmd_run(args) -> int:
    from .runner import build_command, run_agent
    try:
        token = require_token(token_from_env())
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    spec = build_command(args.agent, args.task, url=args.url, model=args.model, effort=args.effort,
                         max_turns=args.max_turns, project_dir=Path(args.project_dir))
    if args.dry_run:
        print(json.dumps({"argv": spec.argv, "extra_env": {k: v for k, v in spec.env.items()}}, indent=2))
        return 0
    return run_agent(spec, token)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stretch4-agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def backend_args(p):
        p.add_argument("--backend", choices=["mock", "sim", "stretch"], default="mock")
        p.add_argument("--viewer", action="store_true", help="sim only: open the MuJoCo viewer (macOS needs mjpython)")
        p.add_argument("--mock-latency", type=float, default=0.0, help="mock only: seconds each move takes")
        p.add_argument("--runs-dir", default="runs")
        p.add_argument("--image-max-dim", type=int, default=768)
        p.add_argument("--read-only", action="store_true", help="run everything except motion")
        p.add_argument("--max-moves", type=int, default=150)
        p.add_argument("--max-minutes", type=float, default=30.0)
        p.add_argument("--move-timeout", type=float, default=15.0)
        for k, v in DEFAULT_PER_MOVE.items():
            p.add_argument(f"--max-{k.replace('_', '-')}", dest=f"max_{k}", type=float, default=None,
                           help=f"per-move cap (default {v:g})")

    s = sub.add_parser("serve", help="run the MCP server")
    backend_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--allow-remote-bind", action="store_true")
    s.set_defaults(fn=cmd_serve)

    d = sub.add_parser("doctor", help="print status, limits and camera info; never moves")
    backend_args(d)
    d.set_defaults(fn=cmd_doctor)

    t = sub.add_parser("token", help="generate a bearer token")
    t.add_argument("--file", help="write it here with mode 600 (e.g. .tokens/agent.token)")
    t.set_defaults(fn=cmd_token)

    r = sub.add_parser("run", help="run a headless Claude Code or Codex session against a running server")
    r.add_argument("--agent", choices=["claude", "codex"], required=True)
    r.add_argument("--task", required=True)
    r.add_argument("--url", default=os.environ.get("STRETCH4_AGENT_URL", "http://127.0.0.1:8765/mcp"))
    r.add_argument("--model")
    r.add_argument("--effort")
    r.add_argument("--max-turns", type=int, default=40)
    r.add_argument("--project-dir", default=".")
    r.add_argument("--dry-run", action="store_true", help="print the command instead of running it")
    r.set_defaults(fn=cmd_run)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    # The sim backend uses multiprocessing 'spawn'; entry points need this guard.
    sys.exit(main())
