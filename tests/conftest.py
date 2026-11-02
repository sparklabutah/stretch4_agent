import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from stretch4_agent.backends.mock import MockBackend
from stretch4_agent.safety import SafetyConfig
from stretch4_agent.server import create_app
from stretch4_agent.service import AgentService, ServiceConfig

TOKEN = "test-token-not-a-secret"


class PatternBackend(MockBackend):
    """Gripper depth with a known value at every pixel: 0.5 + 0.001 x + 0.002 y metres."""

    def _render(self, name, s):
        rgb, depth = super()._render(name, s)
        h, w = depth.shape
        yy, xx = np.mgrid[0:h, 0:w]
        return rgb, (0.5 + 0.001 * xx + 0.002 * yy).astype(np.float32)


def make_service(tmp_path: Path, backend=None, safety: SafetyConfig | None = None, **kw) -> AgentService:
    backend = backend or MockBackend(size_scale=0.25)
    cfg = ServiceConfig(safety=safety or SafetyConfig(), runs_dir=tmp_path / "runs", **kw)
    return AgentService(backend, cfg)


@pytest.fixture
def svc(tmp_path):
    return make_service(tmp_path, image_max_dim=80)


@pytest.fixture
def started(svc):
    svc.start_episode("test task")
    return svc


@pytest.fixture
def live(tmp_path):
    """Run the real ASGI app under uvicorn on a free port. Yields (service, url)."""
    service = make_service(tmp_path, image_max_dim=96)
    app = create_app(service, TOKEN)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.02)
    yield service, f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    th.join(timeout=5)
