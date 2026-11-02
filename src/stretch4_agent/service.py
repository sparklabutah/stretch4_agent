"""The tool logic, independent of MCP. Methods are synchronous and run in worker threads."""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from mcp.server.mcpserver import Image

from .backends.base import Backend
from .episode import DepthFrame, Episode, EventLog
from .imaging import encode_jpeg
from .safety import SafetyConfig, validate_command
from .schema import COMPLETED, PARAM, STATE_KEY, UNIT_WORD, FrameTimeout, JOINTS, MotionRefused, UNITS

MAX_DEPTH_POINTS = 64
MAX_DEPTH_CALLS = 32
OUTCOMES = ("success", "failure", "blocked")
FRAME_WAIT_S = 5.0


def _r(x: float, n: int = 3) -> float:
    return round(float(x), n)


def _j(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


# Absolute positions in `state` and `joint_limits` use the bare joint names.
STATE_UNITS = {**{j: UNIT_WORD[UNITS[j]] for j in STATE_KEY},
               "base_x": "metres", "base_y": "metres", "base_theta": "degrees"}


def operating_prompt() -> str:
    return resources.files("stretch4_agent").joinpath("prompts/operating_prompt.md").read_text()


@dataclass
class ServiceConfig:
    safety: SafetyConfig
    runs_dir: Path
    image_max_dim: int = 768
    jpeg_quality: int = 80


class AgentService:
    def __init__(self, backend: Backend, cfg: ServiceConfig, clock: Callable[[], float] = time.monotonic):
        self.backend, self.cfg, self.clock = backend, cfg, clock
        self.ep: Episode | None = None
        self._move_lock = threading.Lock()
        self._server_log: EventLog | None = None
        self._last_result_t: float | None = None
        self._boot_ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self._tl = threading.local()  # per-call details attached to the event being logged

    # ---------------------------------------------------------------- logging
    def _log(self, **event: Any) -> None:
        if self.ep is not None:
            self.ep.log.write(**event)
            return
        if self._server_log is None:  # calls made before any episode (rejections, probes)
            self._server_log = EventLog(self.cfg.runs_dir / f"_server_{self._boot_ts}.jsonl")
        self._server_log.write(**event)

    def logged(self, tool: str, args: dict[str, Any], fn: Callable[[], list]) -> list:
        """Run a tool body, logging args, outcome and the model-wait vs execution split."""
        t0 = time.monotonic()
        wait = None if self._last_result_t is None else t0 - self._last_result_t
        err = None
        try:
            return fn()
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            t1 = time.monotonic()
            self._last_result_t = t1
            if self.ep is not None:
                if wait is not None and tool != "start_episode":
                    self.ep.model_wait_s += wait
                self.ep.exec_s += t1 - t0
            extra, self._tl.extra = getattr(self._tl, "extra", {}), {}
            self._log(tool=tool, args=args, error=err, model_wait_s=None if wait is None else _r(wait),
                      exec_s=_r(t1 - t0), **extra)
            if self.ep is not None:
                self.ep.write_summary()

    def _note(self, **kw: Any) -> None:
        """Attach result details to the event being logged."""
        self._tl.extra = {**getattr(self._tl, "extra", {}), **kw}

    # --------------------------------------------------------------- helpers
    def _state(self) -> dict[str, float]:
        return {k: _r(v) for k, v in self.backend.get_state().items()}

    def _require_active(self) -> Episode:
        if self.ep is None or not self.ep.active:
            raise MotionRefused("no active episode; call start_episode first")
        return self.ep

    def _limits_doc(self) -> dict:
        return {k: [_r(lo), _r(hi)] for k, (lo, hi) in self.backend.get_limits().items()}

    # ------------------------------------------------------------ start / end
    def start_episode(self, task: str) -> list:
        def body() -> list:
            if not task.strip():
                raise MotionRefused("task must be a non-empty string")
            if self.ep is not None and self.ep.active:
                self.ep.active, self.ep.outcome = False, "abandoned"
                self._log(event="episode_abandoned_by_new_start")
                self.ep.write_summary()
            self.ep = Episode.open(self.cfg.runs_dir, task, self.cfg.safety, self.backend.name, self.clock)
            self._note(task=task, episode=self.ep.id)
            ready = self.backend.readiness()
            info = {
                "episode": self.ep.id,
                "read_only": self.cfg.safety.read_only,
                "ready_for_motion": ready.ok and not self.cfg.safety.read_only,
                "readiness_problems": ready.reasons(),
                # Keyed by the `move` argument name, so no other spelling is ever shown as a key.
                "move_arguments": {PARAM[j]: {"unit": UNIT_WORD[UNITS[j]], "max_abs_per_move": self.cfg.safety.per_move[j],
                                              "reported_as": j} for j in JOINTS},
                "joint_limits": self._limits_doc(),
                "state_units": STATE_UNITS,
                "budgets": self.ep.budget.to_dict(),
                "move_timeout_s": self.cfg.safety.move_timeout,
                "cameras": self.backend.cameras,
                "image_max_dim": self.cfg.image_max_dim,
                "state": self._state(),
            }
            return [operating_prompt(), _j(info)]
        return self.logged("start_episode", {"task": task}, body)

    def finish_episode(self, outcome: str, evidence: str) -> list:
        def body() -> list:
            ep = self._require_active()
            if outcome not in OUTCOMES:
                raise MotionRefused(f"outcome must be one of {OUTCOMES}")
            if not evidence.strip():
                raise MotionRefused("evidence must describe what the latest images show")
            if not ep.fresh_observation:
                raise MotionRefused("take an observation after your last move before finishing; "
                                    "the claim must rest on current images")
            ep.active, ep.outcome, ep.evidence = False, outcome, evidence
            self._note(outcome=outcome, evidence=evidence)
            return [_j({"finished": True, "outcome_recorded_as_your_claim": outcome,
                        "note": "motion is rejected until the next start_episode",
                        "counters": ep.counters})]
        return self.logged("finish_episode", {"outcome": outcome, "evidence": evidence}, body)

    # --------------------------------------------------------------- observe
    def observe(self, cameras: list[str] | None = None) -> list:
        def body() -> list:
            ep = self._require_active()
            return self._observe(ep, cameras)
        return self.logged("observe", {"cameras": cameras}, body)

    def _observe(self, ep: Episode, cameras: list[str] | None) -> list:
        names = list(cameras) if cameras else list(self.backend.cameras)
        bad = [c for c in names if c not in self.backend.cameras]
        if bad:
            raise MotionRefused(f"unknown camera(s) {bad}; available: {list(self.backend.cameras)}")
        try:
            frames = self.backend.get_frames(ep.last_motion_end, FRAME_WAIT_S, names)
        except FrameTimeout as e:
            raise MotionRefused(f"no fresh camera frame: {e}") from e
        fid = ep.next_frame_id()
        content: list = []
        cams: dict[str, Any] = {}
        ep.depth_frame = None
        for name in names:
            fr = frames[name]
            enc = encode_jpeg(fr.rgb, self.cfg.image_max_dim, self.cfg.jpeg_quality)
            saved = ep.save_image(fid, name, enc.jpeg)
            cams[name] = {"image_size": [enc.width, enc.height], "full_size": [enc.full_width, enc.full_height],
                          "age_s": _r(time.time() - fr.timestamp, 2), "file": saved}
            content += [f"image: {name} ({enc.width}x{enc.height})", Image(data=enc.jpeg, format="jpeg")]
            if name == "gripper" and fr.depth is not None:
                ep.depth_frame = DepthFrame(fid, fr.depth, enc)
        ep.current_frame_id = fid
        ep.fresh_observation = True
        ep.counters["observations"] += 1
        ready = self.backend.readiness()
        head = {"frame_id": fid, "state": self._state(), "cameras": cams,
                "depth_available": ep.depth_frame is not None,
                "budget": ep.budget.to_dict(), "runstopped": ready.runstopped}
        self._note(frame_id=fid, cameras={k: v["file"] for k, v in cams.items()})
        return [_j(head)] + content

    # ------------------------------------------------------------------ move
    def move(self, reasoning: str, command: dict[str, Any], return_observation: bool = True) -> list:
        args = {"reasoning": reasoning, **command, "return_observation": return_observation}

        def body() -> list:
            ep = self._require_active()
            try:
                deltas = self._check_motion_allowed(ep, command)
            except MotionRefused as e:
                ep.counters["moves_rejected"] += 1
                self._note(rejected=str(e))
                raise
            if not self._move_lock.acquire(blocking=False):
                raise MotionRefused("another move is still running")
            try:
                before = self._state()
                t_exec = time.monotonic()
                try:
                    results = self.backend.execute(deltas, self.cfg.safety.move_timeout)
                except Exception as e:  # backend bug or lost link: stop, then tell the agent
                    self.backend.halt(f"backend error: {e}")
                    ep.counters["moves"] += 1
                    ep.budget.moves += 1
                    self._after_motion(ep)
                    raise MotionRefused(f"backend error during move, robot halted: {type(e).__name__}: {e}") from e
            finally:
                self._move_lock.release()
            ep.counters["moves"] += 1
            ep.budget.moves += 1
            self._after_motion(ep)
            exec_s = time.monotonic() - t_exec
            doc = {"results": {k: results[k].to_dict() for k in JOINTS if k in results},
                   "all_completed": all(r.status == COMPLETED for r in results.values()),
                   "move_seconds": _r(exec_s, 2), "state_before": before, "state_after": self._state(),
                   "budget": ep.budget.to_dict()}
            self._note(results=doc["results"], move_seconds=doc["move_seconds"])
            out: list = [_j(doc)]
            if return_observation:
                try:
                    out += self._observe(ep, None)
                except MotionRefused as e:
                    out.append(_j({"observation_error": str(e)}))
            return out
        return self.logged("move", args, body)

    def _check_motion_allowed(self, ep: Episode, command: dict[str, Any]) -> dict[str, float]:
        if ep.read_only:
            raise MotionRefused("server is in read-only mode: motion is disabled")
        ready = self.backend.readiness()
        if not ready.ok:
            raise MotionRefused("robot not ready: " + "; ".join(ready.reasons()))
        ep.budget.check()
        return validate_command(command, self.backend.get_state(), self.backend.get_limits(), self.cfg.safety)

    def _after_motion(self, ep: Episode) -> None:
        ep.last_motion_end = time.time()
        ep.fresh_observation = False
        ep.current_frame_id = None   # the agent's last image no longer matches the world
        ep.depth_frame = None
        ep.depth_calls = 0

    # ----------------------------------------------------------------- depth
    def get_gripper_depth(self, points: list, frame_id: int) -> list:
        def body() -> list:
            ep = self._require_active()
            if not isinstance(points, list) or not (1 <= len(points) <= MAX_DEPTH_POINTS):
                raise MotionRefused(f"points must be a list of 1 to {MAX_DEPTH_POINTS} [x, y] pairs")
            clean: list[tuple[float, float]] = []
            for p in points:
                ok = (isinstance(p, (list, tuple)) and len(p) == 2 and
                      all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p))
                if not ok:
                    raise MotionRefused(f"each point must be a finite [x, y] pair, got {p!r}")
                clean.append((float(p[0]), float(p[1])))
            if ep.depth_calls >= MAX_DEPTH_CALLS:
                raise MotionRefused(f"at most {MAX_DEPTH_CALLS} depth calls between moves; move or finish")
            if ep.current_frame_id is None or frame_id != ep.current_frame_id:
                raise MotionRefused(f"stale frame_id {frame_id}: the latest observation is "
                                    f"{ep.current_frame_id if ep.current_frame_id is not None else 'none since your last move'}"
                                    "; observe again")
            df = ep.depth_frame
            if df is None or df.frame_id != frame_id:
                raise MotionRefused("that observation had no gripper image or depth; observe the gripper camera")
            out = []
            for x, y in clean:
                if not df.enc.contains(x, y):
                    raise MotionRefused(f"point ({x:g}, {y:g}) is outside the {df.enc.width}x{df.enc.height} "
                                        "gripper image")
                fx, fy = df.enc.to_full(x, y)
                d = float(df.depth[fy, fx])
                out.append(_r(d * 100.0, 1) if math.isfinite(d) and d > 0 else None)
            ep.depth_calls += 1
            ep.counters["depth_queries"] += 1
            self._note(n_points=len(clean))
            return [_j({"frame_id": frame_id, "depth_cm": out, "points": clean,
                        "depth_calls_left_before_next_move": MAX_DEPTH_CALLS - ep.depth_calls})]
        return self.logged("get_gripper_depth", {"points": points, "frame_id": frame_id}, body)

    # ------------------------------------------------------------ e-stop
    def emergency_stop(self, reason: str) -> list:
        def body() -> list:
            self.backend.halt(f"agent emergency_stop: {reason}")   # deliberately lock-free
            if self.ep is not None:
                self.ep.counters["emergency_stops"] += 1
            return [_j({"halted": True, "runstopped": self.backend.readiness().runstopped,
                        "note": "motion is refused until a person clears the stop"})]
        return self.logged("emergency_stop", {"reason": reason}, body)

    def close(self) -> None:
        if self.ep is not None and self.ep.active:
            self.ep.active, self.ep.outcome = False, "abandoned"
            self.ep.write_summary()
        self.backend.close()
