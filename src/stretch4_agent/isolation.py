"""Check that the user an agent runs as cannot reach the robot except through this MCP server.

Codex keeps a shell (`exec`) that no config key removes. On the robot's own computer that shell
could otherwise drive the robot directly and skip every limit in `safety.py`. stretch4_body
clients reach `stretch_body_server` three ways, all checked here:

- IPC sockets in /tmp/stretch_zmq (core/client_server.py), a 2770 directory owned by group
  `users`: the agent user must not be in `users`.
- TCP 23114-23116, bound on all interfaces: blocked for the agent user by a firewall rule
  (scripts/nuc/setup_agent_user.sh rejects with icmp-admin-prohibited, so a block is told apart
  from "nothing listening").
- The motor and camera USB devices (udev: group plugdev 0664; serial ports in dialout).

Run it AS THE AGENT USER, with stretch_body_server and `stretch4-agent serve` running:

    sudo -u stretch-agent -H python3 /srv/stretch4-agent/check_isolation.py

Standard library only, so it runs without access to this project's virtualenv. Exit code 0
only if every check passes; anything unknown counts as a failure.
"""
from __future__ import annotations

import argparse
import errno
import glob
import grp
import os
import socket
import sys
from dataclasses import dataclass

ZMQ_DIR = "/tmp/stretch_zmq"
ZMQ_SOCKETS = ("port_admin", "port_command", "port_status")
ZMQ_TCP_PORTS = (23114, 23115, 23116)
ROBOT_GROUPS = ("users", "dialout", "plugdev")
DEVICE_GLOBS = ("/dev/hello-*", "/dev/ttyACM*", "/dev/ttyUSB*", "/dev/bus/usb/*/*")
PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"


@dataclass
class Result:
    name: str
    status: str
    detail: str


def check_groups(groups: tuple[str, ...] = ROBOT_GROUPS) -> Result:
    mine = set(os.getgroups()) | {os.getgid()}
    member = []
    for g in groups:
        try:
            if grp.getgrnam(g).gr_gid in mine:
                member.append(g)
        except KeyError:
            pass
    if member:
        return Result("groups", FAIL, f"member of {', '.join(member)}: gives access to the robot's sockets or devices")
    return Result("groups", PASS, f"not in {', '.join(groups)}")


def check_ipc(zmq_dir: str = ZMQ_DIR) -> Result:
    if not os.path.isdir(zmq_dir):
        return Result("zmq ipc", UNKNOWN, f"{zmq_dir} does not exist: is stretch_body_server running?")
    if os.access(zmq_dir, os.X_OK):
        open_ = [s for s in ZMQ_SOCKETS if os.access(os.path.join(zmq_dir, s), os.W_OK)]
        if open_:
            return Result("zmq ipc", FAIL, f"can write {', '.join(open_)} in {zmq_dir}")
        return Result("zmq ipc", FAIL, f"can enter {zmq_dir}; it should be closed to this user")
    return Result("zmq ipc", PASS, f"cannot enter {zmq_dir}")


def probe_tcp(host: str, port: int, timeout: float = 1.0) -> str:
    """'open', 'blocked' (firewall reject or drop) or 'closed' (nothing listening)."""
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(fam, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return "open"
        except socket.timeout:
            return "blocked"
        except OSError as e:
            if e.errno == errno.ECONNREFUSED:
                return "closed"
            if e.errno in (errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EACCES, errno.EPERM):
                return "blocked"
            return f"error {e.errno}"


def local_addresses() -> list[str]:
    addrs = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except OSError:
        pass
    try:   # the address used for outbound traffic (Wi-Fi), without sending anything
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(addrs)


def check_tcp(hosts: list[str] | None = None, ports: tuple[int, ...] = ZMQ_TCP_PORTS) -> Result:
    hosts = hosts or local_addresses()
    seen = {f"{h}:{p}": probe_tcp(h, p) for h in hosts for p in ports}
    open_ = [k for k, v in seen.items() if v == "open"]
    if open_:
        return Result("zmq tcp", FAIL, f"can connect to {', '.join(open_)}")
    other = [f"{k} {v}" for k, v in seen.items() if v != "blocked"]
    if other:
        return Result("zmq tcp", UNKNOWN, f"not shown to be blocked: {', '.join(other)} "
                      "(run with stretch_body_server up, after setup_agent_user.sh)")
    return Result("zmq tcp", PASS, f"blocked on {', '.join(hosts)} ports {ports[0]}-{ports[-1]}")


def check_devices(globs: tuple[str, ...] = DEVICE_GLOBS) -> Result:
    found = sorted({p for g in globs for p in glob.glob(g)})
    writable = [p for p in found if os.access(p, os.W_OK)]
    if writable:
        shown = ", ".join(writable[:6]) + (" ..." if len(writable) > 6 else "")
        return Result("devices", FAIL, f"can write {shown}")
    if not found:
        return Result("devices", UNKNOWN, "no robot devices found; run this on the robot")
    return Result("devices", PASS, f"none of {len(found)} USB/serial devices is writable")


def check_mcp(port: int = 8765) -> Result:
    st = probe_tcp("127.0.0.1", port)
    if st == "open":
        return Result("mcp server", PASS, f"127.0.0.1:{port} reachable")
    return Result("mcp server", FAIL, f"127.0.0.1:{port} is {st}: start `stretch4-agent serve` first, "
                  "or the agent cannot reach the robot at all")


def run_checks(mcp_port: int = 8765) -> list[Result]:
    return [check_groups(), check_ipc(), check_tcp(), check_devices(), check_mcp(mcp_port)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mcp-port", type=int, default=8765)
    args = ap.parse_args(argv)
    if os.geteuid() == 0:
        print("run this as the agent user, not root (root can reach everything)", file=sys.stderr)
        return 2
    results = run_checks(args.mcp_port)
    try:
        import pwd
        who = pwd.getpwuid(os.geteuid()).pw_name
    except (ImportError, KeyError):
        who = str(os.geteuid())
    print(f"isolation check as user {who!r}")
    for r in results:
        print(f"  {r.status:8} {r.name:11} {r.detail}")
    ok = all(r.status == PASS for r in results)
    print("OK: this user can reach the robot only through the MCP server" if ok
          else "NOT ISOLATED: do not run an agent with a shell as this user")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
