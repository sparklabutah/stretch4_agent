"""The isolation check that gates running Codex on the robot's own computer."""
import errno
import socket
from types import SimpleNamespace

import pytest

from stretch4_agent import isolation as iso


def test_probe_tells_open_closed_and_blocked_apart(monkeypatch):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    assert iso.probe_tcp("127.0.0.1", port) == "open"
    srv.close()
    assert iso.probe_tcp("127.0.0.1", port) == "closed"      # nothing listening: not proof of a block

    def refuse(err):
        def connect(self, addr):
            raise err
        return connect
    monkeypatch.setattr(socket.socket, "connect", refuse(OSError(errno.EHOSTUNREACH, "No route to host")))
    assert iso.probe_tcp("127.0.0.1", 1) == "blocked"         # what the firewall rule's reject produces
    monkeypatch.setattr(socket.socket, "connect", refuse(socket.timeout()))
    assert iso.probe_tcp("127.0.0.1", 1) == "blocked"         # a DROP rule


@pytest.mark.parametrize("states,expect", [
    ({"open"}, iso.FAIL), ({"blocked", "open"}, iso.FAIL),
    ({"closed"}, iso.UNKNOWN), ({"blocked", "closed"}, iso.UNKNOWN), ({"blocked"}, iso.PASS)])
def test_tcp_check_passes_only_when_every_port_is_blocked(monkeypatch, states, expect):
    seq = iter(sorted(states) * 10)
    monkeypatch.setattr(iso, "probe_tcp", lambda h, p: next(seq))
    assert iso.check_tcp(["127.0.0.1"], (23114, 23115)).status == expect


def test_group_membership_fails(monkeypatch):
    gids = {"users": 100, "dialout": 20, "plugdev": 46}
    monkeypatch.setattr(iso.grp, "getgrnam", lambda g: SimpleNamespace(gr_gid=gids[g]))
    monkeypatch.setattr(iso.os, "getgid", lambda: 1001)
    monkeypatch.setattr(iso.os, "getgroups", lambda: [1001, 100])
    r = iso.check_groups()
    assert r.status == iso.FAIL and "users" in r.detail
    monkeypatch.setattr(iso.os, "getgroups", lambda: [1001])
    assert iso.check_groups().status == iso.PASS


def test_ipc_dir_must_exist_and_be_closed(tmp_path, monkeypatch):
    assert iso.check_ipc(str(tmp_path / "missing")).status == iso.UNKNOWN
    (tmp_path / "port_command").touch()
    monkeypatch.setattr(iso.os, "access", lambda p, m: True)            # e.g. a member of `users`
    r = iso.check_ipc(str(tmp_path))
    assert r.status == iso.FAIL and "port_command" in r.detail
    monkeypatch.setattr(iso.os, "access", lambda p, m: False)
    assert iso.check_ipc(str(tmp_path)).status == iso.PASS


def test_devices(monkeypatch):
    monkeypatch.setattr(iso.glob, "glob", lambda g: ["/dev/hello-motor-lift"] if "hello" in g else [])
    monkeypatch.setattr(iso.os, "access", lambda p, m: True)
    assert iso.check_devices().status == iso.FAIL
    monkeypatch.setattr(iso.os, "access", lambda p, m: False)
    assert iso.check_devices().status == iso.PASS
    monkeypatch.setattr(iso.glob, "glob", lambda g: [])
    assert iso.check_devices().status == iso.UNKNOWN


def test_exit_code_is_zero_only_when_everything_passes(monkeypatch, capsys):
    monkeypatch.setattr(iso.os, "geteuid", lambda: 1001)
    ok = iso.Result("x", iso.PASS, "")
    monkeypatch.setattr(iso, "run_checks", lambda port: [ok, ok])
    assert iso.main([]) == 0
    monkeypatch.setattr(iso, "run_checks", lambda port: [ok, iso.Result("y", iso.UNKNOWN, "")])
    assert iso.main([]) == 1 and "NOT ISOLATED" in capsys.readouterr().out


def test_refuses_to_run_as_root(monkeypatch):
    monkeypatch.setattr(iso.os, "geteuid", lambda: 0)
    assert iso.main([]) == 2
