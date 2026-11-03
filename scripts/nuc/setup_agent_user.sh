#!/usr/bin/env bash
# Create the unprivileged user that agents run as on the robot's own computer (the NUC).
#
# Why: Codex keeps a shell that no config key removes. Run as the robot's normal user, that
# shell could reach stretch_body_server directly and skip every limit in this project. The
# agent user created here can reach the robot only through `stretch4-agent serve`.
#
# Run from the repository root, as the robot's normal user, with sudo:
#     sudo scripts/nuc/setup_agent_user.sh
# Re-running is safe. Then, as printed at the end: log Codex in as the agent user and run
# the isolation check with stretch_body_server and the MCP server up.
#
# Settings (environment variables):
#     AGENT_USER   default stretch-agent
#     WORK_DIR     default /srv/stretch4-agent  (the agent's working directory)
#     CODEX_BIN    default: `command -v codex` for the invoking user
#     CLAUDE_BIN   default: `command -v claude` for the invoking user
set -euo pipefail

AGENT_USER="${AGENT_USER:-stretch-agent}"
WORK_DIR="${WORK_DIR:-/srv/stretch4-agent}"
ROBOT_USER="${SUDO_USER:-}"
ZMQ_PORTS="23114:23116"

die() { echo "error: $*" >&2; exit 1; }
[ "$(id -u)" -eq 0 ] || die "run with sudo"
[ -n "$ROBOT_USER" ] && [ "$ROBOT_USER" != root ] || die "run with sudo from the robot's normal user, not as root"
[ -f src/stretch4_agent/isolation.py ] || die "run from the repository root"
[ "$AGENT_USER" != "$ROBOT_USER" ] || die "AGENT_USER must differ from the robot user"

as_robot() { sudo -u "$ROBOT_USER" -H bash -lc "$1" 2>/dev/null || true; }
CODEX_BIN="${CODEX_BIN:-$(as_robot 'command -v codex')}"
CLAUDE_BIN="${CLAUDE_BIN:-$(as_robot 'command -v claude')}"

echo "== user $AGENT_USER"
if ! id "$AGENT_USER" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash --user-group "$AGENT_USER"
fi
# Never a member of the groups that open the robot: users (ZMQ IPC sockets), dialout and
# plugdev (motor and camera USB devices), nor of anything that grants root.
for g in users dialout plugdev video sudo adm wheel; do
    if id -nG "$AGENT_USER" | tr ' ' '\n' | grep -qx "$g"; then
        gpasswd -d "$AGENT_USER" "$g" >/dev/null
        echo "   removed from group $g"
    fi
done
[ "$(id -gn "$AGENT_USER")" != users ] || die "$AGENT_USER has 'users' as its primary group; change it"

echo "== firewall: $AGENT_USER may not connect to TCP $ZMQ_PORTS (stretch_body_server)"
add_rule() {   # $1 = iptables or ip6tables, $2 = reject type
    command -v "$1" >/dev/null || { echo "   $1 not found"; return 1; }
    local rule=(OUTPUT -p tcp --dport "$ZMQ_PORTS" -m owner --uid-owner "$AGENT_USER" -j REJECT --reject-with "$2")
    if "$1" -C "${rule[@]}" 2>/dev/null; then echo "   $1 rule present"
    else "$1" -I "${rule[@]}" && echo "   $1 rule added"; fi
}
add_rule iptables icmp-admin-prohibited || die "could not add the IPv4 rule"
add_rule ip6tables icmp6-adm-prohibited || echo "   warning: no IPv6 rule (only matters if IPv6 is enabled)"
if command -v netfilter-persistent >/dev/null; then
    netfilter-persistent save >/dev/null && echo "   saved with netfilter-persistent"
else
    echo "   WARNING: these rules are lost on reboot. Persist them with:"
    echo "       sudo apt install iptables-persistent && sudo netfilter-persistent save"
    echo "   and re-run the isolation check after every reboot until you have."
fi

echo "== working directory $WORK_DIR"
# World-readable on purpose: the runner (robot user) starts the CLI from here through sudo, and
# it holds only copies of this repo's client config. Credentials stay in ~$AGENT_USER/.codex.
install -d -m 0755 -o "$AGENT_USER" -g "$AGENT_USER" "$WORK_DIR" "$WORK_DIR/.codex"
install -m 0644 -o "$AGENT_USER" -g "$AGENT_USER" .mcp.json AGENTS.md CLAUDE.md "$WORK_DIR/"
install -m 0644 -o "$AGENT_USER" -g "$AGENT_USER" .codex/config.toml "$WORK_DIR/.codex/config.toml"
install -m 0755 -o "$AGENT_USER" -g "$AGENT_USER" src/stretch4_agent/isolation.py "$WORK_DIR/check_isolation.py"

echo "== sudo rule: $ROBOT_USER may start the agent CLIs as $AGENT_USER (nothing else)"
bins=()
for b in "$CODEX_BIN" "$CLAUDE_BIN"; do
    [ -n "$b" ] || continue
    b="$(readlink -f "$b")"
    if sudo -u "$AGENT_USER" test -x "$b"; then bins+=("$b")
    else echo "   warning: $AGENT_USER cannot execute $b; install the CLI system-wide (e.g. npm -g as root)"; fi
done
if [ "${#bins[@]}" -gt 0 ]; then
    tmp="$(mktemp)"
    printf '%s ALL=(%s) NOPASSWD:SETENV: %s\n' "$ROBOT_USER" "$AGENT_USER" "$(IFS=,; echo "${bins[*]}" | sed 's/,/, /g')" > "$tmp"
    visudo -cf "$tmp" >/dev/null || { rm -f "$tmp"; die "generated sudoers rule is invalid"; }
    install -m 0440 "$tmp" /etc/sudoers.d/stretch4-agent && rm -f "$tmp"
    echo "   /etc/sudoers.d/stretch4-agent: $(cat /etc/sudoers.d/stretch4-agent)"
else
    echo "   warning: no agent CLI found; set CODEX_BIN / CLAUDE_BIN and re-run"
fi

cat <<EOF

Done. Next, in this order:
  1. Log Codex in as the agent user (it keeps its own credentials in ~$AGENT_USER/.codex):
       sudo -u $AGENT_USER -H ${CODEX_BIN:-codex} login
  2. With stretch_body_server and \`stretch4-agent serve\` running, check the isolation:
       sudo -u $AGENT_USER -H python3 $WORK_DIR/check_isolation.py
     Every line must say PASS. Do not run Codex on this robot until it does.
  3. Run agents as that user:
       stretch4-agent run --agent codex --run-as $AGENT_USER --project-dir $WORK_DIR --task "..."
EOF
