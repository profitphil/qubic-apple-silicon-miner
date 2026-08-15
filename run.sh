#!/usr/bin/env bash
# Launch the Apple-Silicon Qubic Metal miner against the live pool.
#   export QLI_TOKEN='<token from pool.qubic.li>'
#   ./run.sh          # DRY-RUN (no submits) — verify it connects + mines
#   ./run.sh --live   # submit shares to your account
# Env: QLI_TOKEN (required), WORKER (default <hostname>-metal),
#      COHORT (default 4; keep it small 2-8 — large values trip the macOS GPU watchdog)
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${QLI_TOKEN:-}" ] && [ ! -f qli-config.json ] && [ ! -f deploy/appsettings.production.json ]; then
  echo "Set your token first:  export QLI_TOKEN='<token from https://pool.qubic.li>'"
  exit 1
fi

# ensure the task is present (fetch current one from qubic/core if missing)
if [ ! -f qiner-metal/bpp9000-live.task ]; then
  echo "Fetching BPP9000 task from qubic/core…"
  curl -fsSL "https://raw.githubusercontent.com/qubic/core/main/data/bpp9000.task" -o qiner-metal/bpp9000-live.task
fi

if [ ! -x qiner-metal/build/harness ]; then
  echo "Miner not built. Run ./install.sh first."; exit 1
fi

export TASK="$(pwd)/qiner-metal/bpp9000-live.task"
export WORKER="${WORKER:-$(hostname -s)-metal}"
MODE=$([ "${1:-}" = "--live" ] && echo "LIVE (submitting)" || echo "DRY-RUN (no submits)")
echo "worker=$WORKER  task=$(shasum -a256 "$TASK" | cut -d' ' -f1 | cut -c1-16)…  mode=$MODE"
echo "(Ctrl-C to stop.  Monitor: watch for [SUBMIT] / [502] ACCEPTED lines.)"
exec node scripts/stratum_miner.mjs "$@"
