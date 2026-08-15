#!/usr/bin/env bash
# One-command setup for the Apple-Silicon Qubic Metal miner.
#   git clone https://github.com/profitphil/qubic-apple-silicon-miner && cd qubic-apple-silicon-miner
#   ./install.sh
set -euo pipefail
cd "$(dirname "$0")"
echo "=== Qubic Apple-Silicon miner — install ==="

# 1) prerequisites
[ "$(uname -s)" = "Darwin" ] || { echo "ERROR: macOS required."; exit 1; }
if [ "$(uname -m)" != "arm64" ]; then echo "WARNING: not Apple Silicon — the Metal GPU path expects an Apple GPU."; fi
command -v clang++ >/dev/null 2>&1 || { echo "ERROR: Xcode Command Line Tools missing. Run:  xcode-select --install"; exit 1; }
command -v node    >/dev/null 2>&1 || { echo "ERROR: Node.js 20+ missing (22+ recommended).  brew install node"; exit 1; }
NODEV=$(node -e 'console.log(+process.versions.node.split(".")[0])')
[ "$NODEV" -ge 20 ] || { echo "ERROR: Node $NODEV too old; need 20+ (22+ has built-in WebSocket)."; exit 1; }
echo "  OK: $(uname -m) macOS, clang $(clang++ --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1), node $(node -v)"

# 2) build the Metal miner (runtime-compiled shader; no offline Metal toolchain needed)
echo "=== building Metal miner ==="
( cd qiner-metal && ./build.sh )

# 3) fetch the current BPP9000 task from the qubic/core repo
echo "=== fetching BPP9000 task (qubic/core) ==="
curl -fsSL "https://raw.githubusercontent.com/qubic/core/main/data/bpp9000.task" -o qiner-metal/bpp9000-live.task
echo "  task sha256: $(shasum -a256 qiner-metal/bpp9000-live.task | cut -d' ' -f1 | cut -c1-16)…"

# 4) self-test: prove GPU==CPU bit-exact on THIS machine
echo "=== self-test (GPU vs CPU bit-exactness) ==="
( cd qiner-metal && ./build/harness quick )

cat <<'EOF'

=== install complete ===
Next steps:
  1) Get an access token at https://pool.qubic.li  (register -> control panel)
  2) export QLI_TOKEN='<your token>'
  3) ./run.sh          # DRY-RUN: connect + mine live jobs, submit nothing (safe first check)
     ./run.sh --live   # submit shares to your pool account

Notes:
  - A base M-series (e.g. 7-8 GPU cores) will rarely if ever land a share on real BPP9000 —
    it is a proof-of-concept. More GPU cores (Pro/Max/Ultra) scale roughly linearly.
  - See QUICKSTART.md for monitoring, stopping, and honest expectations.
EOF
