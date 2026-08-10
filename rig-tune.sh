#!/usr/bin/env bash
# rig-tune.sh — apply the two verified OS-level optimizations for the qli-Client trainer on Ubuntu:
#   1) hugepages = 52 x threads (trainer falls back to slow malloc without them)
#   2) performance CPU governor (persisted via a small systemd unit)
# Usage:  sudo bash rig-tune.sh <cpuThreads>
# Example: sudo bash rig-tune.sh 16
# Idempotent: safe to re-run. Re-run after changing cpuThreads in appsettings.

set -euo pipefail

[ "${EUID}" -eq 0 ] || { echo "Run with sudo: sudo bash rig-tune.sh <cpuThreads>"; exit 1; }
THREADS="${1:-}"
[[ "${THREADS}" =~ ^[0-9]+$ ]] || { echo "Usage: sudo bash rig-tune.sh <cpuThreads>  (the value in your appsettings trainer.cpuThreads)"; exit 1; }

PAGES=$(( 52 * THREADS ))
echo "==> Hugepages: setting vm.nr_hugepages=${PAGES} (52 x ${THREADS} threads)"
sysctl -w vm.nr_hugepages="${PAGES}"
printf 'vm.nr_hugepages=%s\n' "${PAGES}" > /etc/sysctl.d/99-qubic.conf
ALLOC=$(grep HugePages_Total /proc/meminfo | awk '{print $2}')
if [ "${ALLOC}" -lt "${PAGES}" ]; then
  echo "    WARNING: only ${ALLOC}/${PAGES} pages allocated (fragmented memory)."
  echo "    A reboot allocates them cleanly at boot (now persisted in /etc/sysctl.d/99-qubic.conf)."
fi
echo "    NOTE: if /var/log/qli.log prints a different 'want:' number, re-run with that thread count."

echo "==> CPU governor: performance (persisted via systemd unit)"
if command -v cpupower >/dev/null 2>&1; then
  cpupower frequency-set -g performance >/dev/null
else
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g" 2>/dev/null || true
  done
  echo "    (tip: apt install linux-tools-common linux-tools-\$(uname -r) for cpupower)"
fi
cat > /etc/systemd/system/cpu-performance-governor.service <<'EOF'
[Unit]
Description=Set performance CPU governor for Qubic mining
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now cpu-performance-governor.service >/dev/null 2>&1 || true
echo "    governor now: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"

echo "==> Optional epoch-rollover restart timer (Wed 12:05 UTC)"
if systemctl list-unit-files 2>/dev/null | grep -q '^qli\.service'; then
  cat > /etc/systemd/system/qli-epoch-restart.service <<'EOF'
[Unit]
Description=Restart qli miner at Qubic epoch rollover

[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl restart qli
EOF
  cat > /etc/systemd/system/qli-epoch-restart.timer <<'EOF'
[Unit]
Description=Qubic epoch rollover restart (epochs flip Wednesday 12:00 UTC)

[Timer]
OnCalendar=Wed *-*-* 12:05:00 UTC
Persistent=false

[Install]
WantedBy=timers.target
EOF
  systemctl daemon-reload
  systemctl enable --now qli-epoch-restart.timer
  echo "    installed + enabled qli-epoch-restart.timer"
  systemctl restart qli 2>/dev/null && echo "    restarted qli to pick up hugepages" || true
else
  echo "    qli service not found — skipped timer install (install the client first, then re-run)"
fi

echo
echo "Done. Verify in /var/log/qli.log that:"
echo "  - the hugepages line no longer says 'Falling back to use malloc memory'"
echo "  - the downloaded worker matches your CPU (qli-worker-AVX512 / -AVX2)"
