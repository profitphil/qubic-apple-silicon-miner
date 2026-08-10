#!/usr/bin/env bash
# rig-diagnose.sh — read-only report of everything that affects qli-Client performance on Ubuntu.
# Run on the mining rig:  bash rig-diagnose.sh
# Safe: makes no changes. sudo only improves the RAM-speed section if available.

set -u
line() { printf '%s\n' "----------------------------------------------------------------------"; }
section() { line; printf '## %s\n' "$1"; line; }

section "CPU"
lscpu | grep -E '^(Model name|Socket|Core\(s\) per socket|Thread\(s\) per core|CPU\(s\)|NUMA node\(s\)|CPU max MHz)' || true
PHYS_CORES=$(( $(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l) ))
echo "Physical cores (start cpuThreads here): ${PHYS_CORES}"
echo -n "SIMD support: "
FLAGS=$(grep -m1 '^flags' /proc/cpuinfo)
for f in avx avx2 avx512f avx512bw avx512vl; do
  echo "$FLAGS" | grep -qw "$f" && printf '%s ' "$f"
done; echo
if echo "$FLAGS" | grep -qw avx512f; then
  echo "=> trainer.cpuVersion should be AVX512 (verify qli.log actually downloads qli-worker-AVX512)"
elif echo "$FLAGS" | grep -qw avx2; then
  echo "=> trainer.cpuVersion should be AVX2"
else
  echo "=> WARNING: no AVX2 — only the slow GENERIC trainer will run; not competitive"
fi

section "SMT / NUMA"
[ -r /sys/devices/system/cpu/smt/control ] && echo "SMT: $(cat /sys/devices/system/cpu/smt/control)"
command -v numactl >/dev/null && numactl --hardware | head -5 || echo "numactl not installed (only matters on multi-socket/EPYC)"

section "Memory"
free -h
echo "Channels/speed (needs sudo dmidecode):"
sudo -n dmidecode -t memory 2>/dev/null | grep -E 'Configured Memory Speed|^\s+Size:' | grep -v 'No Module' | sort | uniq -c \
  || echo "  (run with sudo for RAM speed/channel details)"

section "Hugepages  (trainer wants 52 x cpuThreads)"
grep -E 'HugePages_(Total|Free)|Hugepagesize' /proc/meminfo
echo "Suggested for ${PHYS_CORES} threads: vm.nr_hugepages=$(( 52 * PHYS_CORES ))  (use the exact 'want:' number from /var/log/qli.log if it differs)"

section "CPU frequency governor + thermals"
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
echo "Governor: ${GOV}  $( [ "$GOV" != performance ] && echo '<-- set to performance (rig-tune.sh)' )"
command -v sensors >/dev/null && sensors 2>/dev/null | grep -Ei 'tctl|tdie|package|core 0' | head -5 \
  || echo "lm-sensors not installed: sudo apt install lm-sensors && sudo sensors-detect"

section "GPU (NVIDIA)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,driver_version,power.limit,power.draw,temperature.gpu,clocks.sm --format=csv
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  echo "Driver major ${DRV}: need >=535 (RTX 3000) / >=550 (RTX 4000)"
else
  echo "nvidia-smi not found (no NVIDIA GPU or driver not installed)"
fi

section "GLIBC (need >= 2.31)"
ldd --version | head -1

section "qli client state"
if systemctl list-unit-files 2>/dev/null | grep -q '^qli\.service'; then
  systemctl is-active qli && echo "service: active" || echo "service: NOT active"
  for f in /q/appsettings.production.json /q/appsettings.json; do
    [ -r "$f" ] && { echo "--- $f:"; cat "$f"; }
  done
  echo "--- recent log (worker selection, hugepages, it/s):"
  grep -aE 'worker|hugepage|it/s|epoch|Falling back' /var/log/qli.log 2>/dev/null | tail -25 \
    || sudo -n tail -25 /var/log/qli.log 2>/dev/null || echo "  (cannot read /var/log/qli.log without sudo)"
  echo "--- recent errors:"
  tail -10 /var/log/qli.error.log 2>/dev/null || sudo -n tail -10 /var/log/qli.error.log 2>/dev/null || true
else
  echo "qli systemd service not installed. Official installer:"
  echo "  wget https://dl.qubic.li/cloud-init/qli-Service-install-auto.sh"
  echo "  ./qli-Service-install.sh <threads> <accessToken|payoutId> [alias]"
fi

line
echo "Done. Feed this output back for specific tuning of this rig."
