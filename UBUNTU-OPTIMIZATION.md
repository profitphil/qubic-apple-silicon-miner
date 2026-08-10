# Qubic Miner — Ubuntu Optimization Playbook

**Verified against qli-Client v3.7.2 (released 2026-08-09) and the live qubic-li/client README, epoch ~225, algorithm BPP9000.**
All claims below were researched and adversarially fact-checked on 2026-08-09. Qubic rotates its
training algorithm every 2–3 months and swaps trainer binaries server-side every epoch, so
**re-verify anything performance-related after each Wednesday 12:00 UTC epoch change.**

---

## 0. What you are actually mining (August 2026)

- CPU/GPU miners run **100% Aigarth AI training** — currently **BPP9000** (since Epoch 224, late
  July 2026): searching configurations of a recurrent ternary-LUT neural network, where **lower
  score = better** (a reversal from earlier algorithms).
- The Monero/XMRig dual-mining era is **over** (XMR support removed in client v3.5.3; Monero custom
  mining ended April 15, 2026). Any guide mentioning MSR mods, RandomX, or "Qubic mines only 50% of
  the time" is stale — the trainer now runs continuously.
- Dogecoin merge-mining exists but is **Scrypt-ASIC only** (Antminer L3+/L7/L9 → `stratum+tcp://doge.qubic.li:12480`); it does not involve your CPU/GPU.
- ⚠️ **Second halving at Epoch 227 (~Wednesday, August 19, 2026):** burn-rate ceiling rises from 55%
  to 77.5% (managed dynamically by the Supply Watcher contract). Worst case, net miner emissions
  roughly halve. Redo any profitability math after that date.

## 1. Client deployment — do these first

| Do | Why |
|---|---|
| Run **native Linux x64 v3.7.2+** (`https://dl.qubic.li/downloads/qli-Client-3.7.2-Linux-x64.tar.gz`) or the systemd installer | v3.7.0+ contains the required BPP9000 changes |
| **Do not use Docker** | Docker Hub `qubicli/client` is stuck at **3.4.5** (Sept 2025) — it predates BPP9000 and cannot mine the current algorithm. (The README's `qubicli/qubic-client` name doesn't even exist.) |
| Put your config in `/q/appsettings.production.json` (systemd install) | It **survives client upgrades**; `appsettings.json` may be overwritten |
| Match the config **root key** to your install: systemd installs use `"Settings"`, portable/screen installs use `"ClientSettings"` | Both appear in official samples; using the wrong root silently ignores your settings |
| Set `"autoUpdate": true` on a headless rig (or monitor releases closely) | Algorithm changes ship as client releases (3.7.0 = BPP9000); a stale client mines nothing. Tradeoff: you're auto-running new closed-source binaries — see §8 |
| Run as a **dedicated non-root user** | The client auto-downloads and executes closed-source `qli-worker*` binaries every epoch; the README itself says least-privilege |
| Prefer a **registered accessToken** (JWT from pool.qubic.li) over registerless `qubicAddress`/`payoutId` | Pool officially recommends registration; gives you the dashboard for benchmarking |

systemd quick install (official): `wget https://dl.qubic.li/cloud-init/qli-Service-install-auto.sh`
→ `./qli-Service-install.sh <threads> <accessToken|payoutId> [alias]` → installs to `/q`, logs to
`/var/log/qli.log`, manage with `systemctl {start,stop,status} qli`.

## 2. CPU trainer settings (biggest config wins)

```jsonc
// /q/appsettings.production.json (systemd root key shown)
{
  "Settings": {
    "accessToken": "YOUR_JWT",
    "alias": "ubuntu-rig-1",
    "pps": true,
    "autoUpdate": true,
    "displayDetailedHashrates": true,
    "trainer": {
      "cpu": true,
      "cpuVersion": null,      // null = auto. PIN IT if the log picks wrong (see below)
      "cpuThreads": 16,        // start = physical cores; then A/B test (see below)
      "gpu": true,
      "gpuVersion": "CUDA",
      "gpuCards": null         // "-1" per card = auto-tune; "0,..." disables a card
    }
  }
}
```

- **`cpuVersion` — verify, don't trust auto-detect.** Check `/var/log/qli.log` at startup for which
  worker it downloaded (`qli-worker-AVX512` etc.). Auto-detection has historically picked AVX2 on
  AVX-512 CPUs and GENERIC on AVX2 CPUs (issues #46, #91). Check your CPU with
  `grep -o 'avx512f\|avx2' /proc/cpuinfo | sort -u` and pin `"AVX512"`, `"AVX2"`, or `"SKYLAKE"`
  (Skylake-X/SP Intel) accordingly. If the AVX512 worker crash-loops (`trap invalid opcode` — seen
  on some Skylake-SP Xeons, #110), drop to `"AVX2"`.
- **`cpuThreads` — more is often worse.** The README itself says "sometimes less threads is more
  effective." Verified community results: a 7950X was fastest at **16 threads (physical cores), not
  32 SMT threads**; a dual-EPYC 7742 did 500 it/s at 32 threads but only 200 it/s at 128–256.
  Protocol: start at physical core count → run ≥15 min → record it/s → try −2/+2 → keep the winner.
  Also **verify the startup log honors your value** (v3.4.4 ignored it on Windows, #133).
- **SMT/Hyper-Threading:** worth one BIOS A/B test. One dual-Xeon report: HT on = ~+14% it/s but
  unstable (intermittent collapse); HT off = stable at −12%. Test on your silicon.
- If the GPU trainer runs on the same box, leave 1–2 cores unpinned for it.

## 3. Hugepages (the #1 Linux-only free win)

The trainer wants **52 hugepages × thread count**; without them it silently falls back to `malloc`
and loses it/s. It prints the exact number it wants in the log
(`have: X - want: N (52 x number of threads)`).

```bash
# example for 31 threads (52 × 31 = 1612) — use the number YOUR log prints
sudo sysctl -w vm.nr_hugepages=1612
# persist:
echo 'vm.nr_hugepages=1612' | sudo tee /etc/sysctl.d/99-qubic.conf
```

Set it **before** the service starts (fragmented memory late in uptime can fail large allocations).
`rig-tune.sh` in this folder automates this.

## 4. NUMA / multi-socket (EPYC, dual Xeon)

The client is **not NUMA-aware**. Verified worst case: 2× EPYC 7742 tripled its it/s going from
default BIOS to **NPS1 + "L3 Cache As NUMA Domain" enabled** (issue #27). If you have a
single-socket desktop CPU, skip this section.

- BIOS: **NPS1**, L3-as-NUMA enabled; avoid NPS2/NPS4.
- Alternative: one client instance per socket/node, pinned with `numactl --cpunodebind=N --membind=N`
  (or `trainer.cpuAffinity`), each with `UseAliasAsIdentifier: true` so the pool shows them
  separately. Expect diminishing returns — shared memory bandwidth is the bottleneck (#59).
- Historical note: Windows only ever loads ~50–64 threads on big rigs (#7, #32, #82) — you're
  already on Ubuntu, which is the right call.

## 5. GPU (NVIDIA)

- Driver ≥ **535** (RTX 3000) / ≥ **550** (RTX 4000); CUDA 12 runtime. Wrong driver is the top
  "GPU trainer won't start" cause.
- `gpuCards`: leave at auto (`-1` per card). Known bug: auto-tune sizes gThreads by VRAM and can
  misallocate on mixed rigs (#94) — then set explicit values, e.g. `"512,256"`.
- **Back off aggressive core overclocks.** `qli-worker-CUDA not running or has exited` crash loops
  were fixed by dropping +225 → +150 core (#114). The workload pushes cards to max power; a locked
  core clock + power-limit cap gets most of the it/s at far fewer watts. 2024-era starting points
  (re-tune under BPP9000): 30-series `--setclocks 1500 --setcoreoffset 150`, 40-series
  `--setclocks 2400 --setcoreoffset 150`, then cap PL and measure it/s-per-watt.
- AMD GPUs: the `"AMD"` trainer is epoch-dependent and often absent — NVIDIA is the safe path.
- Datacenter GPUs are wasted here: an H800 benchmarked ≈ RTX 4090 (workload is FP32-bound, ignores
  tensor cores) (#21).
- No public RTX 50-series/BPP9000 numbers exist yet — benchmark before buying anything.

## 6. OS-level Ubuntu tuning

```bash
# performance governor (rig-tune.sh does this + persistence)
sudo apt install -y linux-tools-common linux-tools-$(uname -r)
sudo cpupower frequency-set -g performance
```

- **Cooling is a real tuning knob.** The maintainer's only direct performance statement: it/s
  variance tracks thermal throttling. Watch `sensors` during a run; fix airflow before chasing configs.
- **Efficiency > peak clocks** for $/day: on Ryzen, a negative Curve Optimizer / lowered PPT
  undervolt cuts watts far more than it/s. AVX-512 all-core is the worst-case stability load —
  validate each step (community CO range −15…−30 is from general Zen 4 tuning, not Qubic-specific).
- RAM: ≥16 GB, **both channels populated**, highest stable speed (official HiveOS docs: higher RAM
  frequency measurably helps this workload).
- GLIBC ≥ 2.31 required (Ubuntu 22.04/24.04 are fine).
- Bare metal only — no VMs/WSL for production mining.

## 7. Epoch operations (where silent losses hide)

- Epochs run **Wednesday 12:00 UTC → Wednesday 12:00 UTC**; payouts weekly; mine the full week
  (late joiners lose eligibility).
- **Restart the miner at each epoch rollover** — stale-parameter and worker-crash bugs at epoch
  boundaries are a recurring theme (#57, #85, #135). Automate it:

```ini
# /etc/systemd/system/qli-epoch-restart.timer
[Unit]
Description=Restart qli at Qubic epoch rollover
[Timer]
OnCalendar=Wed *-*-* 12:05:00 UTC
[Install]
WantedBy=timers.target
```
```ini
# /etc/systemd/system/qli-epoch-restart.service
[Unit]
Description=Restart qli miner
[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl restart qli
```
`sudo systemctl daemon-reload && sudo systemctl enable --now qli-epoch-restart.timer`

- **Watchdog:** grep `/var/log/qli.error.log` for `not running or has exited` on a 5-min cron; if
  seen repeatedly, run the official reset:
  `systemctl stop qli --no-block; pkill -f qli; rm /q/*.lock; rm /q/qli-worker*; systemctl start qli`
- **Benchmark discipline:** only compare it/s within the same epoch/trainer version. All published
  per-device numbers (7950X ~850 it/s, 4090 ~2100 it/s) are 2024-era, pre-BPP9000 — use them as
  relative rankings only. hashrate.no stopped tracking Qubic entirely for this reason.
- The `idling{}` block (run another program during network idle) is likely **vestigial** since the
  April 2026 move to continuous training — don't invest in idle-phase dual mining.

## 8. Pool choice & security

| Pool | Fee | Model | Notes |
|---|---|---|---|
| qubic.li (qli client default) | up to 7% | PPS or solo | Largest; the qli client is built for it |
| Qubic-Solutions | **0%** | PPS/solo | Different miner stack (rqiner/OXZD, CUDA/ROCm/AVX-512); config via poolhub.io |
| Apool | 10% | PPLNS | Pays one epoch delayed |
| MinerLab | 10% (negotiable) | PPS | Fastest payout |
| JETSKI | 8% | PPLNS | — |

Keep `pps: true` unless you have farm-scale hashrate (solo variance). If the ~7% qli fee bothers
you, the real alternative is switching miner stacks to Qubic-Solutions' 0%-fee OXZD/rqiner — worth
an A/B week against the qli client on the same hardware.

Security posture (you're executing closed-source binaries fetched at runtime):
- dedicated non-root user; systemd hardening (`NoNewPrivileges=true`, `ProtectHome=true`);
- egress firewall allowlist: `dl.qubic.li` (downloads) + `wss.qubic.li`/`wps.qubic.li` (pool);
- `downloadUrlRewrites` exists if you ever want to proxy/mirror trainer downloads for a farm.

## 9. Priority checklist (do in this order)

1. Native client ≥ 3.7.2, systemd install, config in `appsettings.production.json` (correct root key).
2. Hugepages = 52 × threads (run `rig-tune.sh`, confirm no "falling back to malloc" in the log).
3. Verify the log downloaded the right CPU worker (AVX512/AVX2); pin `cpuVersion` if not.
4. Thread A/B: physical cores first, then ±2; verify the log honors the setting.
5. Performance governor + confirm no thermal throttling.
6. GPU: driver ≥535/550, tame the core OC, cap power limit, measure it/s per watt.
7. Epoch-restart timer + crash watchdog.
8. Re-baseline every Wednesday; recheck profitability after Epoch 227 (~Aug 19).

Files in this folder: [rig-diagnose.sh](rig-diagnose.sh) (read-only report — run first),
[rig-tune.sh](rig-tune.sh) (applies hugepages + governor),
[appsettings.production.json.example](appsettings.production.json.example).
