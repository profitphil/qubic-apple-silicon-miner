# Qubic on Apple Silicon — Metal GPU miner for BPP9000

First Apple-Silicon miner for Qubic's live **BPP9000** algorithm: a Metal GPU scorer that is
**bit-exact with x86** (so its solutions are consensus-valid) and connects to the official
qubic.li pool over Stratum, crediting shares to your account. Also includes a field-tested
Ubuntu tuning guide for the official [qubic-li client](https://github.com/qubic-li/client).

> **Honest expectations.** On a base M-series GPU (7–10 cores) this is a **proof of concept** —
> it will rarely, if ever, land a share, because the pool difficulty needs far more
> nonces-per-job than a small GPU can try before jobs rotate (~30 s). Bigger Apple GPUs
> (Pro / Max / Ultra, newer generations) scale roughly linearly. It's a real, valid miner
> regardless — not a simulation.

## Requirements

- Apple Silicon Mac (arm64), macOS
- Xcode Command Line Tools: `xcode-select --install`
- Node.js 20+ (22+ recommended): `brew install node`
- A free qubic.li account + access token — register at <https://pool.qubic.li>, copy the
  token from the control panel

No Xcode app or offline Metal toolchain needed — the shader is compiled at runtime.

## Quick start

```sh
git clone https://github.com/profitphil/qubic-apple-silicon-miner
cd qubic-apple-silicon-miner
./install.sh                    # prereq check + build + fetch task + bit-exact self-test

export QLI_TOKEN='<token from pool.qubic.li>'
./run.sh                        # DRY-RUN: connect + mine live jobs, submit NOTHING (do this first)
./run.sh --live                 # submit shares to your account
```

Your token is never written to disk or printed by the tool. (Alternative: create
`qli-config.json` → `{"accessToken":"<token>"}`, git-ignored.) Multiple Macs can share one
token — the worker name defaults to `<hostname>-metal`; override with `WORKER=name ./run.sh`.

### What you'll see

```
[ws] login OK: Welcome <worker>
[job] epoch=226 diff=3838 pubkey=… taskMatch=true
[stratum] 30 it/s | 12 nonces | 0 shares | best-this-window=4210 (need <=3838)
[SUBMIT #1] score=3771<=3838 nonce=…      # a found share (live mode)
[502] ACCEPTED (1/1)                       # credited to your account
```

`it/s` is your GPU's score-evaluation rate; `best-this-window` is the lowest (best) score
reached recently — a share needs a score ≤ the job's difficulty.

### Monitor / stop

```sh
nohup ./run.sh --live > live.log 2>&1 &    # run in the background
tail -f live.log                           # watch it
pkill -f stratum_miner                     # stop it
```

Your worker and shares show up on the dashboard at <https://platform.qubic.li>.

### Verify it yourself (no account needed)

```sh
cd qiner-metal
./build/harness quick         # ~8 s: proves GPU == CPU bit-exact on your machine
./build/harness verify 2      # ~5 min: full 100-mutation bit-exactness (CPU ref dominates)
./build/harness perf 32 100   # GPU cohort throughput
```

## Safety

The miner will **not** submit anything unless `--live` is set *and* the loaded task's hash
matches the live job — so it can never spam invalid shares. It also skips "timeout" pubkeys
(unminable) and drops shares whose job already rotated. See [STRATUM.md](STRATUM.md) for the
full protocol and design.

## What's here

| Path | Contents |
|---|---|
| [`install.sh`](install.sh) / [`run.sh`](run.sh) | one-command setup and launcher |
| [`qiner-metal/`](qiner-metal/) | Metal GPU BPP9000 scorer + `stratum` pool-mining mode |
| [`scripts/stratum_miner.mjs`](scripts/stratum_miner.mjs) | Stratum WebSocket driver — login, follow jobs, submit, reconnect |
| [`qiner-macos/`](qiner-macos/) | arm64 Qiner port (scalar BPP9000 + K12) — the CPU reference |
| [`STRATUM.md`](STRATUM.md) | pool protocol + pipeline design |
| [`docs/METAL-KERNEL-SPEC.md`](docs/METAL-KERNEL-SPEC.md) | full kernel spec + parallelism analysis |
| [`UBUNTU-OPTIMIZATION.md`](UBUNTU-OPTIMIZATION.md) | tuning the official qli client on Ubuntu (x86) |
| [`docs/reference/`](docs/reference/) | unmodified reference headers from [qubic/core](https://github.com/qubic/core) |

## Results (measured, Apple M1, 7-core GPU)

| | CPU scalar (1 thread) | Metal GPU |
|---|---|---|
| per `score()` eval | ~1.2 s | **29.8 ms** — ~40× a CPU thread, ~5–6× the whole 8-core CPU |
| offline bench rate | ~0.5 nonces/min | **~20 nonces/min** |
| live real-task rate | — | ~10–40 it/s (heavier — many live pubkeys hit worst-case scoring) |

**Bit-exact vs x86** — the property that makes solutions valid: the scalar `bench` was built on
both an M1 (arm64) and a Ryzen 9950X (x86); all nonce scores matched (5320 / 5282 / 5310 / 5295)
and the K12 test vectors pass on both. GPU↔CPU on-device is likewise bit-exact (0/8088 window
diffs).

## Status

- ✅ **Pool mining via the sanctioned path** — `wss://wps.qubic.li/stratum` (per the Qubic CTO).
  No qli-Client needed; shares credit your token's account.
- ✅ **Consensus-valid** — arm64↔x86 bit-exact, so x86 verifier nodes reproduce the scores.
- ⚠️ **Throughput-bound on small GPUs** — a base M-series rarely lands shares (see the note at
  top); larger Apple GPUs scale ~linearly. This is the open frontier.
- ⏳ **Algorithm rotation** — BPP9000 is replaced by the "ant colony" algorithm ~Epoch 228
  (late Aug 2026); this miner targets BPP9000 and will need an update for the next algorithm.

## Attribution & license

`qiner-macos/` and the scorer semantics derive from the official
[qubic/core](https://github.com/qubic/core) sources; that project's license applies to all
derived files, and reference headers are reproduced unmodified in `docs/reference/`. The Metal
kernel, Stratum driver, spec, and Ubuntu tooling are original work for the Qubic community.
