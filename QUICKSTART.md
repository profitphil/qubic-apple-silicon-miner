# Quickstart — mine Qubic on an Apple Silicon Mac

The first Apple-Silicon miner for Qubic's live BPP9000 algorithm. It computes **bit-exact**
scores (verified against x86) and connects to the official qubic.li pool over Stratum, so any
shares it finds credit your pool account.

> **Honest expectations.** On a base M-series chip (7–10 GPU cores) this is a **proof of
> concept** — it will rarely, if ever, land a share on the real task, because the network's
> difficulty needs far more nonces-per-job than a small GPU can try before the job rotates
> (~30s). More GPU cores (Pro / Max / Ultra, newer generations) scale roughly linearly, so a
> big Apple GPU is where this gets genuinely interesting. Either way it's a real, valid miner —
> not a simulation.

## Requirements

- Apple Silicon Mac (arm64), macOS.
- **Xcode Command Line Tools**: `xcode-select --install`
- **Node.js 20+** (22+ recommended): `brew install node`
- A **qubic.li account + access token** (free): register at <https://pool.qubic.li>, then copy
  your token from the control panel.

No Xcode app or offline Metal toolchain needed — the shader is compiled at runtime.

## Install (one command)

```sh
git clone https://github.com/profitphil/qubic-apple-silicon-miner
cd qubic-apple-silicon-miner
./install.sh
```

This checks prerequisites, builds the Metal miner, fetches the current BPP9000 task from the
`qubic/core` repo, and runs a self-test proving the GPU and CPU produce identical scores on
**your** machine (`QUICK: ALL PASS`).

## Run

```sh
export QLI_TOKEN='<your token from pool.qubic.li>'

./run.sh          # DRY-RUN: connect + mine live jobs, submit NOTHING (do this first)
./run.sh --live   # submit shares to your account
```

Your token is never written to disk by the tool and never printed. Alternatively create
`qli-config.json` → `{"accessToken":"<your token>"}` (git-ignored) instead of the env var.

### What you'll see

```
[ws] login OK: Welcome <worker>
[job] epoch=NNN diff=3838 pubkey=… taskMatch=true
[stratum] 30 it/s | 12 nonces | 0 shares | best-this-window=4210 (need <=3838)
[SUBMIT #1] score=3771<=3838 nonce=…        # a found share (live mode)
[502] ACCEPTED (1/1)                         # credited to your account
```

- `it/s` is your GPU's score-evaluation rate. `best-this-window` is the lowest score reached
  recently (lower is better; a share needs ≤ the job's difficulty).
- Worker name defaults to `<hostname>-metal`; override with `WORKER=myname ./run.sh`.
- Multiple machines can share one token — just use different worker names.

### Monitor / stop

```sh
# if you launched it in the background (nohup ./run.sh --live > live.log 2>&1 &):
tail -f live.log
pkill -f stratum_miner      # stop
```

Check your worker and shares on the dashboard at <https://platform.qubic.li>.

## How it works

- `qiner-metal/` — the Metal GPU BPP9000 scorer + a `stratum` mode (jobs in on stdin, shares
  out on stdout).
- `scripts/stratum_miner.mjs` — the WebSocket driver: login, follow jobs, submit, reconnect.
- Safety: it will **not** submit unless `--live` is set *and* the loaded task's hash matches
  the live job — so it can never spam invalid shares. It also drops shares whose job already
  rotated. See [STRATUM.md](STRATUM.md) for the full protocol and design.

## Notes

- The BPP9000 task rotates by epoch; `install.sh`/`run.sh` always fetch the current one from
  `qubic/core`. If a run logs `taskMatch=false`, your task file is stale — re-run `./install.sh`
  or delete `qiner-metal/bpp9000-live.task` and re-run.
- Qubic replaces BPP9000 with the "ant colony" algorithm around epoch 228; this miner targets
  BPP9000 and will need an update for the next algorithm.
- Cross-platform validity: scores are consensus-checked by x86 nodes; this miner is verified
  bit-exact against x86, so its solutions are valid. See the repo README.
