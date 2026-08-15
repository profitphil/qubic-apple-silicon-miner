# Qubic Stratum pipeline (Apple Silicon)

Connects the Metal BPP9000 miner to the live qubic.li pool over its Stratum WebSocket, so
shares credit your pool account. First Apple-Silicon pool-mining path for Qubic.

## Architecture

```
 pool  ── wss://wps.qubic.li/stratum ──►  scripts/stratum_miner.mjs  (Node driver)
                                            │  login, follow jobs, submit, reconnect
                                            │  stdin: {seed,pubkey,difficulty,epoch}
                                            ▼
                              qiner-metal/build/harness stratum  (Metal compute)
                                            │  re-key pool per seed, mine per pubkey
                                            ▲  stdout: {type:"share",nonce,score,…}
```

- **Node** owns the flaky/JSON WebSocket half (proven reliable) and all submit decisions.
- **Metal** owns compute: for each job it grinds fresh nonces, and emits any whose
  bit-exact BPP9000 score ≤ the job's `DifficultyBpp9000`.

## Protocol (verified live, epoch 226)

- Login: `{"Method":"StratumLogin","AccessToken":<JWT>,"Worker":..,"Os":..}` → `MessageType 2` welcome.
- Job (`MessageType 501`): `{Epoch, RandomSeed(hex32), PublicKey(hex32), DifficultyBpp9000, Bpp9000TaskHash, Bpp9000TaskUrl:null}`. `RandomSeed`+`Bpp9000TaskHash` are epoch-stable; `PublicKey` rotates ~every 15–30s. Legacy `Difficulty`/`DifficultyAddition` are maxed (disabled).
- Submit (`MessageType 502`): `{"Method":"StratumSubmit",Epoch,RandomSeed,PublicKey,Nonce(hex32)}` → `{Success,Message,…}`. Only the nonce is sent; the pool re-derives the score. `Success=true` is provisional (async validation).
- Nonce layout the miner already emits: `nonce[0]=0x01` (AlgoType Bpp9000), `nonce[1]=L∈[1,10]`, `nonce[2]=K=0`, rest = grind counter.

## Run it

Dry-run (safe — mines live jobs, logs would-be shares, **submits nothing**):
```sh
node scripts/stratum_miner.mjs
```

Go live (submits shares) — **only** takes effect when the loaded task's sha256 matches the
live job's `Bpp9000TaskHash`; otherwise it auto-stays dry-run:
```sh
TASK=/path/to/realtask.bin node scripts/stratum_miner.mjs --live
```

Env: `TASK` (task .bin, default = example task, never matches live), `WORKER`, `COHORT` (default 4).
Token is read from `deploy/appsettings.production.json` (gitignored) and never printed.

## Safety interlocks

- **No submit unless `--live` AND the loaded task hash == the live job's hash.** Prevents
  spamming invalid shares (which risks rate-limiting) while the real task isn't wired.
- **No stale submits:** a found nonce is dropped unless its `PublicKey` still equals the
  current job's (pubkey rotates fast).

## The one remaining blocker

The pool does **not** send the task on `/stratum` (`Bpp9000TaskUrl` is always null), and the
qli-Client fetches it over its proprietary `/ws` channel (NativeAOT, no public URL). So the
real task must be obtained via the CTO's fetch method. Once we have `realtask.bin`, going live
is: `TASK=realtask.bin node scripts/stratum_miner.mjs --live`.

**Open question to Qubic:** for a custom `/stratum` miner, how is the BPP9000 task fetched for
a given `Bpp9000TaskHash` when `Bpp9000TaskUrl` is null?

## Notes / tuning

- The M1 does only a few cohorts per pubkey before it rotates, so submittable (current-pubkey)
  shares are sparse — expected for a 7-core GPU. Tune `COHORT` and mine the real task to gauge.
- BPP9000 is replaced by "ant colony" at EP228 (~Aug 26, 2026); this pipeline targets BPP9000.
