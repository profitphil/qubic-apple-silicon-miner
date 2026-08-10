# qiner-metal — Apple GPU (Metal) port of the Qubic BPP9000 scorer

First Apple-GPU implementation of the live BPP9000 mining scorer (recurrent ternary-LUT
ANN, 64 neurons, 8088 sliding windows, 100-step hill climb per nonce). Verified **bit-exact**
against the scalar arm64 reference (`qiner-macos/`) on an M1 (7-core GPU, 8 GB).

## Build & run

```sh
cd qiner-metal
./build.sh                    # clang++ harness.mm; NO offline Metal toolchain needed
./build/harness quick         # ~8 s : unit tests (initial score, per-window diff, timeout)
./build/harness verify 2      # ~5 min: full bit-exactness, nonces 0+1 (CPU reference dominates)
./build/harness perf 32 100   # GPU throughput, cohort of 32 nonces, 100 lockstep steps
```

Run from `qiner-metal/` (paths default to `kernel.metal` and
`../qiner-macos/data/example_task_bpp9000.bin`; override with `KERNEL=` / `TASK=` env vars).

The offline Metal shader compiler is missing on this machine, so `kernel.metal` is a
**template loaded and compiled at runtime** via `newLibraryWithSource:` (~250 ms cold,
~2 ms warm from the system shader cache). That constraint is exploited: the harness bakes
the task topology (per-neuron neighbor wiring, signal/output bit-slots, input masks) into
the generated MSL as literal constants before compiling.

## Results (Apple M1, 8 GB, macOS, measured 2026-08-09)

| Metric | CPU scalar (1 thread) | GPU (Metal) |
|---|---|---|
| initial score, bench inputs | 5458 | **5458** (bit-exact, all 8088 window predictions identical) |
| nonce 0 best (100 mutations) | 5320 | **5320** PASS |
| nonce 1 best (100 mutations) | 5282 | **5282** PASS |
| forced-timeout LUT | `0xFFFFFFFF` (by construction) | **`0xFFFFFFFF`** PASS |
| per `score()` eval | 1.21–1.26 s | **29.8 ms** (C=2) / 38.8 ms (C=32) |
| est. nonces/min | ~0.49 (1 thread) / ~3–4 (8 threads) | **~20** (C=2) / 15.5 (C=32) |
| speedup | 1x | **40.5×** vs 1 CPU thread; ~5–6× the whole 8-core CPU (scalar) |

Cohort-size measurements (verified clean rebuild, 2026-08-09 ~23:40 EDT):

| Config | ms/dispatch | ms/eval | GPU util | est. nonces/min |
|---|---|---|---|---|
| `verify 2` (C=2, 100 steps) | 59.6 | 29.8 | 99% | 20.1 |
| `perf 32 100` (C=32, 100 steps) | 1242.0 | 38.8 | 100% | 15.5 |

Note: the smaller cohort measured *faster per eval* (29.8 vs 38.8 ms) — per-eval efficiency
drops at C=32, so C≈2–8 looks like the sweet spot on the 7-core M1 GPU; sweep before
production use. Bit-exactness: initial score 5458, nonce 0 → 5320, nonce 1 → 5282, all
matching CPU references computed at runtime (124 s / 119 s scalar runs), plus 0/8088
per-window prediction diffs and a verbatim `0xFFFFFFFF` timeout sentinel.

Reference scores are computed **at runtime** by the linked scalar kernel — nothing hardcoded.

## Design

Follows `docs/METAL-KERNEL-SPEC.md` exactly:

- **Thread = window.** The 8088 windows inside one `score()` are independent (each resets
  all 64 neurons); one GPU thread runs one window's whole recurrent tick loop privately —
  zero synchronization in the hot loop (one threadgroup barrier total, for LUT staging).
- **Packed state.** 64 neuron trits × 2 bits = one `uint4` in registers (UNKNOWN=2 →
  reset pattern `0xAAAAAAAA`). Per-neuron updates are emitted as unrolled MSL with literal
  bit-slot shifts; LUT index composed exactly as the scalar (`t0 + 3*t1 + 9*t2`, exact
  neighbor order — never sorted).
- **LUT rows** (46 non-input neurons × 27 entries × 2 bits) packed into `{lo,hi}` u32
  pairs, staged in threadgroup memory (368 B), dense row order = ascending
  `updatedNeuronIndices` (the same index space the mutator uses, so mutation decode is
  untouched).
- **Feed table.** Per sample, the 18 input trits pre-positioned at their state bit-slots
  (8760 × `uint4` = 140 KB); feeding a sample is `cur = (cur & ~INPUT_MASK) | feed[w+fc]`.
- **Cohort lockstep.** The 100 mutation steps are strictly sequential (hill-climb accept/
  reject chain), but every nonce runs exactly 100 steps → C nonces advance in lockstep,
  one dispatch per step, grid = (32 threadgroups × C candidates) × 256 threads. Mutation
  decode, accept/rollback, K12 and `random2` stay on the CPU (µs per step).
- **Initial score is pubkey-only** (verified: 5458 for the bench pubkey) → computed once
  per pubkey on GPU and shared by every nonce in the cohort.
- **One 512 MB pool.** Page-aligned `mmap` (16 KB pages), generated in-place by the CPU
  Keccak squeeze (~0.35 s), wrapped zero-copy with `newBufferWithBytesNoCopy`
  (storageModeShared) — the same allocation serves the CPU reference and is GPU-visible
  with zero duplication. (The kernel itself never reads the pool: all RNG is CPU-side.)
  Peak process footprint ≈ 0.6 GB.
- **Watchdog safety.** One command buffer per mutation step; at C=2 a dispatch is ~60 ms,
  at C=32 well under 1 s. Status + error checked after every `waitUntilCompleted`.

## Files

- `kernel.metal` — MSL template (markers `//__CONSTANTS__`, `//__NEURON_UPDATES__`
  substituted at runtime with generated, topology-baked code).
- `harness.mm` — Objective-C++ host: pool + task setup, MSL codegen, GPU driver
  (cohort lockstep search), CPU-reference comparison, benchmarks.
- `src/score_bpp9000_ref.h` — copy of the scalar reference with two minimal patches:
  external pool pointer (single shared 512 MB allocation) and an optional per-window
  trace hook used to isolate any GPU/CPU divergence window-by-window.
- `build.sh` — clean checkout → runnable harness.
- `metal-hello/` — earlier standalone runtime-Metal proofs (kept for reference).

## Bit-exactness notes

The kernel preserves the full contract from the spec: trit domain with UNKNOWN=2
participating in LUT indexing; synchronous two-phase tick (read `cur`, accumulate `nxt`);
feed/settle/break protocol order with feed counter saturating at exactly 672;
timeout after exactly 100000 loop iterations poisons the whole eval with the verbatim
`0xFFFFFFFF` sentinel (evaluating remaining windows anyway — bit-identical); u32 score
arithmetic; `INFINITE_ERROR <= INFINITE_ERROR` accepts; u64 mutation decode with 64-bit
modulo; equal-score accepts; full-LUT rollback on reject.

Divergence tooling (used during bring-up, all green): the kernel always writes each
window's predicted trit (255 on timeout) to a debug buffer; the patched reference can
write the same trace, and `quick` mode diffs all 8088 windows.

## Known limitations / follow-ups

- Timeout sentinel verified GPU-side only (a full scalar timeout run would take hours;
  the scalar path is trivially `INFINITE_ERROR` by construction).
- Tick-count divergence across a simdgroup costs ~5–10% (measured spread p10–p90 is
  ±6%); window binning could recover part of it.
- Cone-of-influence neuron filtering (bit-exact skip of neurons that can't reach
  signal/output) not applied — this task's cone is the full 46 neurons anyway.
- A NEON (`tbl`) CPU port could add a further multi-x on the CPU side (AVX2 fallback
  maps directly), independent of this kernel.
- For production mining: double-buffer two cohorts (A/B staggered) to hide the ~1%
  CPU turnaround, and serialize pipelines with `MTLBinaryArchive`.
