# METAL KERNEL SPEC — BPP9000 scorer on Apple M1 (7-core GPU, Metal 4)

Status: definitive porting specification, derived by reading the code (not summaries).

Authoritative sources (all read line-by-line for this spec):

| Role | File |
|---|---|
| Scalar semantics (bit-exact reference) | `qiner-macos/src/score_bpp9000.h` (505 lines) |
| Pool + random2 RNG | `qiner-macos/src/score_common.h` |
| Task file layout / trit packing | `qiner-macos/src/task_file.h` |
| Exact bench inputs (pubkey/seeds/nonces) | `qiner-macos/bench.cpp` |
| How the pros batch windows (AVX-512/AVX2) | scratchpad `qubic-core/src/mining/score_bpp9000.h` (1073 lines) |

Measured on this machine (instrumented `-O3` build of the scalar kernel, bench inputs,
miningSeed = 01..20, pubkey = A0..BF, task `data/example_task_bpp9000.bin`):

- pool + task init: **0.44 s**
- one `score()` eval: **1.26 s** → ~127 s per nonce single-threaded (101 evals)
- **initial score (root LUT, identical for every nonce under one pubkey): 5458**
- ticks-to-settle per window: **min 1562, p10 1649, p50 1700, p90 1754, p99 1801, max 1863, avg ≈ 1701**
  (distribution is tight: max/min ≈ 1.19 — crucial for SIMD divergence, see §6.5)
- 3-mutation short search reproduces monotone improvement (5458 → 5352), mutated LUTs have the
  same tick distribution (avg 1699.7) — mutation does not change settle-time behavior materially.

Prior full-run baseline (unverified this session, from earlier work): nonce 0 → 5320, nonce 1 → 5282,
2.5–4 min/nonce/thread. Re-run `build/bench` before using these as parity targets.

---

## 1. Constants and data structures (exact sizes)

### 1.1 Algorithm constants (`src/score_bpp9000.h:15-29`)

| Constant | Value | Notes |
|---|---|---|
| `NUMBER_OF_INPUT_NEURONS` (N) | 18 | |
| `NUMBER_OF_OUTPUT_NEURONS` (M) | 1 | `score()` grades only output 0 (static_assert :70-72) |
| `POPULATION_THRESHOLD` (P) | 64 | = `maxNumberOfNeurons`; power of 2 (static_assert :64-66) |
| `NUMBER_OF_NEIGHBORS` (K) | 3 | LUT index hardcoded for 3 (static_assert :58-60) |
| `NUMBER_OF_MUTATIONS` | 100 | hill-climb steps ⇒ 101 `score()` evals/nonce |
| `MAX_NUMBER_OF_TICKS` | 100000 | per-window timeout |
| `MAX_LUT_ENTRIES_PER_STEP` | 10 | upper clamp of L = nonce[1] |
| `SEQUENCE_LENGTH` (T) | 8760 | = 24×365 |
| `WINDOW_WIDTH` (W) | 672 | = 24×28 |
| `NUMBER_OF_WINDOWS` | **8088** | = T − W (:25, :44) |
| `SOLUTION_THRESHOLD` | **6469** | = (8088−1)·4/5 with unsigned truncating division (:29) |
| `TRIT_UNKNOWN` | 2 | trit domain {0,1,2}; 2 doubles as "unknown" (:52) |
| `INFINITE_ERROR` | 0xFFFFFFFF | u32 sentinel (:54) |
| `lutSize` | 27 | = 3³ entries per neuron |
| updated (non-input) neurons | **46** | = P − N; includes the output and signal neurons |

Derived: number of LUT lines the mutator can hit = 46 × 27 = **1242**.

### 1.2 Host-side structures (scalar layout, `src/score_bpp9000.h:246-292`)

| Structure | Layout | Size |
|---|---|---|
| `inputs[T][N]` | u8 trits, row-major per sample | 8760×18 = **157,680 B** |
| `outputs[T][M]` | u8 trits | 8760 B |
| `Neuron` | `{ enum Type (4B); u8 value; pad }` | 8 B |
| `ANN` | `neurons[64]` (512 B) + `lut[64·27]` (1728 B) | **2240 B** ×3 (best/current/prev) |
| `InitValue.lutInit` | u8[64·27] raw RNG bytes | **1728 B** |
| `InitValue.mutationSeed` | u64[100·10] | **8000 B** |
| `neighborIndices` | u32[64·3], `neighborIndices[n·3+k]` = k-th neighbor of neuron n, ∈[0,64) | 768 B |
| `inputNeuronIndices` / `outputNeuronIndices` / `signalNeuronIndex` | u32[18] / u32[1] / u32 | 72+4+4 B |
| `updatedNeuronIndices` | u64[≤64], ascending list of the 46 non-input neurons (:236-243) | 368 B used |
| task file | 96 B header + 848 B topology + 43,800 B packed data (5 B/row: ceil(18/5)+ceil(1/5)) | 44,744 B |

Topology is *explicit per-neuron wiring*: any neuron may pick any 3 neighbors, duplicates allowed
among a neuron's neighbors (validation `:167-218` only checks range and that input/output/signal
indices are distinct from each other). Do **not** assume a ring or lattice.

Only the LUT changes under mutation; neuron wiring, roles, inputs/outputs are frozen per task.
The AVX-512 engine's `ANN` accordingly holds only the LUT (ref `:82-88`).

### 1.3 GPU-side repacked structures (proposed; see §5)

| Structure | Layout | Size |
|---|---|---|
| packed neuron state | 64 trits × 2 bits = 128 bits = 4×u32 (neuron j at bits 2j mod 32 of word j/16); UNKNOWN pattern = 0xAAAAAAAA | 16 B/window, in registers |
| packed LUT row (per updated neuron) | 27 entries × 2 bits = 54 bits; store as 2×u32 (entries 0–15 in `lo`, 16–26 in `hi`) to avoid emulated 64-bit shifts | 8 B × 46 = 368 B/candidate |
| pre-positioned feed table | per sample: 4×u32 with the 18 input trits already at their state bit-slots, other bits 0; plus the constant input mask | 8760×16 = **140,160 B** |
| expected outputs | u8[8088] = `outputs[t+672][0]` | 8088 B |
| per-candidate result | `{atomic_uint failures; atomic_uint timeout;}` | 8 B |

---

## 2. Full loop nest, with parallelism verdicts

Notation: `computeScore` is the per-nonce entry (`src/score_bpp9000.h:440-496`).

```
computeScore(pubkey, nonce)                                  [per nonce: INDEPENDENT → DATA-PARALLEL across nonces]
 ├─ initializeANN (:455 → :402-436)                          [SEQUENTIAL prologue, mostly CPU-friendly]
 │   ├─ K12(pubkey) → rootHash; random2 → lutInit 1728 B     (:407-409)
 │   ├─ K12(pubkey‖nonce, bytes32..34:=0) → searchHash;
 │   │   random2 → mutationSeed 8000 B                       (:411-419)
 │   ├─ LUT init: lut[n·27+l] = lutInit[n·27+l] % 3          (:427-433)
 │   └─ score()                                              [see below]
 ├─ LOOP s = 0..99  mutation steps (:459)                    [SEQUENTIAL — hill-climb chain, see 2.1]
 │   ├─ snapshot prevANN (:461)
 │   ├─ LOOP i = 0..L-1  mutate(seed[s·10+i]) (:463-466)     [SEQUENTIAL in i (mutations may hit the same line), trivial cost]
 │   ├─ r = score() (:468)                                   [the 99.9% hot spot]
 │   └─ accept iff r <= cur (K=0, :470-478); else rollback (:486); track best (:489-493)
 └─ return best

score()  (:327-383)
 └─ LOOP trainingEntryIndex t = 0..8087 windows (:333)       [DATA-PARALLEL — proven independent, see 2.2]
     ├─ reset all 64 neuron values := UNKNOWN (:337-340)
     ├─ LOOP tick = 0..99999 (:343)                          [SEQUENTIAL — recurrent dynamics]
     │   ├─ if signal==UNKNOWN && feedCounter>=672: break    (:345-351)  ← settled; break BEFORE processTick
     │   ├─ if signal==UNKNOWN: feed inputs[t+feedCounter][0..17] into the 18 input neurons; feedCounter++ (:352-357)
     │   ├─ else: input neurons := UNKNOWN (:359-364)
     │   └─ processTick() (:366)
     │       ├─ LOOP n = 0..63 compute nextNeuronValue (:300-313)   [DATA-PARALLEL across neurons, barrier after]
     │       │     input neuron: next = value (copy-through)
     │       │     else: next = lut[n·27 + t0 + 3·t1 + 9·t2], tk = value of neighborIndices[n·3+k]
     │       └─ LOOP n = 0..63 commit (:315-321)                    [the two phases = synchronous update; double-buffer]
     ├─ if tick==100000: return INFINITE_ERROR (:369-372)    ← ANY window timing out poisons the whole score
     └─ grade: predicted = output-neuron value at break tick; expected = outputs[t+672][0];
        failures += (predicted != expected) (:374-379)       [sum-reduction; feedCounter==672 exactly at break]
```

### 2.1 The 100 mutations are SEQUENTIAL (verified)

`mutate()` (`:386-398`) reads `oldTrit` **from `currentANN.lut`** and writes `(oldTrit+1+delta)%3`
back. `currentANN` at step s+1 is either the mutated LUT of step s (accept) or the rollback copy
`prevANN` (`:486`). Therefore the LUT that step s+1's mutations apply to depends on the
accept/reject outcome of step s — a genuine data dependence. The mutation *seed sequence* is
precomputed and independent of outcomes (`initValue.mutationSeed`, drawn once at init `:419`), but
the *effect* of a seed depends on the current LUT contents. Verdict:

- **Across steps s: SEQUENTIAL.** Cannot batch the 100 steps of one nonce. (Speculating both
  accept/reject branches doubles work for 2× parallelism — break-even, not worth it; deeper
  speculation is exponential. Rejected.)
- **Within a step (i = 0..L−1): SEQUENTIAL** in principle — two seeds can select the same LUT
  line and the second reads the first's write. Irrelevant for GPU: run `mutate` on the CPU
  (≤10 byte-writes per step per nonce).
- **Across nonces: DATA-PARALLEL.** Each nonce is an independent hill climb (own mutation-seed
  stream); every nonce performs exactly 100 steps, so a cohort of nonces stays in lockstep —
  step s of all nonces can share one GPU dispatch (see §5.4).
- **Bonus (verified):** the initial LUT depends only on `pubkey` (`:407-409` uses no nonce), and
  neuron values start all-UNKNOWN, so the **initial `score()` is one shared eval per
  (pubkey, miningSeed, task)** — compute once (measured value 5458 for the bench inputs), reuse as
  `cur`/`best` starting point for every nonce. 101 evals/nonce → 100 + 1/cohort.

### 2.2 Windows within one score() are DATA-PARALLEL (verified)

Each window iteration begins by resetting **all 64** neuron values to UNKNOWN (`:337-340`); the only
state crossing window iterations is `numberOfFailures` (a sum) and the early `return INFINITE_ERROR`
(an OR of per-window timeout flags — returning after evaluating all windows is bit-identical
because the return value is the constant sentinel either way). The AVX-512 engine confirms this by
running 128 windows per batch as independent lanes (ref `:456-534`: 2 windows per byte, lo nibble =
window base+l, hi nibble = window base+64+l, per-lane feed counters, per-lane done masks). Windows
are *self-clocked*: each window's feed pace depends on its own signal-neuron dynamics, so lanes
desynchronize in feedCounter but never interact.

Per-window sample span: window t reads `inputs[t .. t+671]` (in feed order, one sample per feed
tick) and grades against `outputs[t+672][0]`. Windows overlap heavily in the data they read
(good for caches), never in what they write.

### 2.3 Neurons within a tick are DATA-PARALLEL (synchronous update)

`processTick` (`:295-322`) computes all `nextNeuronValue[]` from the pre-tick values, then commits.
On GPU this is a double-buffered update inside one thread (no cross-thread sync needed when one
thread owns a whole window; see §5). Only the 46 non-input neurons need computing; input neurons
are pass-through. The AVX-512 engine further shrinks work with a **cone-of-influence filter**
(ref `:267-299`): only neurons in the closure of {signal, output} under the neighbor relation can
affect the result — skipping the rest is bit-exact (comment at ref `:268-269`). Apply the same
filter at shader-generation time; worst case 46 live neurons.

### 2.4 Ticks are SEQUENTIAL

Recurrent dynamics; tick t+1 reads tick t's committed state. ~1701 ticks per window on the bench
task (measured; ≈ 672 feed ticks + ~1029 settle ticks, i.e., the signal neuron is UNKNOWN on ~40%
of ticks). This is the irreducible serial axis *inside* a window; the parallel axes are
windows (8088) × nonces (cohort size) × neurons (46, inside a thread).

---

## 3. Integer semantics that must be preserved (bit-exactness contract)

1. **Trit domain.** All neuron values and LUT entries ∈ {0,1,2}; UNKNOWN = 2 participates in LUT
   indexing as an ordinary value. LUT index = `t0 + 3·t1 + 9·t2` ∈ [0,26] with the *exact neighbor
   order* `neighborIndices[n·3+0], [n·3+1], [n·3+2]` (`:309-312`). Do not canonicalize/sort.
2. **Synchronous tick.** All non-input neurons must read pre-tick neighbor values (two-phase /
   double-buffered). Input neurons hold the externally written value through the tick (`:302-305`);
   since the driver rewrites all 18 input slots every tick (sample or UNKNOWN), this is equivalent
   to recomposing input bits into the next state each tick.
3. **Feed/settle/break protocol, in this order at each tick start** (`:345-364`):
   signal==UNKNOWN && fc≥672 → break (grade current state, **before** any further tick);
   signal==UNKNOWN && fc<672 → write `inputs[t+fc][i]` to the 18 input neurons, fc++;
   signal!=UNKNOWN → write UNKNOWN to the 18 input neurons. Then run the tick.
   fc saturates at exactly 672; `expected = outputs[t + fc][0] = outputs[t+672][0]` (`:374-375`).
4. **Timeout.** A window that has not broken after exactly 100000 loop iterations makes the whole
   `score()` return `0xFFFFFFFF` (`:369-372`). Settling *at* tick 100000 is a timeout. GPU must
   count iterations identically (the break iteration itself performs no `processTick`).
5. **Score type.** `unsigned int` (u32). Failure count ≤ 8088 so no wrap in practice, but the
   sentinel `0xFFFFFFFF` must survive verbatim (accept/rollback compares include it; note
   `INFINITE_ERROR <= INFINITE_ERROR` accepts).
6. **Mutation decode** (`:386-397`) — u64 arithmetic, then u8:
   `delta = seed & 1`; `flatIdx = (seed >> 1) % 1242` (**64-bit** unsigned modulo);
   `neuron = updatedNeuronIndices[flatIdx / 27]` (ascending non-input order); `line = flatIdx % 27`;
   `newTrit = (oldTrit + 1 + delta) % 3` in u8. The AVX-512 engine's dense-row remap
   (ref `:959-973`, `:994-1003`) shows the safe way to change storage layout: keep the *RNG index
   space* (updated-neuron position × 27) fixed and only remap the storage address.
7. **Accept rule** (`:470-478` with K=0 per `:453`): accept iff `r <= cur` (worse-*or-equal* never;
   equal **is** accepted). On reject, restore the **entire LUT** from the snapshot. `best` is the
   running minimum of accepted `cur`, seeded with the initial score (`:456-457`).
8. **L clamp** (`:442-450`): `L = clamp(nonce[1], 1, 10)`. Seeds are consumed at fixed stride 10:
   step s uses `mutationSeed[s·10 + i]` for i < L (`:465`); seeds with i ≥ L are drawn but unused.
9. **RNG hashes** (`:407-419`): `rootHash = K12(pubkey, 32→32)`;
   `searchHash = K12(pubkey‖nonce, 64→32)` **with combined[32..34] (= nonce[0..2]) zeroed** —
   nonces differing only in algo/L/K bytes share a mutation-seed stream.
10. **`random2`** (`score_common.h:59-105`): 8 independent 32-bit LCG streams `x[0..7]` initialized
    from the seed's 8 little-endian u32s. Output is produced in 64-byte segments; element i of each
    segment is a 64-bit read of the pool *at bit offset `x[i]`* over a little-endian u64 view:
    `base = (x>>3)>>3` (= x>>6, u64 index), `m = x & 63`,
    value = `m==0 ? pool64[base] : (pool64[base] >> m) | (pool64[base+1] << (64-m))`
    (`:82-97`; the m==0 special case avoids UB shift-by-64). Then
    `x = x·1664525 + 1013904223` **mod 2³²** (`:100`). Both draw sizes here (1728, 8000) are
    multiples of 64, so the padding buffer (`:65-67`) never truncates mid-segment.
11. **LUT init** (`:431`): `lutInit[n·27+line] % 3` — bytes 0..255 taken mod 3 (non-uniform; must
    match exactly). Rows are drawn for **all 64** neurons; the 18 input-neuron rows are never read.
12. **Pool generation** (`score_common.h:45-57`): Keccak-P1600 12-round squeeze; state = miningSeed
    (32 B) ‖ zeros; each permutation emits the full 200-byte state. `POOL_VEC_SIZE = (2³²+64)/8 =
    536,870,920 B`; padded to `POOL_VEC_PADDING_SIZE = 536,871,000 B` (2,684,355 × 200). The +64
    bits exist precisely so `pool64[base+1]` at the max LCG offset (`base+1 = 2²⁶`) stays in range.
13. **Endianness.** Everything is native little-endian (u64 pool views, u32 topology fields, u32 LCG
    seeding). Apple GPUs and arm64 are little-endian — safe, but do not introduce byte-swaps.
14. **Task unpack** (`task_file.h:105-126`): 5 trits/byte base-3 (`t0+3t1+9t2+27t3+81t4`, valid
    bytes < 243); unpack on CPU once; GPU sees only unpacked/repacked tables.

Non-requirements (safe to change): storage layout of LUT/state (per items 6's remap rule), window
evaluation order, per-window early exit once settled, evaluating all windows despite a timeout,
skipping cone-excluded neurons, skipping the never-read input-neuron LUT rows.

---

## 4. How initializeANN consumes the 512 MB pool

Per `score()`-independent init (`src/score_bpp9000.h:402-419` + `score_common.h`):

| Draw | Seed | Output | Size | Segments (64 B) | Pool reads |
|---|---|---|---|---|---|
| 1 (per **pubkey**) | `K12(pubkey)` | `initValue.lutInit` | 1728 B | 27 | 27×8 = 216 × (2 aligned u64) |
| 2 (per **nonce**) | `K12(pubkey‖nonce, [32..34]=0)` | `initValue.mutationSeed` | 8000 B | 125 | 125×8 = 1000 × (2 aligned u64) |

- Addresses are **not** sequential offsets: each of the 8 LCG streams walks
  `x ← x·1664525+1013904223 (mod 2³²)` and every value is a *bit* offset into the full 2³²-bit
  pool — uniformly scattered random access over all 512 MB. The pool must exist in full; there is
  no locality to exploit.
- Total per-nonce pool traffic: 1216 reads ≈ **19.5 KB touched** — negligible (<10 µs on CPU).
- **Consequence for the port: run all of §4 on the CPU.** K12, `random2`, LUT init (mod 3) and the
  per-step `mutate` are microseconds per nonce; only `score()` belongs on the GPU. The pool then
  never needs to be GPU-visible.
- Memory rule (8 GB machine): allocate the pool **once** — a single page-aligned
  512 MB (536,871,000 B rounded up to 16 KB pages) allocation, generated in-place (0.44 s). If GPU
  access is ever wanted later, wrap the *same* allocation with
  `newBufferWithBytesNoCopy:length:options:MTLResourceStorageModeShared deallocator:` — never copy
  or duplicate it. Free it only when the mining phase (miningSeed) changes, then regenerate in-place.

---

## 5. Proposed Metal decomposition

### 5.1 Mapping (mirrors the AVX-512 engine's "lane = window", ref `:456-534`)

- **One GPU thread = one window** of one candidate LUT. Threads never synchronize with each other
  (windows are independent, §2.2); each thread runs its own tick loop with private state.
- **One threadgroup = 256 threads** (8 SIMD-groups of 32) covering 256 consecutive windows of one
  candidate. 8088 windows → 32 threadgroups per candidate (last group 152 active threads,
  bounds-checked).
- **Grid = (32, C, 1) threadgroups**, where C = candidates in the dispatch = cohort of nonces at
  the same mutation step (recommend C = 32–64; see §5.4). One dispatch = one `score()` for every
  candidate. C=32 → 258,816 threads: saturates the 7-core GPU many times over.
- **Per-thread state (registers):** `cur` = 4×u32 packed trits (2 bits/neuron, UNKNOWN=0b10, reset
  pattern 0xAAAAAAAA), `nxt` accumulated the same way, `fc` (u16-in-u32), `tick` (u32), window id.
  No spills expected (~30 live u32 values) → full occupancy.
- **Runtime shader generation is the superpower here.** The Metal offline compiler is missing
  anyway, so we must compile with `device.makeLibrary(source:)` / `newLibraryWithSource:` — exploit
  it: **bake the task topology into the generated MSL source**. For each of the ≤46 live
  (cone-filtered, §2.3) neurons emit an unrolled update with *literal* bit-slot shifts for its 3
  neighbors and its own slot; bake the input mask (4×u32), the signal and output slots, W=672 and
  maxTicks=100000 as literals. Everything dynamically-indexed in the scalar code becomes
  constant-shift ALU; per-thread state stays entirely in registers. Regenerate + recompile once per
  task (topology/data change per epoch; compile cost ≲1 s, amortized over minutes of mining).

### 5.2 Buffers

| # | Buffer | Contents | Size | Mode | Update cadence |
|---|---|---|---|---|---|
| 0 | `feedTable` | per sample: 4×u32 input trits pre-positioned at their state bit-slots | 140,160 B | shared, `device const` | once per task |
| 1 | `expected` | u8[8088] = outputs[t+672][0] | 8088 B | shared | once per task |
| 2 | `lutRows` | C × 46 × {u32 lo, u32 hi} packed 2-bit LUT rows, dense by live-neuron order | C×368 B | shared | CPU-written every step |
| 3 | `results` | C × {atomic_uint failures, atomic_uint timeoutFlag} | C×8 B | shared | zeroed every step, read back |
| 4 | tiny uniforms (`setBytes`) | C, activeWindowCount | ~8 B | — | per dispatch |

No pool buffer (§4). Total GPU-resident footprint ≈ 150 KB + C·376 B — irrelevant next to the
512 MB CPU pool; the 8 GB budget is safe (well under 1 MB of Metal buffers plus the one shared pool
allocation on the CPU side).

### 5.3 Kernel body (generated MSL, sketch)

```
kernel void score(device const uint4* feed      [[buffer(0)]],
                  device const uchar* expected  [[buffer(1)]],
                  constant uint2*     lutRows   [[buffer(2)]],   // C*46 entries
                  device ResultSlot*  results   [[buffer(3)]],
                  uint2 tgid [[threadgroup_position_in_grid]],
                  uint  lid  [[thread_index_in_threadgroup]])
{
    threadgroup uint2 lut[46];                       // one candidate per threadgroup
    if (lid < 46) lut[lid] = lutRows[tgid.y*46 + lid];
    threadgroup_barrier(mem_flags::mem_threadgroup); // the ONLY barrier in the kernel
    uint w = tgid.x*256 + lid;                       // window id
    if (w >= 8088) return;
    uint4 cur = uint4(0xAAAAAAAA);                   // all-UNKNOWN
    uint fc = 0, tick = 0;
    uchar predicted; bool timedOut = true;
    for (; tick < 100000; ++tick) {
        uint sig = (cur[SIG_W] >> SIG_SH) & 3;       // baked constants
        if (sig == 2u) {
            if (fc >= 672u) { predicted = (cur[OUT_W] >> OUT_SH) & 3; timedOut = false; break; }
            cur = (cur & ~INPUT_MASK) | feed[w + fc]; ++fc;      // pre-positioned sample bits
        } else {
            cur = (cur & ~INPUT_MASK) | INPUT_UNKNOWN_BITS;      // constants
        }
        uint4 nxt = cur & INPUT_MASK;                // inputs pass through
        // 46 unrolled updates; for neuron k with baked neighbor slots (wA,sA)(wB,sB)(wC,sC), own slot (wk,sk):
        //   uint idx = ((cur[wA]>>sA)&3) + 3*((cur[wB]>>sB)&3) + 9*((cur[wC]>>sC)&3);
        //   uint s2  = idx << 1;
        //   uint t   = (idx < 16u) ? (lut[k].x >> s2) : (lut[k].y >> (s2 - 32u));
        //   nxt[wk] |= (t & 3u) << sk;
        cur = nxt;
    }
    if (timedOut) atomic_store_explicit(&results[tgid.y].timeout, 1u, memory_order_relaxed);
    else {
        uint fail = (predicted != expected[w]) ? 1u : 0u;
        fail = simd_sum(fail);                       // simdgroup reduce, then one atomic per simdgroup
        if (simd_is_first()) atomic_fetch_add_explicit(&results[tgid.y].failures, fail, memory_order_relaxed);
    }
}
```

Notes:
- LUT rows in **threadgroup memory** with uniform (baked-constant) per-update addresses: all 32
  lanes read the same address → banked-broadcast, effectively free; the compiler will typically
  hoist the loop-invariant loads into (uniform) registers anyway. Avoid u64/`ulong` LUT rows —
  64-bit shifts are emulated on the 32-bit ALUs; the {lo,hi}+select form stays in u32.
- Feed table access: thread w at feed step fc loads `feed[w+fc]` — neighboring lanes hit
  neighboring samples (lanes desynchronize by only a few fc, §5.5), so a threadgroup's working set
  is a ~15 KB sliding region of the 140 KB table: L1-resident.
- A thread that settles simply exits the loop; the sentinel accounting (timeout OR, failures sum)
  is done once per thread at the end. Result mapping on CPU:
  `score = timeout ? 0xFFFFFFFF : failures` — bit-exact per §3.4-5.
- Optional (bit-exact, only helps pathological LUTs): Brent cycle detection on the 128-bit state
  during no-feed stretches — if the autonomous state cycles while the signal is never UNKNOWN, the
  window provably times out; set the flag early instead of grinding to 100000 ticks. Ship v1
  without it; the measured tick ceiling (1863) shows healthy LUTs never go near the timeout.

### 5.4 Host driver loop (batching across nonces AND mutation steps)

```
per task:      unpack task; cone filter; generate + compile MSL; build feed/expected buffers.
per phase:     generate 512MB pool in-place (0.44 s, CPU).
per pubkey:    CPU: rootHash→random2→lutInit→L0 (dense 46-row form);
               score L0 once on GPU (C=1 dispatch) → score0   [shared by all nonces; bench value 5458]
per cohort of C nonces (all at step s together — legal because every nonce runs exactly 100 steps):
    CPU: per nonce, K12+random2 → its 1000 mutation seeds (8 KB); cur=score0, best=score0, lut=L0.
    for s in 0..99:
        for each nonce n: snapshot prev[n]; apply L mutations to lut[n]   (µs, CPU)
        write C×46 packed rows into lutRows; zero results; dispatch (32, C); commit.
        on completion: r[n] = timeout ? INF : failures;
                       accept iff r[n] <= cur[n] else lut[n]=prev[n]; best[n]=min(...)
    report best[n]; solution iff best[n] <= 6469.
```

- The chain step→step is sequential (§2.1), so hide the CPU turnaround by running **two cohorts A/B
  staggered**: encode B's dispatch while A's completion handler does accept/mutate. CPU work per
  step is ~C×(1.2 KB memcpy + ≤10 mutations) — microseconds; the pipeline bubble is <1%.
- With C=32, one dispatch ≈ 2.0×10¹⁰ neuron-updates (32×8088×1701×46) — hundreds of ms on M1
  (§6): dispatch overhead (~tens of µs) is noise. 100 dispatches per cohort.
- `waitUntilCompleted`/handler per step; use a shared `MTLBuffer` for results (unified memory read).

### 5.5 Divergence and occupancy

- Within a 32-lane simdgroup, per-window tick counts vary by only ~±6% (measured p10–p90
  1649–1754; lanes are *consecutive* windows, which correlate further). A simdgroup retires at its
  max lane ≈ **+5–10% overhead**. No lane compaction needed.
- The feed/no-feed branch is divergent per lane but both paths are ~4 ops (mask merge); negligible.
- Threadgroup memory: 368 B/threadgroup — occupancy limited only by registers; the packed-state
  design targets <64 regs/thread → maximum occupancy (M1: 24,576 resident threads; we offer 259k).

### 5.6 What NOT to do (considered and rejected)

- Threadgroup-per-window with thread-per-neuron: 2 barriers × 1701 ticks × 8088 windows of sync,
  46/64 lane utilization — orders of magnitude worse than zero-sync thread-per-window.
- The AVX-512 two-windows-per-byte nibble packing: its payoff comes from `permutexvar` doing 64
  table lookups per instruction; a GPU thread's LUT lookup is a variable shift, which cannot be
  shared across two windows in one u32. One window per thread is the right grain.
- Batching mutation steps of one nonce (illegal, §2.1) or speculating accept/reject (break-even).
- GPU-side random2/K12: legal but pointless (19.5 KB/nonce, CPU does it in µs) and would force the
  512 MB pool into the GPU address space.

---

## 6. Arithmetic intensity & bound analysis (7-core M1)

Workload per nonce (measured tick data): 101 evals × 8088 windows × ~1701 ticks ≈ **1.39×10⁹
window-ticks**, × 46 neuron updates ≈ **6.4×10¹⁰ neuron-updates**.

Per neuron-update in the packed-u32 design: 3 trit extracts (baked shifts) + base-3 index (2 mad)
+ two shifts/select/mask + OR-insert ≈ **12–16 u32 ALU ops** (fewer if the compiler emits Apple's
bitfield-extract ops). Plus ~30 ops/tick loop overhead (signal test, feed merge, counters) →
≈ 700–900 ALU ops per window-tick → ≈ **1.0–1.25×10¹² int ops per nonce** (initial-eval sharing
saves ~1%).

Throughput ceiling: 7 cores × 128 ALUs × ~1.28 GHz ≈ **1.15×10¹² int32 ops/s** peak.

Memory per dispatch (C=32): DRAM-new data ≈ feed table 140 KB + LUTs 12 KB + results — the hot
loop's loads are all L1/threadgroup-broadcast. Arithmetic intensity vs DRAM ≳ 10⁴ ops/byte.
**Verdict: decisively COMPUTE-BOUND (integer-ALU-bound).** The 68 GB/s memory system and the 512 MB
pool are irrelevant to steady-state throughput; the kernel's fate is decided by ops/tick and
occupancy, not bandwidth. (Corollary: the 2-bit packing and baked constant shifts are the
optimization, not cache tricks.)

Expected performance at 40–90% ALU efficiency (divergence ~+7%, dependent shift chains, loop
overhead): **≈1–3 s per nonce ⇒ ~20–60 nonces/min**, versus measured CPU scalar 127 s/nonce/thread
(~4 nonces/min on all 8 cores, thermally optimistic on 8 GB M1). Net **≈5–15× the whole CPU**, with
upside if bitfield extracts land. Treat as an estimate to be validated by a microbenchmark of
`processTick` ops/s before building the full driver (see §8 Q3).

---

## 7. Validation plan (bit-exactness)

1. Re-run `build/bench 2` to reconfirm scalar baselines (prior: nonce0=5320, nonce1=5282; initial
   eval measured now: 5458 shared).
2. Unit: GPU `score()` of the *initial* LUT vs scalar `initializeANN` return — must equal 5458 on
   the bench inputs. Debug variant of the kernel writes per-window `{predicted, tick}` to a buffer;
   diff all 8088 windows against an instrumented scalar run (the tickprobe harness in the session
   scratchpad already logs per-window ticks).
3. Integration: full 100-step driver for bench nonces 0..7 must reproduce the scalar `bench 8`
   scores exactly (same accept/reject trajectory ⇒ same best).
4. Edge cases: a hand-mutated LUT that forces a timeout (score 0xFFFFFFFF); L=1 and L=10 nonces;
   equal-score accept (r == cur must accept).
5. CPU-side RNG: compare `lutInit`/`mutationSeed` byte streams against the scalar for 3 nonces
   (they share code if the port links the same `score_common.h` — then this is free).

## 8. Open questions

1. **Tick distribution across many mutated LUTs**: measured only for the root LUT ±3 mutation
   steps (avg ≈1700, max 1863). If deep-in-search LUTs drift toward longer settle times or
   timeouts, the divergence and early-abort story changes. Cheap to measure: extend the tickprobe
   run to 100 steps and log per-eval max ticks.
2. **Compiler quality**: does `makeLibrary(source:)` on this Metal 4 runtime hoist the 46
   threadgroup-LUT loads into uniform registers, and does it emit bitfield-extract for the baked
   shifts? Decides the constant factor in §6; inspect with a GPU capture or ~30-line
   microbenchmark.
3. **Actual sustained int-ops/s** of the 7-core M1 on this dependent-shift mix (the 40–90%
   efficiency band in §6 is the biggest error bar).
4. **Cohort size C and threadgroup size**: 256 threads/group and C=32 are reasoned defaults; sweep
   {64,128,256} × C∈{8..64} once the kernel runs. Larger C raises latency per step (all nonces
   wait for the slowest dispatch) but improves occupancy tail.
5. **Thermals/memory pressure on the shared 8 GB machine**: sustained GPU load will throttle the
   package; the 512 MB pool plus user apps may swap. Consider a duty-cycle or QoS knob in the
   driver.
6. **Whether to also port the CPU path to NEON** (the AVX2 fallback at ref `:840-957` maps
   directly to 16-lane NEON `tbl` lookups): a possible ~5–10× CPU-side bonus that would change the
   GPU-vs-CPU calculus but not this kernel spec.
