// kernel.metal — BPP9000 scorer, Apple GPU (Metal) port. TEMPLATE.
//
// This file is loaded at runtime by harness.mm, which substitutes the two
// marker lines below with task-specific generated code (the offline Metal
// compiler is missing on this machine, so runtime compilation via
// newLibraryWithSource: is mandatory — we exploit it by baking the task
// topology into the source as literal constants):
//
//   CONSTANTS marker      : #defines TG_SIZE, NUM_UPDATED, NUM_WINDOWS,
//                           WINDOW_W, MAX_TICKS, SIG_EXPR, OUT_EXPR,
//                           INPUT_MASK, INPUT_UNKNOWN
//   NEURON UPDATES marker : one unrolled two-phase update per non-input
//                           neuron, with literal bit-slot shifts for its 3
//                           neighbors (exact neighbor order preserved:
//                           idx = t0 + 3*t1 + 9*t2)
//
// Decomposition: one thread = one window (8088 independent windows per
// score() eval); one threadgroup = TG_SIZE consecutive windows of one
// candidate; grid = (ceil(8088/TG_SIZE), C) for a cohort of C candidates.
// Neuron state: 64 trits packed 2 bits each into a uint4 (neuron n lives in
// component n>>4 at bit 2*(n&15)); UNKNOWN=2 -> reset pattern 0xAAAAAAAA.
// LUT rows: 27 entries x 2 bits packed into {lo,hi} u32 pairs (entries 0-15
// in lo, 16-26 in hi) staged once per threadgroup in threadgroup memory.

#include <metal_stdlib>
using namespace metal;

struct ResultSlot
{
    atomic_uint failures;
    atomic_uint timeoutFlag;
};

//__CONSTANTS__

kernel void score_bpp9000(
    device const uint4*  feed     [[buffer(0)]],  // per sample: input trits pre-positioned at their state bit-slots
    device const uchar*  expected [[buffer(1)]],  // expected[w] = outputs[w+WINDOW_W][0]
    device const uint2*  lutRows  [[buffer(2)]],  // C * NUM_UPDATED packed 2-bit LUT rows
    device ResultSlot*   results  [[buffer(3)]],  // per candidate
    device uchar*        dbgPred  [[buffer(4)]],  // per (candidate,window) predicted trit; 255 = timeout
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  lid  [[thread_index_in_threadgroup]])
{
    threadgroup uint2 lut[NUM_UPDATED];
    if (lid < NUM_UPDATED)
    {
        lut[lid] = lutRows[tgid.y * NUM_UPDATED + lid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // the only barrier in the kernel

    const uint w = tgid.x * TG_SIZE + lid;             // window id
    if (w >= NUM_WINDOWS)
    {
        return;
    }

    uint4 cur = uint4(0xAAAAAAAAu);                    // all 64 neurons = UNKNOWN
    uint fc = 0u;                                      // feed counter, saturates at WINDOW_W
    uint predicted = 0u;
    bool timedOut = true;

    for (uint tick = 0u; tick < MAX_TICKS; ++tick)
    {
        const uint sig = (SIG_EXPR) & 3u;
        if (sig == 2u)                                 // signal is UNKNOWN
        {
            if (fc >= WINDOW_W)                        // settled: grade BEFORE any further tick
            {
                predicted = (OUT_EXPR) & 3u;
                timedOut = false;
                break;
            }
            cur = (cur & ~INPUT_MASK) | feed[w + fc];  // feed sample fc
            ++fc;
        }
        else
        {
            cur = (cur & ~INPUT_MASK) | INPUT_UNKNOWN; // inputs go UNKNOWN while the net is busy
        }

        // Two-phase synchronous tick: every update below reads `cur` (pre-tick
        // values) and accumulates into `nxt`; input neurons pass through.
        uint4 nxt = cur & INPUT_MASK;
//__NEURON_UPDATES__
        cur = nxt;
    }

    dbgPred[tgid.y * NUM_WINDOWS + w] = timedOut ? (uchar)255 : (uchar)predicted;

    if (timedOut)
    {
        // Any timed-out window poisons the whole score with INFINITE_ERROR;
        // evaluating the remaining windows anyway is bit-identical.
        atomic_store_explicit(&results[tgid.y].timeoutFlag, 1u, memory_order_relaxed);
    }
    else
    {
        const uint fail = (predicted != (uint)expected[w]) ? 1u : 0u;
        const uint sum = simd_sum(fail);
        if (simd_is_first())
        {
            atomic_fetch_add_explicit(&results[tgid.y].failures, sum, memory_order_relaxed);
        }
    }
}
