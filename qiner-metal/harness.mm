// harness.mm — Apple GPU (Metal) port of the Qubic BPP9000 mining scorer.
//
// Host-side driver + bit-exactness harness. The scalar CPU reference
// (src/score_bpp9000_ref.h, a patched copy of qiner-macos/src/score_bpp9000.h)
// is linked in and its scores computed AT RUNTIME — nothing is hardcoded.
//
// Modes:
//   ./build/harness quick             fast unit tests (~5 s): initial-score GPU vs CPU,
//                                     per-window predicted-trit diff, timeout sentinel.
//   ./build/harness verify [N=2]      full bit-exactness: N bench nonces, GPU cohort
//                                     (100 lockstep mutation steps) vs CPU computeScore.
//   ./build/harness perf [C=8] [S=100] GPU throughput: cohort of C candidates, S steps.
//   ./build/harness mine [C=8]        continuous mining with live per-step output
//                                     (fresh nonces each round; Ctrl-C for clean summary).
//
// Build: see build.sh (clang++ .mm + runtime MSL compile; no offline metal toolchain).

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <chrono>
#include <csignal>
#include <initializer_list>

#include "src/score_bpp9000_ref.h"

using Clock = std::chrono::steady_clock;
static double secondsSince(Clock::time_point t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

using MinerT = score_bpp9000::Miner<
    score_bpp9000::NUMBER_OF_INPUT_NEURONS,
    score_bpp9000::NUMBER_OF_OUTPUT_NEURONS,
    score_bpp9000::SEQUENCE_LENGTH,
    score_bpp9000::WINDOW_WIDTH,
    score_bpp9000::MAX_NUMBER_OF_TICKS,
    score_bpp9000::NUMBER_OF_NEIGHBORS,
    score_bpp9000::POPULATION_THRESHOLD,
    score_bpp9000::NUMBER_OF_MUTATIONS,
    score_bpp9000::SOLUTION_THRESHOLD>;

static constexpr uint32_t POP        = 64;      // neurons
static constexpr uint32_t LUTN       = 27;      // LUT entries per neuron
static constexpr uint32_t LUT_BYTES  = POP * LUTN;   // 1728, full scalar-layout LUT
static constexpr uint32_t SEQ_LEN    = 8760;
static constexpr uint32_t WIN_W      = 672;
static constexpr uint32_t NUM_WIN    = SEQ_LEN - WIN_W;   // 8088
static constexpr uint32_t NUM_STEPS  = 100;     // mutation steps per nonce
static constexpr uint32_t MAXC       = 64;      // max candidates per dispatch
static constexpr uint32_t INF        = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// MSL code generation: bake the task topology into the template kernel.
// ---------------------------------------------------------------------------

static const char* comp(uint32_t n) { static const char* c[4] = {"x","y","z","w"}; return c[n >> 4]; }
static uint32_t     shft(uint32_t n) { return 2u * (n & 15u); }

static bool replaceOnce(std::string& s, const std::string& marker, const std::string& body)
{
    size_t p = s.find(marker);
    if (p == std::string::npos) return false;
    s.replace(p, marker.size(), body);
    return true;
}

static std::string generateMSL(const std::string& tmpl, const MinerT* m, uint32_t tgSize)
{
    const uint32_t nUpd = (uint32_t)m->numberOfUpdatedNeurons;

    // Input mask / input-unknown pattern (2 bits per input-neuron slot).
    uint32_t mask[4] = {0,0,0,0}, unk[4] = {0,0,0,0};
    for (uint32_t i = 0; i < score_bpp9000::NUMBER_OF_INPUT_NEURONS; ++i)
    {
        uint32_t n = m->inputNeuronIndices[i];
        mask[n >> 4] |= 3u << shft(n);
        unk [n >> 4] |= 2u << shft(n);
    }

    char buf[512];
    std::string cst;
    snprintf(buf, sizeof(buf),
        "#define TG_SIZE %uu\n"
        "#define NUM_UPDATED %uu\n"
        "#define NUM_WINDOWS %uu\n"
        "#define WINDOW_W %uu\n"
        "#define MAX_TICKS %lluu\n"
        "#define SIG_EXPR (cur.%s >> %uu)\n"
        "#define OUT_EXPR (cur.%s >> %uu)\n"
        "#define INPUT_MASK uint4(0x%08Xu, 0x%08Xu, 0x%08Xu, 0x%08Xu)\n"
        "#define INPUT_UNKNOWN uint4(0x%08Xu, 0x%08Xu, 0x%08Xu, 0x%08Xu)\n",
        tgSize, nUpd, NUM_WIN, WIN_W,
        (unsigned long long)score_bpp9000::MAX_NUMBER_OF_TICKS,
        comp(m->signalNeuronIndex), shft(m->signalNeuronIndex),
        comp(m->outputNeuronIndices[0]), shft(m->outputNeuronIndices[0]),
        mask[0], mask[1], mask[2], mask[3],
        unk[0], unk[1], unk[2], unk[3]);
    cst = buf;

    // One unrolled update per non-input neuron, dense row order = ascending
    // updatedNeuronIndices (the same index space the mutator uses).
    std::string upd;
    for (uint32_t k = 0; k < nUpd; ++k)
    {
        uint32_t n = (uint32_t)m->updatedNeuronIndices[k];
        uint32_t a = m->neighborIndices[n * 3 + 0];
        uint32_t b = m->neighborIndices[n * 3 + 1];
        uint32_t c = m->neighborIndices[n * 3 + 2];
        snprintf(buf, sizeof(buf),
            "        { const uint idx = ((cur.%s >> %uu) & 3u)"
            " + 3u * ((cur.%s >> %uu) & 3u)"
            " + 9u * ((cur.%s >> %uu) & 3u);\n"
            "          const uint s2 = idx << 1u;\n"
            "          nxt.%s |= ((((idx < 16u) ? lut[%u].x : lut[%u].y) >> (s2 & 31u)) & 3u) << %uu; }\n",
            comp(a), shft(a), comp(b), shft(b), comp(c), shft(c),
            comp(n), k, k, shft(n));
        upd += buf;
    }

    std::string src = tmpl;
    if (!replaceOnce(src, "//__CONSTANTS__", cst) ||
        !replaceOnce(src, "//__NEURON_UPDATES__", upd))
    {
        fprintf(stderr, "kernel.metal template markers missing\n");
        exit(1);
    }
    return src;
}

// ---------------------------------------------------------------------------
// GPU scorer
// ---------------------------------------------------------------------------

struct GpuScorer
{
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pso = nil;
    id<MTLBuffer> feedBuf = nil, expBuf = nil, lutBuf = nil, resBuf = nil, dbgBuf = nil;
    uint32_t tgSize = 256, numTg = 0;
    const MinerT* m = nullptr;

    // stats
    double gpuBusySec = 0.0;
    uint64_t dispatches = 0, evals = 0;

    bool init(id<MTLDevice> device, const MinerT* miner, const char* metalPath)
    {
        dev = device;
        m = miner;
        queue = [dev newCommandQueue];

        NSError* err = nil;
        NSString* tmpl = [NSString stringWithContentsOfFile:@(metalPath)
                                                   encoding:NSUTF8StringEncoding error:&err];
        if (!tmpl) { fprintf(stderr, "cannot read %s\n", metalPath); return false; }

        // Compile; drop threadgroup size if register pressure caps the pipeline.
        for (uint32_t tg : {256u, 128u, 64u, 32u})
        {
            tgSize = tg;
            std::string src = generateMSL(tmpl.UTF8String, m, tgSize);
            MTLCompileOptions* opts = [MTLCompileOptions new];
            auto t0 = Clock::now();
            id<MTLLibrary> lib = [dev newLibraryWithSource:@(src.c_str()) options:opts error:&err];
            if (!lib)
            {
                fprintf(stderr, "MSL compile FAILED: %s\n", err.localizedDescription.UTF8String);
                fprintf(stderr, "---- generated source ----\n%s\n", src.c_str());
                return false;
            }
            id<MTLFunction> fn = [lib newFunctionWithName:@"score_bpp9000"];
            pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!pso)
            {
                fprintf(stderr, "pipeline FAILED: %s\n", err.localizedDescription.UTF8String);
                return false;
            }
            if (pso.maxTotalThreadsPerThreadgroup >= tgSize)
            {
                printf("[gpu] compiled kernel in %.0f ms; tgSize=%u, maxThreads/tg=%lu, execWidth=%lu\n",
                       secondsSince(t0) * 1e3, tgSize,
                       (unsigned long)pso.maxTotalThreadsPerThreadgroup,
                       (unsigned long)pso.threadExecutionWidth);
                break;
            }
            fprintf(stderr, "[gpu] tgSize=%u exceeds pipeline max %lu, retrying smaller\n",
                    tgSize, (unsigned long)pso.maxTotalThreadsPerThreadgroup);
            pso = nil;
        }
        if (!pso) { fprintf(stderr, "no viable threadgroup size\n"); return false; }
        numTg = (NUM_WIN + tgSize - 1) / tgSize;

        const uint32_t nUpd = (uint32_t)m->numberOfUpdatedNeurons;

        // Static per-task buffers.
        feedBuf = [dev newBufferWithLength:SEQ_LEN * 16 options:MTLResourceStorageModeShared];
        uint32_t* fp = (uint32_t*)feedBuf.contents;
        memset(fp, 0, SEQ_LEN * 16);
        for (uint32_t t = 0; t < SEQ_LEN; ++t)
        {
            for (uint32_t i = 0; i < score_bpp9000::NUMBER_OF_INPUT_NEURONS; ++i)
            {
                uint32_t n = m->inputNeuronIndices[i];
                fp[t * 4 + (n >> 4)] |= ((uint32_t)m->inputs[t][i]) << shft(n);
            }
        }
        expBuf = [dev newBufferWithLength:NUM_WIN options:MTLResourceStorageModeShared];
        uint8_t* ep = (uint8_t*)expBuf.contents;
        for (uint32_t w = 0; w < NUM_WIN; ++w) ep[w] = m->outputs[w + WIN_W][0];

        lutBuf = [dev newBufferWithLength:MAXC * nUpd * 8 options:MTLResourceStorageModeShared];
        resBuf = [dev newBufferWithLength:MAXC * 8 options:MTLResourceStorageModeShared];
        dbgBuf = [dev newBufferWithLength:MAXC * NUM_WIN options:MTLResourceStorageModeShared];
        return true;
    }

    // Score C candidates (one score() eval each). luts = C consecutive
    // full-scalar-layout LUTs (64*27 bytes each). Results into out[0..C-1].
    bool score(const uint8_t* luts, uint32_t C, uint32_t* out)
    {
        const uint32_t nUpd = (uint32_t)m->numberOfUpdatedNeurons;
        // Pack dense 2-bit rows (row k = updatedNeuronIndices[k]).
        uint32_t* lp = (uint32_t*)lutBuf.contents;
        for (uint32_t cix = 0; cix < C; ++cix)
        {
            const uint8_t* lut = luts + (size_t)cix * LUT_BYTES;
            for (uint32_t k = 0; k < nUpd; ++k)
            {
                uint32_t n = (uint32_t)m->updatedNeuronIndices[k];
                uint32_t lo = 0, hi = 0;
                for (uint32_t e = 0; e < LUTN; ++e)
                {
                    uint32_t v = lut[n * LUTN + e];
                    if (e < 16) lo |= v << (2 * e);
                    else        hi |= v << (2 * (e - 16));
                }
                lp[(cix * nUpd + k) * 2 + 0] = lo;
                lp[(cix * nUpd + k) * 2 + 1] = hi;
            }
        }
        memset(resBuf.contents, 0, (size_t)C * 8);

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:feedBuf offset:0 atIndex:0];
        [enc setBuffer:expBuf offset:0 atIndex:1];
        [enc setBuffer:lutBuf offset:0 atIndex:2];
        [enc setBuffer:resBuf offset:0 atIndex:3];
        [enc setBuffer:dbgBuf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(numTg, C, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted)
        {
            fprintf(stderr, "GPU command buffer failed: status=%ld error=%s\n", (long)cb.status,
                    cb.error ? cb.error.localizedDescription.UTF8String : "(nil)");
            return false;
        }
        gpuBusySec += cb.GPUEndTime - cb.GPUStartTime;
        dispatches += 1;
        evals += C;

        const uint32_t* rp = (const uint32_t*)resBuf.contents;
        for (uint32_t cix = 0; cix < C; ++cix)
        {
            out[cix] = rp[cix * 2 + 1] ? INF : rp[cix * 2 + 0];
        }
        return true;
    }

    const uint8_t* debugPred(uint32_t cix) const
    {
        return (const uint8_t*)dbgBuf.contents + (size_t)cix * NUM_WIN;
    }
};

// ---------------------------------------------------------------------------
// CPU-side search bookkeeping (mutation decode + accept rule, exact replicas)
// ---------------------------------------------------------------------------

static void makeBenchNonce(int n, unsigned char nonce[32])
{
    memset(nonce, 0, 32);
    nonce[0] = 1;                               // AlgoType::Bpp9000
    nonce[1] = (unsigned char)(1 + (n % 10));   // L in [1,10]
    nonce[2] = 0;                               // K
    nonce[3] = (unsigned char)n;
    nonce[4] = 0x5A;
}

// mine mode: unique nonce per global index (64-bit counter spread over bytes 3..10)
static void makeMineNonce(uint64_t idx, unsigned char nonce[32])
{
    memset(nonce, 0, 32);
    nonce[0] = 1;                               // AlgoType::Bpp9000
    nonce[1] = (unsigned char)(1 + (idx % 10)); // L in [1,10]
    nonce[2] = 0;                               // K
    for (int b = 0; b < 8; ++b) nonce[3 + b] = (unsigned char)(idx >> (8 * b));
    nonce[11] = 0xA5;
}

static volatile sig_atomic_t g_stop = 0;
static void onStop(int) { g_stop = 1; }

struct Candidate
{
    uint8_t lut[LUT_BYTES];
    uint8_t prev[LUT_BYTES];
    uint64_t seeds[NUM_STEPS * score_bpp9000::MAX_LUT_ENTRIES_PER_STEP]; // 1000
    uint32_t L = 1;
    uint32_t cur = 0, best = 0;
};

static void deriveMutationSeeds(const MinerT* m, const unsigned char pubkey[32],
                                const unsigned char nonce[32], uint64_t* seeds)
{
    unsigned char combined[64];
    memcpy(combined, pubkey, 32);
    memcpy(combined + 32, nonce, 32);
    combined[32] = 0; combined[33] = 0; combined[34] = 0;   // algo/L/K knobs excluded from RNG
    unsigned char searchHash[32];
    KangarooTwelve(combined, 64, searchHash, 32);
    random2(searchHash, m->pool, (unsigned char*)seeds,
            NUM_STEPS * score_bpp9000::MAX_LUT_ENTRIES_PER_STEP * 8);
}

static void buildInitialLut(const MinerT* m, const unsigned char pubkey[32], uint8_t* lutOut)
{
    unsigned char rootHash[32];
    KangarooTwelve((unsigned char*)pubkey, 32, rootHash, 32);
    std::vector<uint8_t> lutInit(LUT_BYTES);
    random2(rootHash, m->pool, lutInit.data(), LUT_BYTES);
    for (uint32_t i = 0; i < LUT_BYTES; ++i) lutOut[i] = (uint8_t)(lutInit[i] % 3);
}

// Exact replica of Miner::mutate (u64 arithmetic).
static void applyMutation(const MinerT* m, uint8_t* lut, uint64_t seed)
{
    const uint64_t delta = seed & 1ull;
    const uint64_t totalLines = m->numberOfUpdatedNeurons * (uint64_t)LUTN;
    const uint64_t flatIdx = (seed >> 1) % totalLines;
    const uint64_t neuronIdx = m->updatedNeuronIndices[flatIdx / LUTN];
    const uint64_t line = flatIdx % LUTN;
    const uint8_t oldTrit = lut[neuronIdx * LUTN + line];
    lut[neuronIdx * LUTN + line] = (uint8_t)((oldTrit + 1 + delta) % 3);
}

// Full GPU-side hill climb for a cohort of C nonces advanced in lockstep,
// one mutation step per dispatch (legal: every nonce runs exactly 100 steps).
// Returns per-nonce best scores. steps < 100 allowed for perf probing.
static bool runCohortGPU(GpuScorer& gpu, const MinerT* m, const unsigned char pubkey[32],
                         const uint8_t* lut0, uint32_t score0,
                         const std::vector<std::array<unsigned char,32>>& nonces,
                         uint32_t steps, uint32_t* bestOut, bool verbose = false)
{
    const uint32_t C = (uint32_t)nonces.size();
    std::vector<Candidate> cand(C);
    for (uint32_t n = 0; n < C; ++n)
    {
        memcpy(cand[n].lut, lut0, LUT_BYTES);
        deriveMutationSeeds(m, pubkey, nonces[n].data(), cand[n].seeds);
        uint32_t L = nonces[n][1];
        if (L < 1) L = 1;
        if (L > score_bpp9000::MAX_LUT_ENTRIES_PER_STEP) L = score_bpp9000::MAX_LUT_ENTRIES_PER_STEP;
        cand[n].L = L;
        cand[n].cur = score0;
        cand[n].best = score0;
    }

    std::vector<uint8_t> luts((size_t)C * LUT_BYTES);
    std::vector<uint32_t> r(C);
    for (uint32_t s = 0; s < steps && !g_stop; ++s)
    {
        auto tStep = Clock::now();
        for (uint32_t n = 0; n < C; ++n)
        {
            memcpy(cand[n].prev, cand[n].lut, LUT_BYTES);
            for (uint32_t i = 0; i < cand[n].L; ++i)
            {
                applyMutation(m, cand[n].lut,
                              cand[n].seeds[s * score_bpp9000::MAX_LUT_ENTRIES_PER_STEP + i]);
            }
            memcpy(&luts[(size_t)n * LUT_BYTES], cand[n].lut, LUT_BYTES);
        }
        if (!gpu.score(luts.data(), C, r.data())) return false;
        for (uint32_t n = 0; n < C; ++n)
        {
            if (r[n] <= cand[n].cur)                       // K=0: accept worse never, equal yes
            {
                cand[n].cur = r[n];
            }
            else
            {
                memcpy(cand[n].lut, cand[n].prev, LUT_BYTES);
            }
            if (cand[n].cur < cand[n].best) cand[n].best = cand[n].cur;
        }
        if (verbose)
        {
            uint32_t mn = cand[0].best;
            for (uint32_t n = 1; n < C; ++n) if (cand[n].best < mn) mn = cand[n].best;
            printf("  [step %3u/%u] cohort-best=%u  (%.0f ms)\n",
                   s + 1, steps, mn, secondsSince(tStep) * 1e3);
        }
    }
    for (uint32_t n = 0; n < C; ++n) bestOut[n] = cand[n].best;
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string mode = (argc > 1) ? argv[1] : "verify";
    const char* taskPath = getenv("TASK") ? getenv("TASK") : "../qiner-macos/data/example_task_bpp9000.bin";
    const char* kernelPath = getenv("KERNEL") ? getenv("KERNEL") : "kernel.metal";

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
    printf("[gpu] device: %s, unified=%d\n", dev.name.UTF8String, (int)dev.hasUnifiedMemory);

    // --- ONE 512MB pool: page-aligned mmap, generated in place, shared zero-copy. ---
    const size_t page = (size_t)getpagesize();
    const size_t poolLen = (POOL_VEC_PADDING_SIZE + page - 1) & ~(page - 1);
    void* poolMem = mmap(nullptr, poolLen, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (poolMem == MAP_FAILED) { perror("mmap pool"); return 1; }

    unsigned char miningSeed[32];
    for (int i = 0; i < 32; ++i) miningSeed[i] = (unsigned char)(i + 1);
    auto t0 = Clock::now();
    generateRandom2Pool(miningSeed, (unsigned char*)poolMem);
    printf("[cpu] 512MB pool generated in %.2f s (page-aligned mmap @%p, len=%zu)\n",
           secondsSince(t0), poolMem, poolLen);

    // Wrap the SAME allocation as a no-copy shared MTLBuffer (unified memory).
    // The kernel never reads the pool (all RNG stays on CPU), but this proves the
    // pool is GPU-visible with zero duplication for future GPU-side use.
    id<MTLBuffer> poolBuf = [dev newBufferWithBytesNoCopy:poolMem length:poolLen
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    if (!poolBuf) { fprintf(stderr, "newBufferWithBytesNoCopy failed\n"); return 1; }
    printf("[gpu] pool wrapped no-copy: contents==mmap: %d\n", poolBuf.contents == poolMem);

    // --- CPU reference miner (uses the SAME pool; no second allocation). ---
    auto miner = std::make_unique<MinerT>();
    t0 = Clock::now();
    if (!miner->initialize((const unsigned char*)poolMem, taskPath))
    {
        fprintf(stderr, "task load failed: %s\n", taskPath);
        return 1;
    }
    printf("[cpu] task loaded in %.2f s; updatedNeurons=%llu, signal=%u, output=%u\n",
           secondsSince(t0), miner->numberOfUpdatedNeurons,
           miner->signalNeuronIndex, miner->outputNeuronIndices[0]);

    // --- GPU scorer: codegen + runtime compile + buffers. ---
    GpuScorer gpu;
    if (!gpu.init(dev, miner.get(), kernelPath)) return 1;

    unsigned char pubkey[32];
    for (int i = 0; i < 32; ++i) pubkey[i] = (unsigned char)(0xA0 + i);

    // Shared initial LUT + initial score (pubkey-only, shared by all nonces).
    std::vector<uint8_t> lut0(LUT_BYTES);
    buildInitialLut(miner.get(), pubkey, lut0.data());
    uint32_t score0 = 0;
    t0 = Clock::now();
    if (!gpu.score(lut0.data(), 1, &score0)) return 1;
    double gpuEval0 = secondsSince(t0);
    printf("[gpu] initial score (pubkey-only, shared): %u  (%.1f ms wall)\n", score0, gpuEval0 * 1e3);

    int failures = 0;

    if (mode == "quick" || mode == "verify")
    {
        // Unit 1: initial score, GPU vs CPU scalar; plus per-window predicted-trit diff.
        std::vector<uint8_t> gpuPred(NUM_WIN);
        memcpy(gpuPred.data(), gpu.debugPred(0), NUM_WIN);

        unsigned char nonce0[32];
        makeBenchNonce(0, nonce0);
        std::vector<uint8_t> cpuPred(NUM_WIN, 0xEE);
        miner->windowTrace = cpuPred.data();
        t0 = Clock::now();
        uint32_t cpu0 = miner->initializeANN(pubkey, nonce0);
        double cpuEvalSec = secondsSince(t0);   // includes K12+random2, <1% of an eval
        miner->windowTrace = nullptr;

        uint32_t winDiff = 0, firstDiff = INF;
        for (uint32_t w = 0; w < NUM_WIN; ++w)
        {
            if (gpuPred[w] != cpuPred[w]) { ++winDiff; if (firstDiff == INF) firstDiff = w; }
        }
        printf("[unit] initial score: gpu=%u cpu=%u -> %s\n", score0, cpu0,
               score0 == cpu0 ? "PASS" : "FAIL");
        printf("[unit] per-window predicted trits: %u/%u differ%s -> %s\n",
               winDiff, NUM_WIN,
               winDiff ? ([&]{ static char b[64]; snprintf(b, sizeof(b), " (first at window %u)", firstDiff); return (const char*)b; }()) : "",
               winDiff == 0 ? "PASS" : "FAIL");
        if (score0 != cpu0 || winDiff) ++failures;
        printf("[cpu] scalar score() eval: %.2f s\n", cpuEvalSec);

        // Unit 2: timeout sentinel. Clamp the signal neuron's LUT row to a constant
        // non-UNKNOWN trit: the signal never returns to UNKNOWN, no window can settle,
        // every window must hit MAX_TICKS => INFINITE_ERROR. (CPU cross-check is
        // infeasible: 100000 ticks x 8088 windows scalar; the sentinel path itself
        // is trivially INFINITE_ERROR by construction in the scalar code.)
        {
            std::vector<uint8_t> lutTO(lut0);
            for (uint32_t e = 0; e < LUTN; ++e) lutTO[miner->signalNeuronIndex * LUTN + e] = 0;
            uint32_t rto = 0;
            t0 = Clock::now();
            if (!gpu.score(lutTO.data(), 1, &rto)) return 1;
            printf("[unit] forced-timeout LUT: gpu=0x%08X (%.2f s) -> %s\n", rto,
                   secondsSince(t0), rto == INF ? "PASS" : "FAIL");
            if (rto != INF) ++failures;
        }

        if (mode == "quick")
        {
            printf(failures ? "QUICK: FAIL (%d)\n" : "QUICK: ALL PASS\n", failures);
            return failures ? 1 : 0;
        }
    }

    if (mode == "verify")
    {
        const int N = (argc > 2) ? atoi(argv[2]) : 2;
        std::vector<std::array<unsigned char,32>> nonces(N);
        for (int n = 0; n < N; ++n) makeBenchNonce(n, nonces[n].data());

        // GPU cohort: all N nonces in lockstep, 100 dispatches.
        printf("[gpu] running %d-nonce cohort, %u lockstep steps...\n", N, NUM_STEPS);
        std::vector<uint32_t> bestGpu(N);
        double gpuBusyBefore = gpu.gpuBusySec;
        t0 = Clock::now();
        if (!runCohortGPU(gpu, miner.get(), pubkey, lut0.data(), score0, nonces,
                          NUM_STEPS, bestGpu.data())) return 1;
        double cohortWall = secondsSince(t0);
        double cohortBusy = gpu.gpuBusySec - gpuBusyBefore;
        printf("[gpu] cohort done: wall %.2f s, GPU busy %.2f s, %.1f ms/dispatch, %.1f ms/eval\n",
               cohortWall, cohortBusy, cohortWall * 1e3 / NUM_STEPS,
               cohortBusy * 1e3 / (NUM_STEPS * N));

        // CPU reference, computed at runtime (never hardcoded).
        int pass = 0;
        double cpuTotal = 0;
        for (int n = 0; n < N; ++n)
        {
            printf("[cpu] reference computeScore nonce %d (this takes ~2 min)...\n", n);
            t0 = Clock::now();
            uint32_t ref = miner->computeScore(pubkey, nonces[n].data());
            double sec = secondsSince(t0);
            cpuTotal += sec;
            bool ok = (ref == bestGpu[n]);
            printf("nonce %d: gpu=%u cpu=%u (%.1f s cpu) -> %s\n",
                   n, bestGpu[n], ref, sec, ok ? "PASS" : "FAIL");
            if (ok) ++pass; else ++failures;
        }

        double cpuPerEval = cpuTotal / (N * 101.0);
        double gpuPerEvalWall = cohortWall / (NUM_STEPS * (double)N);
        printf("\n=== RESULTS ===\n");
        printf("bit-exact: %d/%d nonces %s\n", pass, N, pass == N ? "PASS" : "FAIL");
        printf("CPU scalar per score() eval : %.3f s\n", cpuPerEval);
        printf("GPU per score() eval (wall, C=%d cohort): %.4f s  (GPU-busy %.4f s)\n",
               N, gpuPerEvalWall, cohortBusy / (NUM_STEPS * (double)N));
        printf("speedup vs single CPU thread: %.1fx\n", cpuPerEval / gpuPerEvalWall);
        printf("est nonces/min at C=%d      : %.1f  (100 evals/nonce + shared initial eval)\n",
               N, N * 60.0 / cohortWall);
        return failures ? 1 : 0;
    }

    if (mode == "perf")
    {
        const uint32_t C = (argc > 2) ? (uint32_t)atoi(argv[2]) : 8;
        const uint32_t S = (argc > 3) ? (uint32_t)atoi(argv[3]) : NUM_STEPS;
        if (C < 1 || C > MAXC) { fprintf(stderr, "C must be 1..%u\n", MAXC); return 1; }
        std::vector<std::array<unsigned char,32>> nonces(C);
        for (uint32_t n = 0; n < C; ++n) makeBenchNonce((int)n, nonces[n].data());

        printf("[perf] cohort C=%u, %u lockstep steps...\n", C, S);
        std::vector<uint32_t> best(C);
        double busyBefore = gpu.gpuBusySec;
        t0 = Clock::now();
        if (!runCohortGPU(gpu, miner.get(), pubkey, lut0.data(), score0, nonces, S, best.data()))
            return 1;
        double wall = secondsSince(t0);
        double busy = gpu.gpuBusySec - busyBefore;
        printf("[perf] wall %.2f s, GPU busy %.2f s (util %.0f%%)\n", wall, busy, 100.0 * busy / wall);
        printf("[perf] per dispatch (C=%u evals): wall %.1f ms, busy %.1f ms\n",
               C, wall * 1e3 / S, busy * 1e3 / S);
        printf("[perf] per score() eval: wall %.2f ms, busy %.2f ms\n",
               wall * 1e3 / (S * (double)C), busy * 1e3 / (S * (double)C));
        printf("[perf] est nonces/min: %.1f  (C=%u, full 100-step nonces)\n",
               C * 60.0 / (wall * (NUM_STEPS / (double)S)), C);
        for (uint32_t n = 0; n < C && n < 8; ++n) printf("  best[%u]=%u\n", n, best[n]);
        return 0;
    }

    if (mode == "mine")
    {
        const uint32_t C = (argc > 2) ? (uint32_t)atoi(argv[2]) : 8;
        if (C < 1 || C > MAXC) { fprintf(stderr, "C must be 1..%u\n", MAXC); return 1; }
        signal(SIGINT, onStop);
        signal(SIGTERM, onStop);
        printf("[mine] continuous mining, cohort C=%u, %u steps/nonce — Ctrl-C to stop\n",
               C, NUM_STEPS);

        std::vector<std::array<unsigned char,32>> nonces(C);
        std::vector<uint32_t> best(C);
        uint64_t round = 0, noncesDone = 0;
        uint32_t allTimeBest = score0;
        auto tMine = Clock::now();
        while (!g_stop)
        {
            for (uint32_t n = 0; n < C; ++n)
                makeMineNonce(round * C + n, nonces[n].data());
            printf("[mine] round %llu (nonces %llu..%llu), initial=%u\n",
                   (unsigned long long)round,
                   (unsigned long long)(round * C),
                   (unsigned long long)(round * C + C - 1), score0);
            if (!runCohortGPU(gpu, miner.get(), pubkey, lut0.data(), score0, nonces,
                              NUM_STEPS, best.data(), /*verbose=*/true))
                return 1;
            if (g_stop) break;                       // partial round: don't count it
            uint32_t roundBest = best[0];
            for (uint32_t n = 1; n < C; ++n) if (best[n] < roundBest) roundBest = best[n];
            if (roundBest < allTimeBest) allTimeBest = roundBest;
            ++round;
            noncesDone += C;
            double mins = secondsSince(tMine) / 60.0;
            printf("[mine] round %llu done: round-best=%u  ALL-TIME BEST=%u  "
                   "total %llu nonces in %.1f min (%.1f nonces/min)\n",
                   (unsigned long long)(round - 1), roundBest, allTimeBest,
                   (unsigned long long)noncesDone, mins, noncesDone / mins);
        }
        double mins = secondsSince(tMine) / 60.0;
        printf("\n[mine] stopped. %llu nonces, %.1f min, %.1f nonces/min, all-time best=%u "
               "(lower is better; initial=%u)\n",
               (unsigned long long)noncesDone, mins,
               mins > 0 ? noncesDone / mins : 0.0, allTimeBest, score0);
        return 0;
    }

    fprintf(stderr, "unknown mode '%s' (use quick|verify|perf|mine)\n", mode.c_str());
    return 1;
}
