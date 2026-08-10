// hello.mm — minimal runtime-compiled Metal compute proof for the Qubic port.
// Build: clang++ -std=c++17 -fobjc-arc -O2 hello.mm -framework Metal -framework Foundation -o hello
// Proves: (1) newLibraryWithSource: works without the offline 'metal' toolchain,
//         (2) storageModeShared zero-copy via newBufferWithBytesNoCopy on page-aligned mmap memory,
//         (3) queries the device limits we care about for the kernel design.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <chrono>

static const char *kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Function constant: proves runtime specialization works (used later for
// WINDOW_WIDTH / batch-size specialization without recompiling source).
constant uint kMultiplier [[function_constant(0)]];

kernel void vadd(device const uchar  *a   [[buffer(0)]],
                 device const uchar  *b   [[buffer(1)]],
                 device       uchar  *out [[buffer(2)]],
                 uint gid [[thread_position_in_grid]])
{
    // uchar arithmetic wraps mod 256 (Metal integer ops are well-defined wraparound)
    out[gid] = (uchar)(a[gid] * kMultiplier + b[gid]);
}

// Second kernel: exercises threadgroup memory + simdgroup queries the way the
// real LUT kernel will (per-threadgroup LUT staging).
kernel void tg_probe(device uint *out [[buffer(0)]],
                     uint gid  [[thread_position_in_grid]],
                     uint lid  [[thread_position_in_threadgroup]],
                     uint simd_lane [[thread_index_in_simdgroup]],
                     uint simd_size [[threads_per_simdgroup]])
{
    threadgroup uchar lut[27 * 64];   // 1728 B: one 27-entry LUT per neuron, 64 neurons
    if (lid < 27*64) lut[lid] = (uchar)(lid % 3);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid == 0) out[0] = simd_size;
    if (gid == 1) out[1] = lut[26];
}
)MSL";

int main() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }

    printf("device                     : %s\n", dev.name.UTF8String);
    printf("hasUnifiedMemory           : %d\n", (int)dev.hasUnifiedMemory);
    printf("recommendedMaxWorkingSet   : %.1f MB\n", dev.recommendedMaxWorkingSetSize / 1048576.0);
    printf("maxBufferLength            : %.1f MB\n", dev.maxBufferLength / 1048576.0);
    printf("maxThreadgroupMemoryLength : %lu bytes\n", (unsigned long)dev.maxThreadgroupMemoryLength);
    MTLSize mt = dev.maxThreadsPerThreadgroup;
    printf("maxThreadsPerThreadgroup   : %lu x %lu x %lu\n",
           (unsigned long)mt.width, (unsigned long)mt.height, (unsigned long)mt.depth);

    // ---- 1. Runtime shader compilation (no offline toolchain) ----
    NSError *err = nil;
    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.mathMode = MTLMathModeFast;   // (fastMathEnabled deprecated); irrelevant for int work but harmless
    auto t0 = std::chrono::steady_clock::now();
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                           options:opts error:&err];
    auto t1 = std::chrono::steady_clock::now();
    if (!lib) { fprintf(stderr, "compile FAILED: %s\n", err.localizedDescription.UTF8String); return 1; }
    printf("runtime compile            : OK (%.1f ms)\n",
           std::chrono::duration<double,std::milli>(t1-t0).count());

    // Specialize function constant at pipeline-creation time
    MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
    uint32_t mult = 3;
    [fc setConstantValue:&mult type:MTLDataTypeUInt atIndex:0];
    id<MTLFunction> fnAdd = [lib newFunctionWithName:@"vadd" constantValues:fc error:&err];
    if (!fnAdd) { fprintf(stderr, "fn constants FAILED: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLFunction> fnProbe = [lib newFunctionWithName:@"tg_probe"];

    id<MTLComputePipelineState> psoAdd = [dev newComputePipelineStateWithFunction:fnAdd error:&err];
    id<MTLComputePipelineState> psoProbe = [dev newComputePipelineStateWithFunction:fnProbe error:&err];
    if (!psoAdd || !psoProbe) { fprintf(stderr, "pso FAILED: %s\n", err.localizedDescription.UTF8String); return 1; }
    printf("threadExecutionWidth       : %lu\n", (unsigned long)psoAdd.threadExecutionWidth);
    printf("maxTotalThreadsPerTG(vadd) : %lu\n", (unsigned long)psoAdd.maxTotalThreadsPerThreadgroup);

    // ---- 2. Zero-copy pool buffer: page-aligned mmap + newBufferWithBytesNoCopy ----
    const size_t page = (size_t)getpagesize();
    printf("page size                  : %zu\n", page);
    const size_t N = 1u << 20;               // 1 MiB stand-in for the 512 MB pool
    size_t len = (N + page - 1) & ~(page - 1); // length must also be page-multiple
    void *poolMem = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                         MAP_ANON | MAP_PRIVATE, -1, 0);
    if (poolMem == MAP_FAILED) { perror("mmap"); return 1; }
    uint8_t *pa = (uint8_t *)poolMem;
    for (size_t i = 0; i < N; i++) pa[i] = (uint8_t)(i * 7);

    id<MTLBuffer> bufA = [dev newBufferWithBytesNoCopy:poolMem length:len
                              options:MTLResourceStorageModeShared
                              deallocator:^(void *p, NSUInteger l){ munmap(p, l); }];
    if (!bufA) { fprintf(stderr, "newBufferWithBytesNoCopy FAILED\n"); return 1; }
    printf("noCopy shared buffer       : OK (contents=%p, orig=%p, same=%d)\n",
           bufA.contents, poolMem, bufA.contents == poolMem);

    id<MTLBuffer> bufB = [dev newBufferWithLength:N options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufO = [dev newBufferWithLength:N options:MTLResourceStorageModeShared];
    uint8_t *pb = (uint8_t *)bufB.contents;
    for (size_t i = 0; i < N; i++) pb[i] = (uint8_t)(i);

    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:psoAdd];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    // Non-uniform threadgroup dispatch (Apple GPU supports it): exact N threads
    [enc dispatchThreads:MTLSizeMake(N,1,1)
        threadsPerThreadgroup:MTLSizeMake(psoAdd.threadExecutionWidth * 4, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "cb error: %s\n", cb.error.localizedDescription.UTF8String); return 1;
    }
    printf("GPU time (vadd 1MiB)       : %.3f ms\n", (cb.GPUEndTime - cb.GPUStartTime) * 1e3);

    // CPU-side verify (zero-copy: read result straight from contents, no blit)
    uint8_t *po = (uint8_t *)bufO.contents;
    size_t bad = 0;
    for (size_t i = 0; i < N; i++) {
        uint8_t expect = (uint8_t)(pa[i] * 3 + pb[i]);
        if (po[i] != expect) { if (bad < 3) fprintf(stderr, "mismatch @%zu got %u want %u\n", i, po[i], expect); bad++; }
    }
    printf("vadd verify                : %s (%zu bad)\n", bad ? "FAIL" : "PASS", bad);

    // CPU writes into the noCopy buffer are visible to GPU next dispatch (shared)
    pa[0] = 200; pb[0] = 10;
    cb = [q commandBuffer]; enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:psoAdd];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufO offset:0 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    printf("coherency after CPU write  : %s (got %u, want %u)\n",
           po[0] == (uint8_t)(200*3+10) ? "PASS" : "FAIL", po[0], (uint8_t)(200*3+10));

    // ---- 3. simdgroup width probe from inside a kernel ----
    id<MTLBuffer> bufP = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    cb = [q commandBuffer]; enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:psoProbe];
    [enc setBuffer:bufP offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(psoProbe.maxTotalThreadsPerThreadgroup,1,1)
        threadsPerThreadgroup:MTLSizeMake(psoProbe.maxTotalThreadsPerThreadgroup,1,1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    uint32_t *pp = (uint32_t *)bufP.contents;
    printf("threads_per_simdgroup (GPU): %u, tg-mem lut[26]=%u\n", pp[0], pp[1]);

    printf("ALL OK\n");
    return 0;
}
