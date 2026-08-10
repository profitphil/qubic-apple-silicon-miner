// stress.mm — probes for the Qubic Metal port:
//  1. LUT-style integer inner loop throughput (uchar table lookups in threadgroup mem)
//  2. a deliberately multi-second kernel to observe command-buffer / watchdog behavior
//  3. optional out-of-bounds kernel to demo MTL_SHADER_VALIDATION (run with argv[1]=oob)
// Build: clang++ -std=c++17 -fobjc-arc -O2 stress.mm -framework Metal -framework Foundation -o stress
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstring>
#include <initializer_list>

static const char *kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint kIters [[function_constant(0)]];

// Mimics the hot loop of the BPP9000 kernel: dependent uchar LUT lookups from
// threadgroup memory + mod-3-ish index math, one simdgroup per "neuron row".
kernel void lutloop(device const uchar *lutsIn [[buffer(0)]],
                    device uint *out            [[buffer(1)]],
                    uint gid [[thread_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint tgsize [[threads_per_threadgroup]])
{
    threadgroup uchar lut[1728];              // 64 neurons x 27 entries
    for (uint i = lid; i < 1728; i += tgsize) lut[i] = lutsIn[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint state = gid * 2654435761u;
    uchar t = (uchar)(gid % 3);
    for (uint i = 0; i < kIters; i++) {
        // 27-entry LUT index from three trits: t0*9 + t1*3 + t2 (no integer division)
        uint idx = (uint)t * 9u + (state % 3u) * 3u + ((state >> 8) % 3u);
        t = lut[((i & 63u) * 27u) + idx];     // dependent load, like recurrent ANN step
        state = state * 1664525u + 1013904223u + t;
    }
    out[gid] = state;
}

kernel void oob(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]])
{
    out[gid + (1u << 20)] = gid;   // way past the 256-byte buffer we bind
}
)MSL";

static double runIters(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib,
                       uint32_t iters, uint32_t nthreads) {
    NSError *err = nil;
    MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
    [fc setConstantValue:&iters type:MTLDataTypeUInt atIndex:0];
    id<MTLFunction> fn = [lib newFunctionWithName:@"lutloop" constantValues:fc error:&err];
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) { fprintf(stderr, "pso: %s\n", err.localizedDescription.UTF8String); return -1; }

    id<MTLBuffer> luts = [dev newBufferWithLength:1728 options:MTLResourceStorageModeShared];
    uint8_t *pl = (uint8_t *)luts.contents;
    for (int i = 0; i < 1728; i++) pl[i] = (uint8_t)(i % 3);
    id<MTLBuffer> out = [dev newBufferWithLength:nthreads * 4 options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:luts offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(nthreads,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "cb status=%ld err=%s\n", (long)cb.status,
                cb.error ? cb.error.localizedDescription.UTF8String : "(none)");
        return -1;
    }
    return cb.GPUEndTime - cb.GPUStartTime;
}

int main(int argc, char **argv) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                           options:nil error:&err];
    if (!lib) { fprintf(stderr, "compile: %s\n", err.localizedDescription.UTF8String); return 1; }

    if (argc > 1 && !strcmp(argv[1], "oob")) {
        id<MTLFunction> fn = [lib newFunctionWithName:@"oob"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        id<MTLBuffer> out = [dev newBufferWithLength:256 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:out offset:0 atIndex:0];
        [enc dispatchThreads:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
        printf("oob kernel status=%ld error=%s\n", (long)cb.status,
               cb.error ? cb.error.localizedDescription.UTF8String : "(none)");
        return 0;
    }

    if (argc > 2 && !strcmp(argv[1], "long")) {
        uint32_t iters = (uint32_t)atoi(argv[2]);
        double s = runIters(dev, q, lib, iters, 8192);
        printf("long run iters=%u -> %s, gpu=%.2f s\n", iters, s < 0 ? "FAILED" : "OK", s);
        return s < 0 ? 1 : 0;
    }

    const uint32_t NT = 8192;   // enough threads to saturate 7 GPU cores
    // Warmup + scaling runs
    for (uint32_t iters : {1000u, 100000u, 1000000u, 4000000u}) {
        double s = runIters(dev, q, lib, iters, NT);
        if (s < 0) return 1;
        double lookups = (double)iters * NT;
        printf("iters=%8u  gpu=%8.3f s   %8.2f G dependent-LUT-steps/s\n",
               iters, s, lookups / s / 1e9);
    }
    printf("DONE\n");
    return 0;
}
