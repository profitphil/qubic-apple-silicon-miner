// Minimal x86-intrinsics compatibility shim for Apple Silicon / aarch64 builds.
// Covers the only intrinsics Qiner actually uses outside of MSVC-specific code:
//   __m256i (used purely as a 32-byte copy unit), _addcarry_u64, _subborrow_u64,
//   _andn_u64, _rdrand32_step, _rdrand64_step.
// (_umul128 / __shiftleft128 / __shiftright128 already have portable
//  definitions inside K12AndKeyUtil.h for non-MSVC compilers.)
#pragma once
#if defined(__aarch64__) || defined(__arm64__)

#include <cstdint>
#include <cstdlib>

typedef struct alignas(32) __m256i_shim { uint64_t v[4]; } __m256i;

static inline unsigned char _addcarry_u64(unsigned char c_in, unsigned long long a,
                                          unsigned long long b, unsigned long long* out)
{
    unsigned __int128 s = (unsigned __int128)a + b + c_in;
    *out = (unsigned long long)s;
    return (unsigned char)(s >> 64);
}

static inline unsigned char _subborrow_u64(unsigned char b_in, unsigned long long a,
                                           unsigned long long b, unsigned long long* out)
{
    unsigned __int128 d = (unsigned __int128)a - b - b_in;
    *out = (unsigned long long)d;
    return (unsigned char)((d >> 64) & 1);
}

static inline unsigned long long _andn_u64(unsigned long long a, unsigned long long b)
{
    return ~a & b;
}

static inline int _rdrand32_step(unsigned int* p)
{
    *p = arc4random();
    return 1;
}

static inline int _rdrand64_step(unsigned long long* p)
{
    arc4random_buf(p, sizeof(*p));
    return 1;
}

#else
#include <immintrin.h>
#endif
