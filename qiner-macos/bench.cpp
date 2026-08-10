// Offline correctness + perf harness for the arm64-ported Qiner kernel. No networking.
#include <cstdio>
#include <cstring>
#include <chrono>
#include <memory>
#include "score_bpp9000.h"

using Clock = std::chrono::steady_clock;

static double secondsSince(Clock::time_point t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    // 1) KangarooTwelve official test vectors (K12 draft/RFC, len(M)=0 and len(M)=1..17^i patterns).
    {
        unsigned char out[32];
        KangarooTwelve(nullptr, 0, out, 32);
        static const unsigned char expectEmpty[32] = {
            0x1a,0xc2,0xd4,0x50,0xfc,0x3b,0x42,0x05,0xd1,0x9d,0xa7,0xbf,0xca,0x1b,0x37,0x51,
            0x3c,0x08,0x03,0x57,0x7a,0xc7,0x16,0x7f,0x06,0xfe,0x2c,0xe1,0xf0,0xef,0x39,0xe5 };
        bool ok0 = memcmp(out, expectEmpty, 32) == 0;

        unsigned char msg17[17];
        for (int i = 0; i < 17; i++) msg17[i] = (unsigned char)(i % 251);
        KangarooTwelve(msg17, 17, out, 32);
        static const unsigned char expect17[32] = {
            0x6b,0xf7,0x5f,0xa2,0x23,0x91,0x98,0xdb,0x47,0x72,0xe3,0x64,0x78,0xf8,0xe1,0x9b,
            0x0f,0x37,0x12,0x05,0xf6,0xa9,0xa9,0x3a,0x27,0x3f,0x51,0xdf,0x37,0x12,0x28,0x88 };
        bool ok1 = memcmp(out, expect17, 32) == 0;
        printf("K12 vector empty: %s, 17-byte: %s\n", ok0 ? "PASS" : "FAIL", ok1 ? "PASS" : "FAIL");
        if (!ok0 || !ok1) return 1;
    }

    using Miner = score_bpp9000::Miner<
        score_bpp9000::NUMBER_OF_INPUT_NEURONS,
        score_bpp9000::NUMBER_OF_OUTPUT_NEURONS,
        score_bpp9000::SEQUENCE_LENGTH,
        score_bpp9000::WINDOW_WIDTH,
        score_bpp9000::MAX_NUMBER_OF_TICKS,
        score_bpp9000::NUMBER_OF_NEIGHBORS,
        score_bpp9000::POPULATION_THRESHOLD,
        score_bpp9000::NUMBER_OF_MUTATIONS,
        score_bpp9000::SOLUTION_THRESHOLD>;

    unsigned char miningSeed[32];
    for (int i = 0; i < 32; i++) miningSeed[i] = (unsigned char)(i + 1);

    auto miner = std::make_unique<Miner>();

    // 2) Time the one-off 512MB random2 pool generation (scalar Keccak-P1600-12).
    auto t0 = Clock::now();
    if (!miner->initialize(miningSeed, "data/example_task_bpp9000.bin"))
    {
        printf("task load failed\n");
        return 1;
    }
    double poolSec = secondsSince(t0);
    printf("pool init (512MB Keccak fill + task load): %.2f s\n", poolSec);

    // 3) Deterministic nonces through the full per-nonce pipeline (K12 + random2 + 101 ANN scores).
    unsigned char publicKey[32];
    for (int i = 0; i < 32; i++) publicKey[i] = (unsigned char)(0xA0 + i);

    const int N = (argc > 1) ? atoi(argv[1]) : 8;
    t0 = Clock::now();
    unsigned long long scoreSum = 0;
    for (int n = 0; n < N; n++)
    {
        unsigned char nonce[32];
        memset(nonce, 0, 32);
        nonce[0] = 1;                       // AlgoType::Bpp9000
        nonce[1] = (unsigned char)(1 + (n % 10)); // L in [1,10]
        nonce[2] = 0;                       // K
        nonce[3] = (unsigned char)n;
        nonce[4] = 0x5A;
        unsigned int score = 0;
        miner->findSolution(publicKey, nonce, score);
        scoreSum += score;
        printf("nonce %d: score=%u\n", n, score);
    }
    double sec = secondsSince(t0);
    printf("%d nonces in %.2f s -> %.3f it/s (score sum %llu)\n", N, sec, N / sec, scoreSum);
    return 0;
}
