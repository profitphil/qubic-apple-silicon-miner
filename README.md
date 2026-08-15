# Qubic on Apple Silicon — arm64 Qiner port + first Metal GPU BPP9000 scorer

Working Qubic BPP9000 mining code for Apple Silicon, verified **bit-exact** against the
official scalar reference, plus a field-tested Ubuntu tuning guide for the official
[qubic-li client](https://github.com/qubic-li/client).

Built and measured on an Apple M1 (8 GB, 7-core GPU), 2026-08-09, against the BPP9000
algorithm live since Epoch 224.

## Quick start (run it on your Mac)

```sh
git clone https://github.com/profitphil/qubic-apple-silicon-miner
cd qubic-apple-silicon-miner
./install.sh                       # build + fetch task + self-test (needs Xcode CLT + Node 20+)
export QLI_TOKEN='<token from pool.qubic.li>'
./run.sh                           # DRY-RUN: connect + mine live jobs, submit nothing
./run.sh --live                    # submit shares to your account
```

Full guide, monitoring, and honest expectations: **[QUICKSTART.md](QUICKSTART.md)**.
Connects to the official qubic.li pool via Stratum; shares credit your account. On a base
M-series GPU this is a proof-of-concept (shares are unlikely — see QUICKSTART); larger Apple
GPUs scale roughly linearly.

## What's here

| Directory | Contents |
|---|---|
| [`qiner-metal/`](qiner-metal/) | **First Apple-GPU (Metal) implementation of the BPP9000 scorer.** Runtime-compiled MSL kernel (no offline Metal toolchain needed), cohort-lockstep hill-climb driver, bit-exactness harness, benchmarks. |
| [`qiner-macos/`](qiner-macos/) | arm64/macOS port of the official Qiner (scalar BPP9000 + K12), used as the CPU reference. Includes an offline bench harness with a bundled example task. |
| [`docs/METAL-KERNEL-SPEC.md`](docs/METAL-KERNEL-SPEC.md) | Full kernel specification: data layouts, loop-nest parallelism analysis, bit-exactness contract, GPU decomposition design. |
| [`UBUNTU-OPTIMIZATION.md`](UBUNTU-OPTIMIZATION.md) | Tuning playbook for the official qli client on Ubuntu (hugepages, worker-variant pinning, thread counts, NUMA, epoch ops), with [`rig-diagnose.sh`](rig-diagnose.sh) / [`rig-tune.sh`](rig-tune.sh). |
| [`docs/reference/`](docs/reference/) | Unmodified reference headers from [qubic/core](https://github.com/qubic/core) (see attribution note therein). |

## Headline results (measured, M1)

| | CPU scalar (1 thread) | Metal GPU (7-core M1) |
|---|---|---|
| per `score()` eval | 1.21–1.26 s | **29.8 ms** (C=2 cohort) |
| speedup | 1× | **40.5×** per thread ≈ 5–6× the whole 8-core CPU |
| sustained rate | ~0.5 nonces/min | **~20 nonces/min** |

Bit-exactness (all verified on a clean rebuild, references computed at runtime by the
linked scalar kernel — nothing hardcoded): initial score 5458 (pubkey-only, shared across
nonces), nonce 0 → 5320, nonce 1 → 5282 after full 100-mutation hill-climbs, 0/8088
per-window prediction mismatches, verbatim `0xFFFFFFFF` timeout sentinel.

```sh
cd qiner-metal && ./build.sh
./build/harness quick        # ~8 s unit tests
./build/harness verify 2     # ~5 min full bit-exactness (CPU reference dominates)
./build/harness perf 32 100  # GPU cohort throughput
```

## Status & known gaps

- **Not pool-wired.** The qli custom-runner interface was removed in client v3.0, so there
  is currently no sanctioned way to connect a third-party miner to the qubic.li pool. The
  engine here is an offline scorer + (in `qiner-macos/`) the original Qiner node protocol.
- **arm64↔x86 cross-verification pending.** GPU↔CPU is bit-exact on Apple Silicon; the
  arm64↔x86 cross-check (required before submitting real solutions, since x86 verifiers
  must reproduce scores) has not been run yet.
- **Algorithm rotation.** Qubic rotates the training algorithm every 2–3 months; this
  implements BPP9000 (Epoch 224+). The spec documents the porting method, not just the port.
- Cohort-size sweep, A/B double-buffered dispatch, and a NEON CPU path are listed
  follow-ups in [`qiner-metal/README.md`](qiner-metal/README.md).

## Attribution & license

`qiner-macos/` and the scorer semantics derive from the official
[qubic/core](https://github.com/qubic/core) sources; the original license of that project
applies to all derived files, and reference headers are reproduced unmodified in
`docs/reference/` for review convenience. The Metal kernel, harness, spec, and Ubuntu
tooling are original work intended for review by the Qubic developer community.
