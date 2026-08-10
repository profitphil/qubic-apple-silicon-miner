#!/bin/zsh
# build.sh — clean checkout -> runnable harness. No offline Metal toolchain needed:
# the shader (kernel.metal) is compiled AT RUNTIME via newLibraryWithSource:.
set -e
cd "$(dirname "$0")"
mkdir -p build
clang++ -std=c++17 -fobjc-arc -O3 \
    -I ../qiner-macos/src \
    harness.mm \
    -framework Metal -framework Foundation \
    -o build/harness
echo "built build/harness"
echo "run from qiner-metal/:  ./build/harness quick | verify [N] | perf [C] [steps]"
