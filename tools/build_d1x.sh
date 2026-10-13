#!/bin/bash
# build tools/d1x against a llama.cpp checkout built with static libs:
#   cmake -B build -DBUILD_SHARED_LIBS=OFF [-DGGML_CUDA=ON] && cmake --build build -j --target llama-server mtmd
# usage: tools/build_d1x.sh   (env LLAMA_CPP, CUDA=1 to link the cuda backend)
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd); LLAMA_CPP=${LLAMA_CPP:-$HERE/llama.cpp}; B=$LLAMA_CPP/build
LIBS="$B/tools/mtmd/libmtmd.a $B/vendor/hash/libvendor-hash.a $B/src/libllama.a $B/ggml/src/libggml.a"
EXTRA=""
if [ "${CUDA:-0}" = 1 ]; then
    LIBS="$LIBS $B/ggml/src/ggml-cuda/libggml-cuda.a"
    EXTRA="-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lnccl -lcudart_static -lcublas_static -lcublasLt_static -lculibos -lcuda -lrt"
fi
g++ -O2 -std=c++17 -w $HERE/tools/d1x.cpp -I$LLAMA_CPP/include -I$LLAMA_CPP/ggml/include -I$LLAMA_CPP/tools/mtmd -o $HERE/tools/d1x \
    $LIBS $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml-base.a $EXTRA -fopenmp -lpthread -ldl
echo "built $HERE/tools/d1x"
