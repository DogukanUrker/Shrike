#!/bin/bash
# start a stock llama-server for one gguf, run the evals, stop it.
# usage: eval/run_all.sh <model.gguf> <run name> [mmproj.gguf]
# env: LLAMA_SERVER (default $LLAMA_CPP/build/bin/llama-server), PORT (default 18200), NGL (default 99)
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
MODEL=$1; NAME=$2; MMPROJ=${3:-}
LLAMA_SERVER=${LLAMA_SERVER:-${LLAMA_CPP:-$HERE/llama.cpp}/build/bin/llama-server}
PORT=${PORT:-18200}
MM=""; [ -n "$MMPROJ" ] && MM="--mmproj $MMPROJ"
"$LLAMA_SERVER" -m "$MODEL" $MM -ngl ${NGL:-99} -c 32768 -fa on -np 1 -b 4096 -ub 4096 --port $PORT --host 127.0.0.1 > "/tmp/run_all_$NAME.log" 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null" EXIT
for i in $(seq 180); do curl -s 127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && break; sleep 1; done
curl -s 127.0.0.1:$PORT/health | grep -q ok || { echo "server did not start, see /tmp/run_all_$NAME.log"; exit 1; }
EVALS="english multilingual massive"; [ -n "$MMPROJ" ] && EVALS="$EVALS doom"
python3 "$HERE/eval/run_evals.py" http://127.0.0.1:$PORT/v1/systemone "$NAME" $EVALS
