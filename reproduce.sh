#!/bin/bash
# end-to-end: data -> quantize -> heal -> merge -> evaluate. one gpu with 12 GB is enough (rtx 3060).
# usage: ./reproduce.sh [quant type, default IQ1_S] [steps, default 3000]
# needs: llama.cpp built with static libs (see README), python with torch + transformers 4.57 + peft + safetensors + gguf
#        + jinja2 + requests, huggingface-cli (or curl), optional: uv (frame collectors), vizdoom, gym-super-mario-bros.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd); cd $HERE
Q=${1:-IQ1_S}; STEPS=${2:-3000}
export LLAMA_CPP=${LLAMA_CPP:-$HERE/llama.cpp} DM_DATA=${DM_DATA:-$HERE/data} DM_WORK=${DM_WORK:-$HERE/work}
export D1_HF=${D1_HF:-$DM_WORK/hf/d1-3B} D1_GGUF=${D1_GGUF:-$DM_WORK/models/d1-3B-F16.gguf}
BIN=$LLAMA_CPP/build/bin; M=$DM_WORK/models; mkdir -p $M $D1_HF
step() { echo; echo "=== $*"; }

step "1. download the base model (hf weights for training, F16 gguf + mmproj for llama.cpp)"
for f in config.json tokenizer.json model.safetensors; do [ -f $D1_HF/$f ] || curl -L -o $D1_HF/$f https://huggingface.co/LiquidAI/d1-3B/resolve/main/$f; done
for f in d1-3B-F16.gguf mmproj-d1-3B-Q8_0.gguf; do [ -f $M/$f ] || curl -L -o $M/$f https://huggingface.co/LiquidAI/d1-3B-GGUF/resolve/main/$f; done

step "2. datasets (slow: the datasets server rate-limits) and frames"
python3 data/fetch_data.py $DM_DATA
command -v uv >/dev/null && { uv run frames/doom_collect.py 800 5000 || true; uv run frames/mario_collect.py || true; } || echo "no uv: skipping doom/mario frames"

step "3. prompts, imatrix, plain quant"
python3 heal/build_prompts.py
[ -f $M/decision.imatrix.gguf ] || { python3 -c "import json; print('\n\n'.join(json.load(open('$DM_WORK/pool/prompts.json'))[:4000]))" > $DM_WORK/calib.txt
    $BIN/llama-imatrix -m $M/d1-3B-F16.gguf -f $DM_WORK/calib.txt -o $M/decision.imatrix.gguf --chunks 400 -c 512 --no-ppl -ngl 99; }
[ -f $M/d1-3B-$Q.gguf ] || $BIN/llama-quantize --imatrix $M/decision.imatrix.gguf $M/d1-3B-F16.gguf $M/d1-3B-$Q.gguf $Q

step "4. image prompt embeddings (optional, needs tools/d1x)"
if [ -x tools/d1x ] || tools/build_d1x.sh; then
    python3 heal/build_images.py
    rm -f $DM_WORK/img/embd.bin
    tools/d1x $M/d1-3B-F16.gguf $M/mmproj-d1-3B-Q8_0.gguf < $DM_WORK/img/req.txt > /dev/null
else
    echo "no d1x: healing on text only (image answers will drift a little)"
fi

step "5. heal, merge, re-quantize"
RUN=$DM_WORK/runs/$Q; mkdir -p $RUN
python3 heal/heal.py $M/d1-3B-$Q.gguf $RUN $STEPS
heal/merge.sh $RUN $Q $M/d1-3B-$Q-healed.gguf

step "6. evaluate on a stock llama-server: F16, plain, healed"
eval/run_all.sh $M/d1-3B-F16.gguf F16 $M/mmproj-d1-3B-Q8_0.gguf
eval/run_all.sh $M/d1-3B-$Q.gguf $Q $M/mmproj-d1-3B-Q8_0.gguf
eval/run_all.sh $M/d1-3B-$Q-healed.gguf $Q-healed $M/mmproj-d1-3B-Q8_0.gguf
python3 eval/score.py $Q $Q-healed
