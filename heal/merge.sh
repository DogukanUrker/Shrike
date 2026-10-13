#!/bin/bash
# trained peft lora -> one stock gguf: convert_lora_to_gguf -> llama-export-lora into F16 -> llama-quantize (same type).
# usage: heal/merge.sh <run dir (has lora/)> <quant type, e.g. IQ1_S> <out.gguf>
# env: LLAMA_CPP, D1_HF, DM_WORK, IMATRIX (default $DM_WORK/models/decision.imatrix.gguf), F16 (default $DM_WORK/models/d1-3B-F16.gguf)
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
RUN=$1; Q=$2; OUTF=$3
LLAMA_CPP=${LLAMA_CPP:-$HERE/llama.cpp}; WORK=${DM_WORK:-$HERE/work}; D1_HF=${D1_HF:-$WORK/hf/d1-3B}
F16=${F16:-$WORK/models/d1-3B-F16.gguf}; IMATRIX=${IMATRIX:-$WORK/models/decision.imatrix.gguf}
BIN=$LLAMA_CPP/build/bin

# convert_lora_to_gguf wants a text-only base config (d1's config.json is the vision-language one)
TEXT=$WORK/hf/d1-3B-text; mkdir -p $TEXT
python3 - "$D1_HF" "$TEXT" <<'PY'
import json, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
c = json.load(open(f"{src}/config.json"))["text_config"]
c.update(architectures=["Lfm2ForCausalLM"], model_type="lfm2", tie_word_embeddings=True)
json.dump(c, open(f"{dst}/config.json", "w"), indent=1)
shutil.copy(f"{src}/tokenizer.json", dst)
PY
PYTHONPATH=$LLAMA_CPP:$LLAMA_CPP/gguf-py python3 $LLAMA_CPP/convert_lora_to_gguf.py $RUN/lora --base $TEXT --outtype f16 --outfile $RUN/adapter.gguf
$BIN/llama-export-lora -m $F16 --lora $RUN/adapter.gguf -o $RUN/merged-F16.gguf
IM=""; [ -f "$IMATRIX" ] && IM="--imatrix $IMATRIX"
$BIN/llama-quantize $IM $RUN/merged-F16.gguf "$OUTF" $Q
rm -f $RUN/merged-F16.gguf
echo "wrote $OUTF (adapter: $RUN/adapter.gguf)"
