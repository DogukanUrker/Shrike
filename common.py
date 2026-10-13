"""shared paths and helpers. every path can be overridden with an env var.

  D1_GGUF    any d1-3B gguf (only its chat template and vocab are read)   default: $DM_WORK/models/d1-3B-F16.gguf
  D1_HF      local copy of LiquidAI/d1-3B (config.json, tokenizer.json, model.safetensors)   default: $DM_WORK/hf/d1-3B
  LLAMA_CPP  llama.cpp checkout (gguf-py, convert_lora_to_gguf.py, build/bin/*)   default: ./llama.cpp
  DM_DATA    downloaded datasets (data/fetch_data.py)   default: ./data
  DM_WORK    everything generated: models, pools, runs, results   default: ./work"""
import glob, json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DM_DATA", os.path.join(ROOT, "data"))
WORK = os.environ.get("DM_WORK", os.path.join(ROOT, "work"))
D1_GGUF = os.environ.get("D1_GGUF", os.path.join(WORK, "models", "d1-3B-F16.gguf"))
D1_HF = os.environ.get("D1_HF", os.path.join(WORK, "hf", "d1-3B"))
LLAMA_CPP = os.environ.get("LLAMA_CPP", os.path.join(ROOT, "llama.cpp"))
sys.path.insert(0, os.path.join(LLAMA_CPP, "gguf-py"))
for sub in ["eval", "heal", "tools"]:
    sys.path.insert(0, os.path.join(ROOT, sub))


def rows(sub, name):
    """all rows of the pages data/fetch_data.py wrote for <sub>/<name>_<offset>.json, in offset order."""
    fs = sorted(glob.glob(f"{DATA}/{sub}/{name}_*.json"), key=lambda s: int(s.rsplit("_", 1)[1][:-5]))
    out = []
    for f in fs:
        try:
            out += [r["row"] for r in json.load(open(f))["rows"]]
        except Exception:
            pass
    return out


_tmpl = None


def systemone_template():
    """the systemone jinja template from the d1 gguf, the same one llama-server renders."""
    global _tmpl
    if _tmpl is None:
        import gguf, jinja2
        _tmpl = jinja2.Environment().from_string(gguf.GGUFReader(D1_GGUF).fields["tokenizer.chat_template.systemone"].contents())
    return _tmpl
