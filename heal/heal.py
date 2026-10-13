"""heal a low-bit d1-3B gguf with a LoRA adapter (runs on stock llama.cpp with --lora).
student = d1-3B text model whose linear weights are the dequantized low-bit gguf weights (what llama.cpp computes),
plus LoRA on every linear. teacher = full bf16 d1-3B. loss = KL to the teacher's top-k next-token distribution at the
answer position (covers yes/no, option letters, score digits).
usage: heal.py <low-bit gguf> <out dir> [steps]
env: PROMPTS (default $DM_WORK/pool/prompts.json), IMG_EMBD (default $DM_WORK/img/embd.bin if it exists), RANK, LR
needs: torch, transformers 4.57 (Lfm2ForCausalLM), peft, safetensors, the gguf-py of llama.cpp."""
import json, math, os, random, sys, time
import numpy as np
import torch, torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, Lfm2Config, Lfm2ForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # also puts llama.cpp/gguf-py on the path
import gguf

LOWBIT, OUT = sys.argv[1], sys.argv[2]
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
HF = common.D1_HF
TOPK, BS = 32, 8
LR = float(os.environ.get("LR", 2e-4))
RANK = int(os.environ.get("RANK", 16))
dev = "cuda"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); random.seed(0)

# ---------- model
cfg_all = json.load(open(f"{HF}/config.json"))
tc = dict(cfg_all["text_config"]); tc.pop("architectures", None); tc.pop("dtype", None)
cfg = Lfm2Config(**tc)
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file=f"{HF}/tokenizer.json")  # tokenizer_config names a transformers-5 class
tok.pad_token_id = cfg_all.get("pad_token_id", 124893)


def load_hf_state():
    sd = {}
    with safe_open(f"{HF}/model.safetensors", "pt") as f:
        for k in f.keys():
            if k.startswith("model.language_model."):
                sd["model." + k[len("model.language_model."):]] = f.get_tensor(k)
            elif k.startswith("lm_head."):
                sd[k] = f.get_tensor(k)
    return sd


def build(sd):
    torch.set_default_dtype(torch.bfloat16)
    m = Lfm2ForCausalLM(cfg)  # on cpu, real buffers (rotary inv_freq)
    torch.set_default_dtype(torch.float32)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    m.tie_weights()
    return m.to(dev, torch.bfloat16).eval()


# ---------- data: rendered decision prompts (heal/build_prompts.py) and optional image embeddings (heal/build_images.py)
prompts = json.load(open(os.environ.get("PROMPTS", f"{common.WORK}/pool/prompts.json")))
random.shuffle(prompts)
enc = [tok(p, add_special_tokens=False)["input_ids"][-512:] for p in prompts]
print(len(enc), "text prompts, mean tokens", int(np.mean([len(e) for e in enc])), flush=True)
# image prompts: exact llama.cpp input embeddings dumped by d1x (embd mode): int32 n, then n x 2048 fp16
EMB = None
IMG = os.environ.get("IMG_EMBD", f"{common.WORK}/img/embd.bin")
if os.path.exists(IMG):
    raw = np.memmap(IMG, np.uint8, "r")
    off = 0
    while off < len(raw):
        n = int(np.frombuffer(raw[off:off + 4], np.int32)[0])
        enc.append(("e", off + 4, n)); off += 4 + n * 2048 * 2
    EMB = raw
    print(len(enc), "prompts with images", flush=True)
is_img = lambda e: isinstance(e, tuple)


def make(idx):
    items = [enc[j] for j in idx]
    if is_img(items[0]):  # (x is an embedding tensor)
        n = max(e[2] for e in items)
        x = torch.zeros((len(items), n, 2048), dtype=torch.bfloat16); att = torch.zeros((len(items), n), dtype=torch.long)
        for b, (_, o, k) in enumerate(items):
            x[b, :k] = torch.from_numpy(np.frombuffer(EMB[o:o + k * 4096], np.float16).reshape(k, 2048).astype(np.float32)).to(torch.bfloat16)
            att[b, :k] = 1
        last = [e[2] - 1 for e in items]
    else:
        n = max(map(len, items))
        x = torch.full((len(items), n), tok.pad_token_id or 0); att = torch.zeros((len(items), n), dtype=torch.long)
        for b, e in enumerate(items):
            x[b, :len(e)] = torch.tensor(e); att[b, :len(e)] = 1
        last = [len(e) - 1 for e in items]
    return idx, x.to(dev), att.to(dev), torch.tensor(last, device=dev)


def batches(order, shuffle_batches=False):
    groups = [[j for j in order if not is_img(enc[j])], [j for j in order if is_img(enc[j])]]  # one kind per batch
    chunks = [g[i:i + BS] for g in groups for i in range(0, len(g) - BS + 1, BS)]
    if shuffle_batches: random.shuffle(chunks)
    for c in chunks:
        yield make(c)


def last_logits(m, x, att, last):
    inner = m.get_base_model() if hasattr(m, "get_base_model") else m  # peft wraps the causal lm
    if x.dtype == torch.bfloat16:
        h = inner.model(inputs_embeds=x, attention_mask=att).last_hidden_state
    else:
        h = inner.model(input_ids=x, attention_mask=att).last_hidden_state
    h = h[torch.arange(len(last), device=dev), last]
    return inner.lm_head(h).float()


# ---------- 1. teacher top-k at the answer position
tpath = f"{OUT}/teacher.pt"
if not os.path.exists(tpath):
    teacher = build(load_hf_state())
    T_ids = torch.zeros((len(enc), TOPK), dtype=torch.long); T_lp = torch.zeros((len(enc), TOPK))
    with torch.no_grad():
        groups = [[j for j in range(len(enc)) if not is_img(enc[j])], [j for j in range(len(enc)) if is_img(enc[j])]]
        for g in groups:  # cover every sample: pad the last chunk with repeats
          if not g: continue
          g = g + g[:(-len(g)) % BS]
          for i0 in range(0, len(g), BS):
            idx, x, att, last = make(g[i0:i0 + BS])
            lp = F.log_softmax(last_logits(teacher, x, att, last), -1)
            v, i = lp.topk(TOPK, -1)
            for b, j in enumerate(idx):
                T_ids[j], T_lp[j] = i[b].cpu(), v[b].cpu()
    torch.save((T_ids, T_lp), tpath)
    del teacher; torch.cuda.empty_cache()
T_ids, T_lp = torch.load(tpath)
print("teacher done", flush=True)

# ---------- 2. student: dequantized low-bit weights
sd = load_hf_state()
r = gguf.GGUFReader(LOWBIT)
tmap = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LFM2, cfg.num_hidden_layers)
gg = {t.name: t for t in r.tensors}
replaced = 0
for k in list(sd):
    if not k.endswith(".weight") or sd[k].dim() != 2 or "embed" in k or k.startswith("lm_head"):
        continue
    name = tmap.get_name(k[:-len(".weight")], try_suffixes=())
    if name is None or name + ".weight" not in gg:
        print("no gguf tensor for", k); continue
    t = gg[name + ".weight"]
    w = gguf.quants.dequantize(t.data, t.tensor_type)
    assert tuple(w.shape) == tuple(sd[k].shape), (k, w.shape, sd[k].shape)
    sd[k] = torch.from_numpy(np.ascontiguousarray(w)).to(torch.bfloat16)
    replaced += 1
print("replaced", replaced, "linear weights with", os.path.basename(LOWBIT), flush=True)
student = build(sd)
del sd

from peft import LoraConfig, get_peft_model
targets = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
student = get_peft_model(student, LoraConfig(r=RANK, lora_alpha=RANK, lora_dropout=0.0, target_modules=targets))
student.print_trainable_parameters()
student.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
student.get_base_model().config.use_cache = False
opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)


def kl(logits, idx):
    lp = F.log_softmax(logits, -1)
    t_ids, t_lp = T_ids[idx].to(dev), T_lp[idx].to(dev)
    tp = torch.softmax(t_lp, -1)  # renormalized teacher top-k
    s = lp.gather(1, t_ids)
    s = s - torch.logsumexp(s, -1, keepdim=True)
    return (tp * (tp.clamp_min(1e-9).log() - s)).sum(-1).mean()


# baseline: agreement of the top-1 answer with the teacher on a fixed slice
EVAL_IDX = random.Random(5).sample(range(len(enc)), 480)


def agree():
    student.eval(); hit = tot = 0
    with torch.no_grad():
        for idx, x, att, last in batches(EVAL_IDX):
            hit += (last_logits(student, x, att, last).argmax(-1).cpu() == T_ids[idx, 0]).sum().item(); tot += len(idx)
    student.train(); return hit / tot


print(f"before: top-1 agreement on 400 training prompts {agree():.3f}", flush=True)
student.train()
step, t0 = 0, time.time()
sched = lambda s: LR * min(1, s / 50) * 0.5 * (1 + math.cos(math.pi * min(s, STEPS) / STEPS))
while step < STEPS:
    order = list(range(len(enc))); random.shuffle(order)
    for idx, x, att, last in batches(order, shuffle_batches=True):
        for g in opt.param_groups: g["lr"] = sched(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = kl(last_logits(student, x, att, last), idx)
        loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step(); opt.zero_grad()
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step >= STEPS: break
print(f"after: top-1 agreement on 400 training prompts {agree():.3f}", flush=True)
student.save_pretrained(f"{OUT}/lora")
print("saved", f"{OUT}/lora", flush=True)
