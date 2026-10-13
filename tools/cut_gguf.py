"""cut d1-3B after layer L and bake an early-exit probe into a new output head. the result runs on stock llama.cpp.

the probe reads rms(h_L) and says p(yes) = sigmoid(w . rms(h_L) + b). llama.cpp computes
logit_t = sum_i g_i * rms(h)_i * o_t,i  (g = output norm weight, o_t = output row of token t).
so set o_yes = +w_eff / (2g) and o_no = -w_eff / (2g): logit_yes - logit_no = w_eff . rms(h).
the bias b is folded in with u, a vector with u . rms(h) ~= 1 on training states (w_eff = w + b u).
only yes/no answers are valid in the cut model: every other row is still the input embedding.

usage: cut_gguf.py <src.gguf> <L> <out.gguf> <train tasks, comma list>"""
import json, os, sys
import numpy as np
import gguf

src, L, out, train_tasks = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4].split(",")
N_LAYER, N_EMBD = 30, 2048
d = os.environ.get("DUMP_DIR", "work/dumps/noul8")  # items.jsonl, full.jsonl, h.f32 from tools/d1x dump mode

# 1. probe on the chosen tasks only
items = [json.loads(l) for l in open(f"{d}/items.jsonl")]
full = [json.loads(l) for l in open(f"{d}/full.jsonl")]
H = np.memmap(f"{d}/h.f32", np.float32, "r", shape=(len(items), N_LAYER + 1, N_EMBD))
task = np.array([it["task"] for it in items]); pd1 = np.array([f["p_yes"] for f in full])
tr = np.where(np.isin(task, train_tasks))[0]
rms = lambda h: h / np.sqrt((h * h).mean(-1, keepdims=True) + 1e-5)
X = rms(np.asarray(H[tr, L - 1]))


def fit(X, t, lam=1e-3, steps=800, lr=0.5, k=128):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
    _, _, Vt = np.linalg.svd(Z, full_matrices=False); P = Vt[:k].T
    A = Z @ P; s = A.std(0) + 1e-6; A = A / s; w = np.zeros(k); b = 0.0
    for _ in range(steps):
        q = 1 / (1 + np.exp(-(A @ w + b))); g = q - t
        w -= lr * (A.T @ g / len(A) + lam * w); b -= lr * g.mean()
    W = P @ (w / s) / sd; return W, b - mu @ W


W, b = fit(X, pd1[tr])
u = np.linalg.solve(X.T @ X + 1e-2 * np.eye(N_EMBD), X.T @ np.ones(len(X)))
w_eff = (W + b * u).astype(np.float64)
print(f"probe on {len(tr)} states from {train_tasks}, bias {b:.3f}, |u.x - 1| mean {np.abs(X @ u - 1).mean():.4f}")

# 2. new output head from the tied embedding, yes/no rows replaced
r = gguf.GGUFReader(src)
T = {t.name: t for t in r.tensors}
g = np.asarray(T["token_embd_norm.weight"].data, np.float64)
print(f"output norm weight: min |g| {np.abs(g).min():.4f}")
tok = r.fields["tokenizer.ggml.tokens"]
V = {bytes(tok.parts[i]).decode("utf8", "replace"): k for k, i in enumerate(tok.data)}
yes = [V[f] for f in ("yes", "Yes", "YES") if f in V]; no = [V[f] for f in ("no", "No", "NO") if f in V]
E = gguf.quants.dequantize(np.asarray(T["token_embd.weight"].data), T["token_embd.weight"].tensor_type).astype(np.float32)
row = (w_eff / (2 * g)).astype(np.float32)
for t in yes: E[t] = row
for t in no: E[t] = -row
outq = gguf.quants.quantize(E, gguf.GGMLQuantizationType.Q8_0)
del E
print(f"yes rows {yes}, no rows {no}")

# 3. write: first L blocks, per-layer metadata cut to L
w = gguf.GGUFWriter(out, arch=r.fields["general.architecture"].contents())
for field in r.fields.values():
    if field.name == "general.architecture" or field.name.startswith("GGUF."):
        continue
    vt = field.types[0]; st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == "lfm2.block_count":
        val = L
    elif field.name == "lfm2.attention.head_count_kv":
        val = val[:L]
    elif field.name == "general.name":
        val = f"{val}-cut{L}-exit"
    w.add_key_value(field.name, val, vt, sub_type=st)
keep = [t for t in r.tensors if not t.name.startswith("blk.") or int(t.name.split(".")[1]) < L]
for t in keep:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.add_tensor_info("output.weight", outq.shape, outq.dtype, outq.nbytes, gguf.GGMLQuantizationType.Q8_0)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in keep:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.write_tensor_data(outq)
w.close()
print("wrote", out)
