"""drop whole layers from an lfm2 gguf (d1-3B), renumber the rest. runs on stock llama.cpp.
usage: prune_gguf.py <src.gguf> <out.gguf> <layers to drop, comma list, 0-based>"""
import sys
import gguf

src, out = sys.argv[1], sys.argv[2]
drop = {int(x) for x in sys.argv[3].split(",") if x != ""}
r = gguf.GGUFReader(src)
n = int(r.fields["lfm2.block_count"].contents())
keep = [i for i in range(n) if i not in drop]
new = {old: k for k, old in enumerate(keep)}
w = gguf.GGUFWriter(out, arch=r.fields["general.architecture"].contents())
for f in r.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    v = f.contents()
    if f.name == "lfm2.block_count":
        v = len(keep)
    elif f.name == "lfm2.attention.head_count_kv":
        v = [v[i] for i in keep]
    elif f.name == "general.name":
        v = f"{v}-drop{'-'.join(map(str, sorted(drop)))}" if drop else v
    w.add_key_value(f.name, v, vt, sub_type=st)
tensors = []
for t in r.tensors:
    name = t.name
    if name.startswith("blk."):
        i = int(name.split(".")[1])
        if i in drop:
            continue
        name = f"blk.{new[i]}." + name.split(".", 2)[2]
    tensors.append((name, t))
for name, t in tensors:
    w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for _, t in tensors:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
print(f"wrote {out}: {len(keep)} layers, dropped {sorted(drop)}")
