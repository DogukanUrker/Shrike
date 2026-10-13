"""compare runs against a reference run (default F16): agreement of the top answer and accuracy, per eval.
doom (yes/no on images) also reports accuracy at the best single threshold, like an agent would tune it.
usage: score.py <run name>... [--ref F16]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import WORK

args = sys.argv[1:]
ref = "F16"
if "--ref" in args:
    i = args.index("--ref"); ref = args[i + 1]; del args[i:i + 2]
ld = lambda n, e: [json.loads(l) for l in open(f"{WORK}/results/{n}/{e}.jsonl")]
EVALS = ["english", "doom", "multilingual", "massive"]
print(f"| run | " + " | ".join(f"{e} agree / acc" for e in EVALS) + " | mean agree | ms |")
print("|---|" + "---|" * (len(EVALS) + 2))
for n in [ref] + [a for a in args if a != ref]:
    cells, ag, ms = [], [], []
    for e in EVALS:
        try:
            R, M = ld(ref, e), ld(n, e)
        except FileNotFoundError:
            cells.append("-"); continue
        a = np.mean([np.argmax(x["probs"]) == np.argmax(y["probs"]) for x, y in zip(R, M)])
        if e == "doom":
            acc = max(np.mean([(y["probs"][0] >= t) == (y["y"] == 0) for y in M]) for t in np.arange(0.05, 0.96, 0.05))
        else:
            acc = np.mean([np.argmax(y["probs"]) == y["y"] for y in M])
        cells.append(f"{a:.3f} / {acc:.3f}"); ag.append(a); ms += [y["ms"] for y in M[10:]]
    print(f"| {n} | " + " | ".join(cells) + f" | {np.mean(ag):.3f} | {np.median(ms):.1f} |")
