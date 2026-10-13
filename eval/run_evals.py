"""run the evals against a running /v1/systemone server, write $DM_WORK/results/<name>/<eval>.jsonl
usage: run_evals.py <url> <name> [english multilingual massive doom]   (default: all four; doom needs the frames)"""
import json, os, sys, time
import requests
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import WORK, DATA
import tasks

url, name = sys.argv[1], sys.argv[2]
which = sys.argv[3:] or ["english", "multilingual", "massive", "doom"]
out = f"{WORK}/results/{name}"
os.makedirs(out, exist_ok=True)
s = requests.Session()


def doom():
    d = f"{DATA}/frames/doom"
    if not os.path.exists(f"{d}/labels.json"):
        print("no doom frames, skipping (frames/doom_collect.py)"); return
    for l in json.load(open(f"{d}/labels.json"))[-240:]:  # the last 240 frames, never used for healing
        yield {"task": "doom", "kind": "noul", "state": None, "image": f"{d}/{l['file']}",
               "question": "Is there a living (standing, not dead) monster or soldier in this picture?", "y": 0 if l["center"] else 1}


SETS = {"english": lambda: tasks.english(0, 100), "multilingual": tasks.multilingual, "massive": tasks.massive, "doom": doom}
for w in which:
    n = 0
    with open(f"{out}/{w}.jsonl", "w") as f:
        for it in SETS[w]():
            t = time.perf_counter()
            a = s.post(url, json=tasks.request(it)).json()["answers"]["q"]
            ms = (time.perf_counter() - t) * 1000
            f.write(json.dumps({"task": it["task"], "lang": it.get("lang"), "kind": it["kind"], "y": it["y"], "probs": tasks.probs(it, a), "ms": ms}) + "\n")
            n += 1
    print(f"{name} {w}: {n} decisions -> {out}/{w}.jsonl", flush=True)
