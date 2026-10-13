"""image prompts for healing -> $DM_WORK/img/req.txt: one d1x request line per prompt (embd mode).
coco photos (data/img), doom frames 0-559 (the eval uses the last 240) and mario crops (data/frames).
then: tools/d1x <F16 gguf> <mmproj> < req.txt  dumps the exact llama.cpp input embeddings to $DM_WORK/img/embd.bin"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import DATA, WORK
import tasks

random.seed(2)
os.makedirs(f"{WORK}/img", exist_ok=True)
OUTF = f"{WORK}/img/embd.bin"
esc = lambda p: p.replace("\\", "\\\\").replace("\n", "\\n")
noul = lambda q: tasks.render({"kind": "noul", "question": q, "state": None}, image=True)
choice = lambda q, o: tasks.render({"kind": "choice", "question": q, "options": o, "state": None}, image=True)
S = []
coco = json.load(open(f"{DATA}/img/coco_meta.json"))
names = sorted({n for m in coco for n in m["labels"]})
for m in coco:
    img = f"{DATA}/img/{m['file']}"
    if m["labels"]:
        S.append((img, noul(f"Is there a {random.choice(m['labels'])} in this picture?")))
    S.append((img, noul(f"Is there a {random.choice([n for n in names if n not in m['labels']])} in this picture?")))
    if m["labels"]:
        o = [random.choice(m["labels"])] + random.sample([n for n in names if n not in m["labels"]], 3); random.shuffle(o)
        S.append((img, choice("What can you see in this picture?", [x.capitalize() for x in o])))
dd = f"{DATA}/frames/doom"
if os.path.exists(f"{dd}/labels.json"):
    for l in json.load(open(f"{dd}/labels.json"))[:560]:
        img = f"{dd}/{l['file']}"
        S.append((img, noul("Is there a living (standing, not dead) monster or soldier in this picture?")))
        S.append((img, random.choice([noul("Is a monster right in the middle of this picture?"),
                                      choice("What is in this picture?", ["A living monster", "A dead body", "Only walls and floor"])])))
md = f"{DATA}/frames/mario"
if os.path.exists(f"{md}/rows.json"):
    EYES = ["Is there an enemy creature (like a brown mushroom goomba or a turtle) in this picture?",
            "Is there a green pipe (a vertical tube with a rim on top) or a stack of brown stone blocks standing on the ground in this picture? Hills and bushes in the background do not count.",
            "Is there a hole or gap in the brown ground at the bottom of this picture?"]
    rows = json.load(open(f"{md}/rows.json"))
    for r in random.sample(rows, min(1200, len(rows))):
        S.append((f"{md}/{r['file']}", noul(random.choice(EYES))))
random.shuffle(S)
with open(f"{WORK}/img/req.txt", "w") as f:
    for img, p in S:
        f.write(f"{img}\t{esc(p)}\tembd:{OUTF}\n")
print(len(S), "image prompts ->", f"{WORK}/img/req.txt")
