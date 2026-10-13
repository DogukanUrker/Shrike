"""the text prompts used for healing -> $DM_WORK/pool/prompts.json (rendered systemone prompts, teacher answers come later).
sources: train splits / other datasets (data/fetch_data.py) + items >= 100 of the english eval tasks.
held out completely: qqp, emotion (english), MASSIVE (unseen task), and every multilingual test item."""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import DATA, WORK, rows
import tasks

random.seed(1)
P = []


def add(state, q, kind="noul", opts=None):
    it = {"kind": kind, "state": state, "question": q}
    if opts: it["options"] = opts
    P.append(tasks.render(it))


# english, other datasets
REV_Q = ["Is this review positive?", "Is the customer happy?", "Would the writer recommend this?", "Is the writer complaining?"]
STARS = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
for name, key in [("yelp", "text"), ("amazon", "content"), ("rt", "text")]:
    for r in rows("pool", name):
        s, k = r[key][:1500], random.random()
        if k < .6: add(s, random.choice(REV_Q))
        elif k < .8: add(s, "What is the sentiment of this review?", "choice", random.choice([["Negative", "Positive"], ["Negative", "Mixed", "Positive"]]))
        else: add(s, "How positive is this review?", "score", STARS)
for name in ["mnli", "snli"]:
    for r in rows("pool", name):
        s = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
        if random.random() < .5: add(s, "Does the premise imply the hypothesis?")
        else: add(s, "How is the hypothesis related to the premise?", "choice", ["It follows from the premise", "It may or may not be true", "It contradicts the premise"])
for r in rows("pool", "offensive"):
    add(r["text"], random.choice(["Is this tweet offensive?", "Is this tweet polite?", "Does this tweet insult someone?"]))
for r in rows("pool", "boolq"):
    add(r["passage"][:900], r["question"].capitalize() + "?")
AG = tasks.CHOICE["ag"][2]
for r in rows("pool", "ag"):
    if random.random() < .6: add(r["text"], "What is the topic of this news article?", "choice", AG)
    else: add(r["text"], f"Is this article about {random.choice(['sports', 'business', 'politics', 'technology', 'science'])}?")
DB = tasks.CHOICE["dbpedia"][2]
for r in rows("pool", "dbpedia"):
    s = f"{r['title']}: {r['content']}"
    if random.random() < .6: add(s, "What is this Wikipedia article about?", "choice", DB)
    else: add(s, f"Is this article about {random.choice(DB).lower()}?")
_, TQ, TO, _ = tasks.CHOICE["trec"]
for r in rows("pool", "trec"):
    add(r["text"], TQ, "choice", TO)
# english eval tasks, items >= 100 (the eval uses the first 100), held-out tasks skipped
for it in tasks.english(100, 400, skip_held_out=True):
    P.append(tasks.render(it))
# multilingual, train splits
names = sorted({f.rsplit("_", 1)[0] for f in os.listdir(f"{DATA}/mlpool")})
for n in names:
    for r in rows("mlpool", n):
        if n.startswith("xnli"):
            s = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
            if random.random() < .5: add(s, "How is the hypothesis related to the premise?", "choice", tasks.XNLI)
            else: add(s, "Does the premise imply the hypothesis?")
        elif n.startswith("pawsx"):
            add(f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}", "Do the two sentences mean the same thing?")
        else:
            if random.random() < .6: add(r["text"], "What is the sentiment of this text?", "choice", ["Positive", "Neutral", "Negative"])
            else: add(r["text"], random.choice(["Is this text positive?", "Is the writer unhappy?"]))
# pool v3 extras
clinc = rows("pool3", "clinc")
intents = sorted({str(r["intent"]) for r in clinc})
for r in clinc:
    k = str(r["intent"])
    if random.random() < .6:
        o = [k] + random.sample([i for i in intents if i != k], 5); random.shuffle(o)
        add(r["text"], "What does the user want?", "choice", [f"Intent {x}" for x in o])
    else:
        add(r["text"], random.choice(["Is the user asking a question?", "Is this a request to do something?", "Is the user talking about money?"]))
for r in rows("pool3", "spam"):
    k = random.random()
    if k < .5: add(r["sms"], "Is this message spam?")
    elif k < .8: add(r["sms"], "What kind of message is this?", "choice", ["Spam or advertising", "A normal personal message"])
    else: add(r["sms"], "How likely is this message to be spam?", "score", ["Not spam", "Unlikely", "Unsure", "Likely", "Certainly spam"])
for r in rows("pool3", "hate"): add(r["text"], random.choice(["Is this tweet hateful?", "Does this tweet attack a group of people?"]))
for r in rows("pool3", "irony"): add(r["text"], "Is this tweet ironic?")
for r in rows("pool3", "scitail"): add(f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}", "Does the premise support the hypothesis?")
for l in ["arabic", "italian", "portuguese", "hindi"]:
    for r in rows("pool3", f"tsm_{l}"): add(r["text"], "What is the sentiment of this tweet?", "choice", ["Negative", "Neutral", "Positive"])
YT = ["Society and culture", "Science and mathematics", "Health", "Education and reference", "Computers and internet", "Sports",
      "Business and finance", "Entertainment and music", "Family and relationships", "Politics and government"]
for r in rows("pool3", "yahoo"):
    s = f"{r['question_title']}\n{r['question_content']}"[:1200]
    if random.random() < .7: add(s, "What is this question about?", "choice", YT)
    else: add(s, "How clear is this question?", "score", ["Very unclear", "Unclear", "Okay", "Clear", "Very clear"])

random.shuffle(P)
os.makedirs(f"{WORK}/pool", exist_ok=True)
json.dump(P, open(f"{WORK}/pool/prompts.json", "w"))
print(len(P), "text prompts ->", f"{WORK}/pool/prompts.json")
