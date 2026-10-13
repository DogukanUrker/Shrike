"""every task used for evaluation and healing. a task item is a dict:
  {"task", "kind": noul|choice|score, "state", "question", "options" (choice/score), "y" (label index, noul: 0 = yes)}
render(item) gives the exact prompt llama-server builds for /v1/systemone, request(item) the http body."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import rows, systemone_template

# ---------------- english eval: the first 100 items of each are the eval, items >= 100 can feed healing
YESNO = {
    "boolq": ("eval", "boolq", lambda r: (r["passage"][:900], r["question"].capitalize() + "?", r["answer"])),
    "sst2": ("eval", "sst2", lambda r: (r["sentence"], "Is this movie review positive?", r["label"] == 1)),
    "rte": ("eval", "rte", lambda r: (f"Text: {r['sentence1']}\nHypothesis: {r['sentence2']}", "Does the text imply that the hypothesis is true?", r["label"] == 0)),
    "mrpc": ("eval", "mrpc", lambda r: (f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}", "Do the two sentences mean the same thing?", r["label"] == 1)),
    "qnli": ("eval", "qnli", lambda r: (f"Question: {r['question']}\nSentence: {r['sentence']}", "Does the sentence contain the answer to the question?", r["label"] == 0)),
    "cola": ("eval", "cola", lambda r: (r["sentence"], "Is this sentence grammatically correct English?", r["label"] == 1)),
    "qqp": ("eval", "qqp", lambda r: (f"Question 1: {r['question1']}\nQuestion 2: {r['question2']}", "Are the two questions asking the same thing?", r["label"] == 1)),
    "imdb": ("eval", "imdb_a", lambda r: (r["text"][:1500], "Is this movie review positive?", r["label"] == 1)),
}
CHOICE = {
    "ag": ("c_ag", "What is the topic of this news article?", ["World news, politics, international affairs", "Sports", "Business, economy, finance", "Science and technology"],
           lambda r: (r["text"], r["label"])),
    "trec": ("c_trec", "What kind of answer does this question ask for?", ["A description or an abstract concept", "An entity, like an object, an animal or an event",
             "An abbreviation or what it stands for", "A person or a group of people", "A number, a date, an amount", "A location"], lambda r: (r["text"], r["label_coarse"])),
    "emotion": ("c_emotion", "Which emotion does the writer express?", ["Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"], lambda r: (r["text"], r["label"])),
    "dbpedia": ("c_dbpedia", "What is this Wikipedia article about?", ["A company", "A school or university", "An artist", "An athlete", "A politician or office holder",
                "A vehicle, ship or aircraft", "A building", "A natural place like a mountain or river", "A village", "An animal", "A plant", "A music album", "A film",
                "A book or written work"], lambda r: (f"{r['title']}: {r['content']}", r["label"])),
    "tweet": ("c_tweet", "What is the sentiment of this tweet?", ["Negative", "Neutral", "Positive"], lambda r: (r["text"], r["label"])),
}
HELD_OUT = {"qqp", "emotion"}  # never used for healing


def english(start=0, stop=100, skip_held_out=False):
    for task, (sub, name, f) in YESNO.items():
        if skip_held_out and task in HELD_OUT: continue
        rs = rows(sub, name)
        if task == "imdb":  # the source is sorted by label: interleave
            neg, pos = [r for r in rs if r["label"] == 0], [r for r in rs if r["label"] == 1]
            rs = [x for p in zip(neg, pos) for x in p]
        for r in rs[:400][start:stop]:
            s, q, yes = f(r)
            yield {"task": task, "kind": "noul", "state": s, "question": q, "y": 0 if yes else 1}
    for task, (name, q, opts, f) in CHOICE.items():
        if skip_held_out and task in HELD_OUT: continue
        for r in rows("eval", name)[:400][start:stop]:
            s, y = f(r)
            yield {"task": task, "kind": "choice", "state": s, "question": q, "options": opts, "y": int(y)}


# ---------------- multilingual eval (test splits) and the unseen task (MASSIVE)
XNLI = ["The hypothesis follows from the premise", "The hypothesis may or may not be true", "The hypothesis contradicts the premise"]
SENT = ["Positive", "Neutral", "Negative"]


def multilingual():
    for l in ["ar", "de", "es", "fr", "hi", "ru", "tr", "vi", "zh"]:
        for i, r in enumerate(rows("ml", f"xnli_{l}")):
            s = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
            if i % 2 == 0:
                yield {"task": "xnli", "lang": l, "kind": "choice", "state": s, "question": "How is the hypothesis related to the premise?", "options": XNLI, "y": r["label"]}
            else:
                yield {"task": "xnli", "lang": l, "kind": "noul", "state": s, "question": "Does the premise imply the hypothesis?", "y": 0 if r["label"] == 0 else 1}
    for l in ["de", "es", "fr", "ja", "ko", "zh"]:
        for r in rows("ml", f"pawsx_{l}"):
            yield {"task": "pawsx", "lang": l, "kind": "noul", "state": f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}",
                   "question": "Do the two sentences mean the same thing?", "y": 0 if r["label"] == 1 else 1}
    for l in ["arabic", "chinese", "french", "german", "indonesian", "japanese", "portuguese", "spanish"]:
        for r in rows("ml", f"sent_{l}"):
            yield {"task": "sentiment", "lang": l, "kind": "choice", "state": r["text"], "question": "What is the sentiment of this text?", "options": SENT, "y": r["label"]}


MASSIVE = {"alarm": "Setting or changing an alarm", "audio": "Volume or sound settings", "calendar": "Calendar events and reminders",
           "cooking": "Cooking and recipes", "datetime": "Date or time questions", "email": "Email", "general": "Small talk or a joke",
           "iot": "Smart home devices and lights", "lists": "To-do or shopping lists", "music": "Music preferences or info", "news": "News",
           "play": "Playing music, radio, podcasts or audiobooks", "qa": "A factual question or definition", "recommendation": "Recommendations for places or events",
           "social": "Social media", "takeaway": "Ordering food", "transport": "Transport, trains, taxis, traffic", "weather": "Weather"}


def massive():
    keys = list(MASSIVE)
    for l in ["de", "fr", "ja", "zh-CN", "ru", "ar", "hi", "tr", "ko"]:  # es: the datasets server fails on it
        for r in rows("ml", f"massive_{l}"):
            yield {"task": "massive", "lang": l, "kind": "choice", "state": r["text"], "question": "What kind of request is this, said to a voice assistant?",
                   "options": [MASSIVE[k] for k in keys], "y": keys.index(r["label"])}


# ---------------- rendering
def options(item):
    if item["kind"] == "noul":
        return [{"key": "true", "description": None}, {"key": "false", "description": None}]
    if item["kind"] == "score":
        return [{"key": str(i), "description": d} for i, d in enumerate(item["options"])]
    return [{"key": f"o{i}", "description": d, "label": chr(65 + i)} for i, d in enumerate(item["options"])]


def render(item, image=False):
    """the prompt llama-server renders. image=True puts the media marker where the server puts the image."""
    return systemone_template().render(images=["<__media__>"] if image else [], state=item.get("state"), instructions=item["question"],
                                       type=item["kind"], options=options(item))


def request(item, model="d1"):
    q = {"type": item["kind"], "instructions": item["question"]}
    if item["kind"] == "choice":
        q["criteria"] = {f"o{i}": d for i, d in enumerate(item["options"])}
    elif item["kind"] == "score":
        q["criteria"] = list(item["options"])
    body = {"model": model, "state": item.get("state"), "questions": {"q": q}}
    if item.get("image"):
        import base64
        body["images"] = ["data:image/png;base64," + base64.b64encode(open(item["image"], "rb").read()).decode()]
    return body


def probs(item, answer):
    """option probabilities in option order from a /v1/systemone answer."""
    if item["kind"] == "noul":
        return [answer["noul"], 1 - answer["noul"]]
    return [answer["probabilities"][k] for k in (f"o{i}" for i in range(len(item["options"])))] if item["kind"] == "choice" \
        else [answer["probabilities"][str(i)] for i in range(len(item["options"]))]
