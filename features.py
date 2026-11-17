"""Run DistilBERT (SST-2) on 400 validation sentences and extract attention-based features.

Output: results/features.csv with columns
  idx, sentence, true_label, model_label, confidence, pos_score, neg_score, negation

ANFIS inputs are ANFIS_INPUTS only. `confidence` = max(softmax) is ANALYSIS-ONLY: it is
derived from the model's output, so feeding it to ANFIS would leak the target.
"""
import os
import re

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
N = 400
BATCH = 16
OUT = os.path.join("results", "features.csv")
ANFIS_INPUTS = ["pos_score", "neg_score", "negation"]  # never include confidence (leakage)
ANALYSIS_ONLY = ["confidence"]

POS_WORDS = set("""
good great excellent amazing wonderful fantastic brilliant superb best better love loved lovely
enjoy enjoyable enjoyed fun funny charming delightful beautiful moving touching powerful
perfect fresh clever smart witty engaging entertaining compelling impressive remarkable
memorable masterpiece gem rich warm sweet pleasant pleasure solid strong success successful
hilarious inventive original intelligent thoughtful riveting gripping stunning terrific
admirable affecting appealing likable satisfying refreshing heartfelt worth wins win
well nice fine beauty funniest entertain entertainment charm charismatic winning joy joyous
exciting excitement thrilling suspenseful tender poignant elegant graceful lively vivid
sharp smartly intriguing fascinating absorbing captivating mesmerizing haunting magical
inspired inspiring uplifting triumph triumphant wonderfully beautifully honest sincere
genuine authentic insightful wise rewarding treat delight enjoyably gorgeous lovingly
playful sweetly hopeful humor humour laughs laugh classic dazzling spectacular superior
""".split())

NEG_WORDS = set("""
bad worse worst awful terrible horrible poor boring bored dull tedious mess messy stupid
dumb silly lame weak flat waste wasted annoying ugly painful pointless predictable bland
mediocre disappointing disappointment fails failed failure fail hate hated unfunny tired
tiresome clumsy confused confusing forgettable shallow empty lifeless lacks lacking
pretentious ridiculous unpleasant sloppy incoherent overlong repetitive cliche cliched
trite inept problem problems junk garbage nonsense sorry miserable
worthless unwatchable insufferable obnoxious irritating grating sluggish slow plodding
meandering aimless muddled murky labored contrived formulaic derivative stale generic
hollow thin uninspired uninteresting unconvincing implausible absurd preposterous
embarrassing cheap amateurish crude vulgar mindless numbing exhausting depressing
grim unbearable lousy dreadful pathetic hackneyed bore drags drag dreary ponderous
mean cold sour cynical misguided pity fault flaws flawed loses lost suffers
""".split())

def _load_opinion_lexicon():
    """Merge Hu & Liu (2004) opinion lexicon via NLTK; fall back to built-in lists on failure."""
    global POS_WORDS, NEG_WORDS
    try:
        import nltk
        nltk.download("opinion_lexicon", quiet=True)
        from nltk.corpus import opinion_lexicon
        POS_WORDS = POS_WORDS | set(opinion_lexicon.positive())
        NEG_WORDS = NEG_WORDS | set(opinion_lexicon.negative())
    except Exception as e:  # offline etc.
        print(f"[warn] opinion_lexicon unavailable ({e}); using built-in lexicon only")
    overlap = POS_WORDS & NEG_WORDS  # ambiguous words: drop from both
    POS_WORDS, NEG_WORDS = POS_WORDS - overlap, NEG_WORDS - overlap


_load_opinion_lexicon()

NEGATION_RE = re.compile(
    r"\b(not|no|never|none|nobody|nothing|neither|nor|nowhere|cannot|without|hardly|barely)\b|n't",
    re.IGNORECASE,
)


def word_attention(enc, attn_last, b):
    """Return {word: summed CLS attention} for batch item b.

    attn_last: (batch, heads, seq, seq) from the final layer. We average heads, take the
    [CLS] row, drop special/pad tokens, renormalise to sum 1, then merge word pieces.
    """
    cls_row = attn_last[b].mean(dim=0)[0]  # (seq,)
    word_ids = enc.word_ids(b)
    tokens = enc.tokens(b)
    words, weights = {}, {}
    for pos, wid in enumerate(word_ids):
        if wid is None:
            continue
        piece = tokens[pos][2:] if tokens[pos].startswith("##") else tokens[pos]
        words[wid] = words.get(wid, "") + piece
        weights[wid] = weights.get(wid, 0.0) + float(cls_row[pos])
    total = sum(weights.values()) or 1.0
    return [(words[w], weights[w] / total) for w in words]


def main():
    os.makedirs("results", exist_ok=True)
    torch.set_grad_enabled(False)

    ds = load_dataset("nyu-mll/glue", "sst2", split="validation").select(range(N))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, output_attentions=True)
    model.eval()

    rows = []
    for start in range(0, N, BATCH):
        batch = ds[start:start + BATCH]
        sents = batch["sentence"]
        enc = tok(sents, padding=True, truncation=True, max_length=128, return_tensors="pt")
        out = model(**enc)
        probs = out.logits.softmax(-1)
        attn_last = out.attentions[-1]

        for b, sent in enumerate(sents):
            pos_score = neg_score = 0.0
            for word, w in word_attention(enc, attn_last, b):
                if word in POS_WORDS:
                    pos_score += w
                elif word in NEG_WORDS:
                    neg_score += w
            rows.append({
                "idx": start + b,
                "sentence": sent.strip(),
                "true_label": int(batch["label"][b]),
                "model_label": int(probs[b].argmax()),
                "confidence": float(probs[b].max()),  # max(softmax), analysis-only
                "pos_score": pos_score,
                "neg_score": neg_score,
                "negation": int(bool(NEGATION_RE.search(sent))),
            })
        print(f"\r{start + len(sents)}/{N}", end="", flush=True)
    print()

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print(f"Saved {OUT}  shape={df.shape}")
    print(f"Model accuracy vs gold: {(df.model_label == df.true_label).mean():.3f}")
    print(f"Label balance (model): {df.model_label.value_counts().to_dict()}")
    print(f"Negation present: {df.negation.mean():.2%}  | "
          f"pos_score>0: {(df.pos_score > 0).mean():.2%}  | neg_score>0: {(df.neg_score > 0).mean():.2%}")
    zero_both = ((df.pos_score == 0) & (df.neg_score == 0)).mean()
    print(f"pos_score=0 AND neg_score=0: {zero_both:.2%}"
          + ("  <-- above 30%, lexicon coverage too low" if zero_both > 0.30 else ""))
    print(f"ANFIS inputs: {ANFIS_INPUTS} | analysis-only: {ANALYSIS_ONLY}")
    print(df[["confidence", "pos_score", "neg_score"]].describe().loc[["mean", "std", "min", "max"]].round(3))
    print(df.drop(columns="sentence").head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
