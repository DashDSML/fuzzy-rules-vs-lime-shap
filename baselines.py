"""LIME and SHAP explanations of DistilBERT on the ANFIS test sentences.

Uses the first N test indices from results/split.json (sentences from results/features.csv).
Both explainers explain P(positive), so signed scores share a direction:
  score > 0 pushes toward positive, score < 0 toward negative.
LIME also stores its local surrogate's prediction at the instance (local_pred = surrogate
P(positive)); surrogate_label = local_pred > 0.5, for surrogate-vs-DistilBERT agreement.

Saves results/lime.json and results/shap.json (list of records, top-5 words each).
Usage: python baselines.py [--n 50] [--shap-n 50] [--lime-samples 500] [--shap-evals 300]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import shap
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
TOP_K = 5
BATCH = 64
SEED = 0

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()


@torch.no_grad()
def predict_proba(texts):
    """List of strings -> (n, 2) softmax probabilities [neg, pos]."""
    texts = [str(t) for t in texts]
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    return np.concatenate(out)


def p_pos(texts):
    return predict_proba(texts)[:, 1]


def top_k(pairs, k=TOP_K):
    pairs = sorted(pairs, key=lambda p: abs(p[1]), reverse=True)[:k]
    return [{"word": w, "score": float(s)} for w, s in pairs]


def shap_words(tokens, values):
    """Merge SHAP subword tokens into whole words (sum of values); drop special/empty tokens."""
    words, prev_space = [], True
    for t, v in zip(tokens, values):
        if t.strip() == "":
            prev_space = True
            continue
        if words and not prev_space:
            words[-1][0] += t.strip()
            words[-1][1] += v
        else:
            words.append([t.strip(), v])
        prev_space = t.endswith(" ")
    return [(w, v) for w, v in words]


def run_lime(rows, n_samples):
    explainer = LimeTextExplainer(class_names=["negative", "positive"], random_state=SEED)
    res, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        exp = explainer.explain_instance(r.sentence, predict_proba, labels=(1,),
                                         num_features=TOP_K, num_samples=n_samples)
        local_pred = float(np.ravel(exp.local_pred)[0])
        res.append({
            "idx": int(r.idx), "sentence": r.sentence, "model_label": int(r.model_label),
            "model_p_pos": float(r.p_pos),
            "local_pred": local_pred, "surrogate_label": int(local_pred > 0.5),
            "intercept": float(exp.intercept[1]), "r2": float(exp.score),
            "top_words": top_k(exp.as_list(label=1)),
        })
        if i % 10 == 0 or i == len(rows):
            print(f"  LIME {i}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    return res


def run_shap(rows, max_evals):
    explainer = shap.Explainer(p_pos, shap.maskers.Text(tok))
    res, t0 = [], time.time()
    for i, r in enumerate(rows, 1):
        sv = explainer([r.sentence], max_evals=max_evals, batch_size=BATCH, silent=True)
        words = shap_words(list(sv.data[0]), np.asarray(sv.values[0]).ravel())
        res.append({
            "idx": int(r.idx), "sentence": r.sentence, "model_label": int(r.model_label),
            "model_p_pos": float(r.p_pos),
            "base_value": float(np.ravel(sv.base_values[0])[0]),
            "top_words": top_k(words),
        })
        if i % 10 == 0 or i == len(rows):
            print(f"  SHAP {i}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="test sentences for LIME")
    ap.add_argument("--shap-n", type=int, default=None, help="test sentences for SHAP (default = --n)")
    ap.add_argument("--lime-samples", type=int, default=500)
    ap.add_argument("--shap-evals", type=int, default=300)
    a = ap.parse_args()
    shap_n = a.shap_n or a.n
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_csv(os.path.join("results", "features.csv")).set_index("idx", drop=False)
    with open(os.path.join("results", "split.json")) as f:
        test_idx = json.load(f)["test_idx"]
    sel = df.loc[test_idx[:max(a.n, shap_n)]].copy()
    sel["p_pos"] = p_pos(sel.sentence.tolist())
    mismatch = int(((sel.p_pos > 0.5).astype(int) != sel.model_label).sum())
    print(f"Sentences: LIME {a.n}, SHAP {shap_n} | label mismatch vs features.csv: {mismatch}")

    t = time.time()
    lime_res = run_lime(list(sel.head(a.n).itertuples()), a.lime_samples)
    with open(os.path.join("results", "lime.json"), "w") as f:
        json.dump(lime_res, f, indent=1)
    agree = np.mean([r["surrogate_label"] == r["model_label"] for r in lime_res])
    print(f"LIME done {time.time() - t:.0f}s | surrogate agreement {agree:.3f} | "
          f"mean R2 {np.mean([r['r2'] for r in lime_res]):.3f}")

    t = time.time()
    shap_res = run_shap(list(sel.head(shap_n).itertuples()), a.shap_evals)
    with open(os.path.join("results", "shap.json"), "w") as f:
        json.dump(shap_res, f, indent=1)
    print(f"SHAP done {time.time() - t:.0f}s")

    r0, s0 = lime_res[0], shap_res[0]
    print(f"\nExample [{r0['idx']}] {r0['sentence']}")
    print("  LIME:", ", ".join(f"{w['word']} {w['score']:+.3f}" for w in r0["top_words"]))
    print("  SHAP:", ", ".join(f"{w['word']} {w['score']:+.3f}" for w in s0["top_words"]))
    print("Saved results/lime.json, results/shap.json")


if __name__ == "__main__":
    main()
