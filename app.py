"""Gradio demo: sentence -> DistilBERT prediction, ANFIS inputs + memberships, deciding fuzzy rule.

Deciding rule = largest (normalised firing x consequent) in the direction of the ANFIS output
(anfis.deciding_rule, also used by compare.py). Also lists the top 3 rules by firing strength.

Reuses the feature code from features.py and the trained model in results/anfis_model.pt.
No live LIME/SHAP (see results/lime.json, results/shap.json and compare.py).
Usage: python app.py
"""
import os

import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from anfis import ANFIS, INPUTS, NEG_LABELS, NICE, deciding_rule
from features import NEG_WORDS, NEGATION_RE, POS_WORDS, word_attention

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
MF_LABELS = ["Low", "Medium", "High"]
MAX_TOKENS = 128  # tokenizer truncation limit
LOW_COVERAGE = 0.05  # raw firing of the deciding rule below this -> warn

tok = AutoTokenizer.from_pretrained(MODEL)
bert = AutoModelForSequenceClassification.from_pretrained(MODEL, output_attentions=True).eval()

_ck = torch.load(os.path.join("results", "anfis_model.pt"), weights_only=False)
anfis = ANFIS(torch.zeros(2, 3), torch.ones(2, 3))
anfis.load_state_dict(_ck["state_dict"])
anfis.eval()
MF_NAMES = _ck["mf_names"]  # per input: {mf index -> Low/Medium/High}
KEPT = _ck["kept_rules"]


def rule_name(r):
    return f"R{KEPT.index(r) + 1 if r in KEPT else r}"


@torch.no_grad()
def rule_sentence(r):
    i, j, k = anfis.grid[r].tolist()
    proto = torch.tensor([[anfis.c[0, i], anfis.c[1, j], float(k), 1.0]])
    p_rule = float(torch.sigmoid(proto @ anfis.conseq[r]))
    return (f"IF {NICE['pos_score']} is {MF_NAMES[0][i]} AND {NICE['neg_score']} is "
            f"{MF_NAMES[1][j]} AND negation is {NEG_LABELS[k]} "
            f"THEN sentiment is {'POSITIVE' if p_rule >= 0.5 else 'NEGATIVE'}")


@torch.no_grad()
def analyze(sentence):
    """All results for one sentence as a dict (used by explain() and test_cases.py)."""
    sentence = (sentence or "").strip()
    if not sentence:
        raise ValueError("empty sentence")
    enc = tok([sentence], truncation=True, max_length=128, return_tensors="pt")
    out = bert(**enc)
    probs = out.logits.softmax(-1)[0]
    label = int(probs.argmax())
    pred = f"{'POSITIVE' if label else 'NEGATIVE'}  (confidence {float(probs[label]):.1%})"
    truncated = len(tok(sentence)["input_ids"]) > MAX_TOKENS
    if truncated:
        pred += (f"\n\nWarning: input exceeds {MAX_TOKENS} tokens; everything after the limit "
                 f"was ignored by DistilBERT and by the features.")

    # Same feature definition as features.py
    pos = neg = 0.0
    for word, w in word_attention(enc, out.attentions[-1], 0):
        if word in POS_WORDS:
            pos += w
        elif word in NEG_WORDS:
            neg += w
    negation = int(bool(NEGATION_RE.search(sentence)))
    x = torch.tensor([[pos, neg, float(negation)]], dtype=torch.float32)

    # Membership degrees (ordered Low/Medium/High by centre rank)
    mu = anfis.memberships(x)[0]  # (2, 3), shoulder MFs
    rows = []
    for i, (name, val) in enumerate([("pos_score", pos), ("neg_score", neg)]):
        deg = {MF_NAMES[i][m]: float(mu[i, m]) for m in range(3)}
        rows.append([NICE[name], round(val, 3)] + [round(deg[l], 2) for l in MF_LABELS])
    rows.append(["negation", negation, 1.0 - negation, "-", negation])
    table = pd.DataFrame(rows, columns=["input", "value", "Low / Absent", "Medium", "High / Present"])

    # Deciding rule = largest (normalised firing x consequent) toward the ANFIS output
    logit, wn, contrib, decide = deciding_rule(anfis, x)
    r = int(decide[0])
    anfis_label = int(logit[0] > 0)
    raw = anfis.firing(x)[0]  # un-normalised firing (masked rules = 0)
    rule = (f"{rule_name(r)}  (raw firing {float(raw[r]):.3f}, normalised {float(wn[0, r]):.2f}, "
            f"contribution {float(contrib[0, r]):+.2f} logit)\n{rule_sentence(r)}")
    if float(raw[r]) < LOW_COVERAGE:
        rule += (f"\n\nLow rule coverage: no kept rule closely matches this input "
                 f"(best raw match {float(raw.max()):.3f})")
    rule += (f"\n\nANFIS output: {'POSITIVE' if anfis_label else 'NEGATIVE'} "
             f"(logit {float(logit[0]):+.2f}; {'agrees' if anfis_label == label else 'DISAGREES'} "
             f"with DistilBERT)")

    top = wn[0].argsort(descending=True)[:3].tolist()
    top_df = pd.DataFrame([[rule_name(t), rule_sentence(t), round(float(wn[0, t]), 3),
                            round(float(contrib[0, t]), 3)] for t in top],
                          columns=["rule", "IF-THEN", "firing weight", "contribution (logit)"])
    no_evidence = pos == 0.0 and neg == 0.0
    if no_evidence:  # features carry no sentiment evidence: do not present a rule as an explanation
        rule = (f"No lexicon evidence: ANFIS defaults to {'POSITIVE' if anfis_label else 'NEGATIVE'} "
                f"({'agrees' if anfis_label == label else 'DISAGREES'} with DistilBERT). "
                f"No rule explanation is shown.")
        top_df = None
    return {"label": label, "conf": float(probs[label]), "anfis_label": anfis_label,
            "rule": rule_name(r), "raw": float(raw[r]), "best_raw": float(raw.max()),
            "low_coverage": float(raw[r]) < LOW_COVERAGE, "pos": pos, "neg": neg,
            "negation": negation, "n_tokens": int(enc["input_ids"].shape[1]),
            "truncated": truncated, "no_evidence": no_evidence,
            "pred_text": pred, "table": table, "rule_text": rule, "top_df": top_df}


def explain(sentence):
    if not (sentence or "").strip():
        return "Enter a sentence.", None, "", None
    a = analyze(sentence)
    return a["pred_text"], a["table"], a["rule_text"], a["top_df"]


demo = gr.Interface(
    fn=explain,
    inputs=gr.Textbox(label="Sentence", lines=2, placeholder="a charming and often affecting journey"),
    outputs=[gr.Textbox(label="DistilBERT prediction"),
             gr.Dataframe(label="ANFIS inputs and fuzzy memberships"),
             gr.Textbox(label="Deciding rule", lines=5),
             gr.Dataframe(label="Top 3 rules by firing strength")],
    title="Fuzzy-rule explanation of a DistilBERT sentiment classifier",
    examples=[["it 's a charming and often affecting journey ."],
              ["unflinchingly bleak and desperate"],
              ["this is not a good movie ."]],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
