"""Compare ANFIS fuzzy rules, LIME and SHAP as explanations of DistilBERT (SST-2).

Uses the sentences present in BOTH results/lime.json and results/shap.json.
(1) Surrogate agreement with DistilBERT's label:
      ANFIS (global rule base) | LIME local surrogate (surrogate_label) |
      SHAP: exact by construction (base + sum of all values = f(x)), plus a top-5-only
      reconstruction (base + sum of the stored top-5 values > 0.5) for a fairer reading.
(2) Deletion fidelity: delete top-k evidence words (k = 1, 3, 5), re-run DistilBERT,
      measure flip rate and drop in P(originally predicted class).
      LIME/SHAP evidence = stored top words ranked by score toward the predicted class.
      ANFIS evidence = words behind the DECIDING rule's non-Low conditions (deciding rule =
      largest normalised firing x consequent in the direction of the ANFIS output; same
      definition as app.py via anfis.deciding_rule):
        pos-attention Medium/High -> positive-lexicon words, neg-attention Medium/High ->
        negative-lexicon words, negation Present -> negation tokens; ranked by CLS attention.
      Random = k random words (mean of 5 draws) as a floor.
(3) Interpretability: rules / conditions vs feature scores per explanation.

Outputs: results/comparison.csv, results/deletion_fidelity.png, results/agreement.png,
         results/interpretability.png, results/example.png, results/example.txt
"""
import json
import os
import random
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from anfis import ANFIS, INPUTS, deciding_rule
from features import NEG_WORDS, NEGATION_RE, POS_WORDS, word_attention

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
R = "results"
KS = [1, 3, 5]
N_RANDOM = 5
BATCH = 64
SEED = 0
COL = {"ANFIS": "#2a78d6", "LIME": "#eb6834", "SHAP": "#1baf7a", "Random": "#8a8984"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NICE = {0: "positive-word attention", 1: "negative-word attention"}
NEG_LABELS = ["Absent", "Present"]

random.seed(SEED)
torch.manual_seed(SEED)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, output_attentions=True).eval()


# ---------------------------------------------------------------- helpers
@torch.no_grad()
def p_pos(texts):
    out = []
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], padding=True, truncation=True, max_length=128,
                  return_tensors="pt")
        out.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
    return np.concatenate(out) if out else np.array([])


@torch.no_grad()
def cls_attention(sent):
    """{word: CLS attention} using the exact feature-extraction routine."""
    enc = tok([sent], truncation=True, max_length=128, return_tensors="pt")
    att = {}
    for w, a in word_attention(enc, model(**enc).attentions[-1], 0):
        att[w] = att.get(w, 0.0) + a
    return att


def norm(w):
    return re.sub(r"^\W+|\W+$", "", w.lower())


def matches(token, target):
    t, g = norm(token), norm(target)
    return bool(g) and (t == g or g in re.findall(r"\w+", token.lower()))


def delete_words(sent, targets):
    toks = sent.split()
    keep = [t for t in toks if not any(matches(t, g) for g in targets)]
    return " ".join(keep), len(toks) - len(keep)


def load_anfis():
    ck = torch.load(os.path.join(R, "anfis_model.pt"), weights_only=False)
    m = ANFIS(torch.zeros(2, 3), torch.ones(2, 3))
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck["mf_names"], ck["kept_rules"]


def rule_text(m, names, r):
    i, j, k = m.grid[r].tolist()
    proto = torch.tensor([[m.c[0, i], m.c[1, j], float(k), 1.0]])
    p = float(torch.sigmoid(proto @ m.conseq[r]))
    return (f"IF {NICE[0]} is {names[0][i]} AND {NICE[1]} is {names[1][j]} AND negation is "
            f"{NEG_LABELS[k]} THEN {'POSITIVE' if p >= 0.5 else 'NEGATIVE'}"), (i, j, k)


def anfis_evidence(sent, conds, names):
    """Words behind the fired rule's non-Low conditions, ranked by CLS attention."""
    i, j, k = conds
    att = cls_attention(sent)
    words = set()
    for t in sent.split():
        w = t.lower()
        if names[0][i] != "Low" and w in POS_WORDS:
            words.add(w)
        if names[1][j] != "Low" and w in NEG_WORDS:
            words.add(w)
        if k == 1 and NEGATION_RE.search(t):
            words.add(w)
    return sorted(words, key=lambda w: att.get(w, att.get(norm(w), 0.0)), reverse=True)


def toward_pred(top_words, label):
    s = 1 if label == 1 else -1
    return [d["word"] for d in sorted(top_words, key=lambda d: s * d["score"], reverse=True)]


# ---------------------------------------------------------------- main
def main():
    with open(os.path.join(R, "lime.json")) as f:
        lime = {r["idx"]: r for r in json.load(f)}
    with open(os.path.join(R, "shap.json")) as f:
        shp = {r["idx"]: r for r in json.load(f)}
    ids = [i for i in lime if i in shp]
    feats = pd.read_csv(os.path.join(R, "features.csv")).set_index("idx").loc[ids]
    sents = feats.sentence.tolist()
    n = len(ids)

    base = p_pos(sents)
    label = (base > 0.5).astype(int)
    print(f"Shared sentences: {n} | label mismatch vs features.csv: "
          f"{int((label != feats.model_label.values).sum())}")

    # ---- (1) surrogate agreement
    anf, names, kept = load_anfis()
    with torch.no_grad():
        x = torch.tensor(feats[INPUTS].values, dtype=torch.float32)
        logit, wn, contrib, decide = deciding_rule(anf, x)
    anfis_lab = (logit > 0).int().numpy()
    fired = decide.tolist()  # "deciding rule": largest firing x consequent toward the ANFIS output
    lime_lab = np.array([lime[i]["surrogate_label"] for i in ids])
    shap_top5 = np.array([int(shp[i]["base_value"] + sum(d["score"] for d in shp[i]["top_words"]) > 0.5)
                          for i in ids])
    agree = {
        "ANFIS": float((anfis_lab == label).mean()),
        "LIME": float((lime_lab == label).mean()),
        "SHAP (exact)": 1.0,
        "SHAP (top-5 only)": float((shap_top5 == label).mean()),
    }
    anfis_vs_lime = float((anfis_lab == lime_lab).mean())

    # ---- (2) deletion fidelity
    rule_info = [rule_text(anf, names, r) for r in fired]
    evid = {
        "LIME": [toward_pred(lime[i]["top_words"], l) for i, l in zip(ids, label)],
        "SHAP": [toward_pred(shp[i]["top_words"], l) for i, l in zip(ids, label)],
        "ANFIS": [anfis_evidence(s, c, names) for s, (_, c) in zip(sents, rule_info)],
    }
    anfis_cov = float(np.mean([len(e) > 0 for e in evid["ANFIS"]]))

    def score(texts, removed, src_idx):
        p = p_pos(texts)
        p0, l0 = base[src_idx], label[src_idx]
        conf0 = np.where(l0 == 1, p0, 1 - p0)
        conf1 = np.where(l0 == 1, p, 1 - p)
        return {"flip_rate": float(((p > 0.5).astype(int) != l0).mean()),
                "conf_drop": float((conf0 - conf1).mean()),
                "words_removed": float(np.mean(removed))}

    rows, del_res = [], {}
    for k in KS:
        for meth in ["LIME", "SHAP", "ANFIS"]:
            out = [delete_words(s, e[:k]) for s, e in zip(sents, evid[meth])]
            del_res[(meth, k)] = score([o[0] for o in out], [o[1] for o in out], np.arange(n))
        texts, removed, src = [], [], []
        for d in range(N_RANDOM):
            rng = random.Random(SEED + d)
            for si, s in enumerate(sents):
                toks = s.split()
                drop = set(rng.sample(range(len(toks)), min(k, len(toks))))
                texts.append(" ".join(t for ti, t in enumerate(toks) if ti not in drop))
                removed.append(len(drop))
                src.append(si)
        del_res[("Random", k)] = score(texts, removed, np.array(src))
    out = [delete_words(s, e) for s, e in zip(sents, evid["ANFIS"])]
    del_res[("ANFIS", "all")] = score([o[0] for o in out], [o[1] for o in out], np.arange(n))

    for (meth, k), v in del_res.items():
        for met, val in v.items():
            rows.append(("deletion", meth, f"k={k}", met, round(val, 4)))
    for meth, v in agree.items():
        rows.append(("agreement", meth, "", "agreement_with_distilbert", round(v, 4)))
    rows.append(("agreement", "ANFIS vs LIME", "", "label_agreement", round(anfis_vs_lime, 4)))
    rows.append(("agreement", "LIME", "", "mean_surrogate_r2",
                 round(float(np.mean([lime[i]["r2"] for i in ids])), 4)))

    # ---- (3) interpretability
    n_words = np.mean([len(s.split()) for s in sents])
    n_rules = len(kept)
    overlap3 = np.mean([len({norm(w) for w in evid["LIME"][q][:3]} & {norm(w) for w in evid["SHAP"][q][:3]}) / 3
                        for q in range(n)])
    interp = {
        "ANFIS": {"global_rules": n_rules, "global_conditions": n_rules * 3,
                  "units_per_explanation": 3, "units_kind": "conditions in fired rule",
                  "distinct_explanations": len(set(fired)), "evidence_coverage": anfis_cov},
        "LIME": {"global_rules": 0, "global_conditions": 0,
                 "units_per_explanation": len(lime[ids[0]]["top_words"]),
                 "units_kind": "word scores (top-k shown)", "distinct_explanations": n},
        "SHAP": {"global_rules": 0, "global_conditions": 0,
                 "units_per_explanation": round(float(n_words), 2),
                 "units_kind": "word scores (every word scored)", "distinct_explanations": n},
    }
    for meth, d in interp.items():
        for met, val in d.items():
            rows.append(("interpretability", meth, "", met, val))
    rows.append(("interpretability", "LIME vs SHAP", "", "top3_word_overlap", round(float(overlap3), 4)))
    pd.DataFrame(rows, columns=["section", "method", "k", "metric", "value"]).to_csv(
        os.path.join(R, "comparison.csv"), index=False)

    # ---- charts
    def finish(ax):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # deletion fidelity: two panels, one per measure (no dual axis)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, met, title in zip(axes, ["flip_rate", "conf_drop"],
                              ["Prediction flip rate", "Drop in confidence (predicted class)"]):
        for meth in ["ANFIS", "LIME", "SHAP", "Random"]:
            ys = [del_res[(meth, k)][met] for k in KS]
            ax.plot(KS, ys, marker="o", ms=8, lw=2, color=COL[meth], label=meth,
                    ls="--" if meth == "Random" else "-", mec="#fcfcfb", mew=2)
            ax.annotate(meth, (KS[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK2, fontsize=9)
        ax.set_xticks(KS)
        ax.set_xlabel("words deleted (k)")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlim(0.7, 5.9)
        finish(ax)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Deletion fidelity on {n} test sentences (higher = explanation found what the model uses)",
                 x=0.01, ha="left", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "deletion_fidelity.png"), dpi=150)
    plt.close(fig)

    # agreement
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    labs = list(agree)
    cols = [COL["ANFIS"], COL["LIME"], COL["SHAP"], COL["SHAP"]]
    bars = ax.bar(labs, [agree[l] for l in labs], color=cols, width=0.6, edgecolor="#fcfcfb", lw=2)
    bars[2].set_hatch("//")
    bars[2].set_alpha(0.55)
    for b, l in zip(bars, labs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{agree[l]:.2f}",
                ha="center", color=INK, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("agreement with DistilBERT label")
    ax.set_title("Surrogate agreement (SHAP exact = 1.0 by construction)", loc="left", color=INK, fontsize=11)
    finish(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "agreement.png"), dpi=150)
    plt.close(fig)

    # interpretability
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    meths = ["ANFIS", "LIME", "SHAP"]
    vals = [interp[m]["units_per_explanation"] for m in meths]
    axes[0].bar(meths, vals, color=[COL[m] for m in meths], width=0.6)
    for xi, v in enumerate(vals):
        axes[0].text(xi, v + 0.15, f"{v:g}", ha="center", color=INK, fontsize=9)
    axes[0].set_title("Items to read per explanation", loc="left", color=INK, fontsize=11)
    axes[0].set_ylabel("conditions (ANFIS) / word scores")
    vals = [interp[m]["distinct_explanations"] for m in meths]
    axes[1].bar(meths, vals, color=[COL[m] for m in meths], width=0.6)
    for xi, v in enumerate(vals):
        axes[1].text(xi, v + 0.8, f"{v}", ha="center", color=INK, fontsize=9)
    axes[1].set_title(f"Distinct explanations for {n} sentences", loc="left", color=INK, fontsize=11)
    for ax in axes:
        finish(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "interpretability.png"), dpi=150)
    plt.close(fig)

    # ---- side-by-side example: prefer a sentence ANFIS gets right, with evidence words and negation
    cand = [q for q in range(n) if anfis_lab[q] == label[q] and evid["ANFIS"][q]]
    cand_neg = [q for q in cand if feats.negation.values[q] == 1]
    q = (cand_neg or cand or [0])[0]
    idx, s = ids[q], sents[q]
    rtxt = rule_info[q][0]
    rnum = kept.index(fired[q]) + 1 if fired[q] in kept else fired[q]
    lw, sw = lime[idx]["top_words"], shp[idx]["top_words"]
    pred = "POSITIVE" if label[q] else "NEGATIVE"
    txt = [f"Sentence [{idx}]: {s}",
           f"DistilBERT: {pred} (P(pos)={base[q]:.3f})", "",
           f"ANFIS deciding rule R{rnum} (firing {float(wn[q, fired[q]]):.2f}, "
           f"contribution {float(contrib[q, fired[q]]):+.2f} logit): {rtxt}",
           f"  inputs: pos={feats.pos_score.values[q]:.3f} neg={feats.neg_score.values[q]:.3f} "
           f"negation={int(feats.negation.values[q])} | evidence words: {', '.join(evid['ANFIS'][q])}",
           "LIME: " + ", ".join(f"{d['word']} {d['score']:+.3f}" for d in lw)
           + f"  (surrogate={'POS' if lime[idx]['surrogate_label'] else 'NEG'})",
           "SHAP: " + ", ".join(f"{d['word']} {d['score']:+.3f}" for d in sw)]
    with open(os.path.join(R, "example.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")

    fig = plt.figure(figsize=(13, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])
    fig.suptitle(f"\"{s}\"   ->   DistilBERT: {pred} ({max(base[q], 1 - base[q]):.2f})",
                 x=0.01, ha="left", color=INK, fontsize=11)
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    ax.set_title("ANFIS (fuzzy rule)", loc="left", color=COL["ANFIS"], fontsize=11, fontweight="bold")
    parts = rtxt.replace(" AND ", "\nAND ").replace(" THEN ", "\nTHEN ")
    ax.text(0, 0.95, f"Deciding rule R{rnum}  (firing {float(wn[q, fired[q]]):.2f}, "
            f"contrib {float(contrib[q, fired[q]]):+.2f})\n{parts}\n\n"
            f"evidence: {', '.join(evid['ANFIS'][q])}", va="top", color=INK, fontsize=9.5,
            family="monospace", transform=ax.transAxes)
    for g, (name, ws) in enumerate([("LIME", lw), ("SHAP", sw)], start=1):
        ax = fig.add_subplot(gs[g])
        ws = ws[::-1]
        vals = [d["score"] for d in ws]
        ax.barh([d["word"] for d in ws], vals, color=COL[name], height=0.6)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{name} (score toward positive)", loc="left", color=COL[name],
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "example.png"), dpi=150)
    plt.close(fig)

    # ---- summary print
    print("\nAgreement with DistilBERT:", {k: round(v, 3) for k, v in agree.items()},
          f"| ANFIS vs LIME surrogate: {anfis_vs_lime:.3f}")
    tab = pd.DataFrame({f"{m} k={k}": del_res[(m, k)] for m in ["ANFIS", "LIME", "SHAP", "Random"]
                        for k in KS}).T.round(3)
    print("\nDeletion fidelity:\n" + tab.to_string())
    a = del_res[("ANFIS", "all")]
    print(f"ANFIS all evidence words: flip {a['flip_rate']:.3f}, drop {a['conf_drop']:.3f}, "
          f"removed {a['words_removed']:.2f} | evidence coverage {anfis_cov:.2f}")
    print(f"Interpretability: ANFIS {n_rules} rules x 3 conditions ({len(set(fired))} distinct deciding rules); "
          f"LIME {interp['LIME']['units_per_explanation']} scores; SHAP {n_words:.1f} scores/sentence; "
          f"LIME-SHAP top-3 overlap {overlap3:.2f}")
    print("\n" + "\n".join(txt))
    print("\nSaved results/comparison.csv, deletion_fidelity.png, agreement.png, "
          "interpretability.png, example.png, example.txt")


if __name__ == "__main__":
    main()
