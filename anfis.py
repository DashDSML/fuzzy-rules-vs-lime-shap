"""First-order Takagi-Sugeno ANFIS (hand-written, PyTorch) that mimics DistilBERT's labels.

Inputs : pos_score, neg_score (Low/Medium/High MFs: Gaussian with shoulders, i.e. Low=1 below
         its centre and High=1 above its centre, centres initialised from
         10th/50th/90th percentiles of non-zero training values), negation (crisp Absent/Present).
Rules  : full grid 3 x 3 x 2 = 18; consequent f_r = a*pos + b*neg + c*negation + d (logit).
Output : sum_r  wn_r * f_r  -> sigmoid -> P(positive). Target = DistilBERT model_label.
Pruning: keep the TOP_K rules with the highest mean normalised firing strength on train, then
         (coverage-aware) add the best-ranked remaining rules until every membership label
         (pos L/M/H, neg L/M/H, negation Absent/Present) appears in a kept rule, up to MAX_RULES;
         then fine-tune the pruned model.

Saves results/anfis_rules.txt, results/anfis_metrics.json, results/anfis_model.pt, results/split.json
"""
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
TOP_K = 6
MAX_RULES = 8
EPOCHS = 1500
FINETUNE_EPOCHS = 500
LR = 0.02
MIN_SIGMA = 0.02
INPUTS = ["pos_score", "neg_score", "negation"]  # confidence deliberately excluded (leakage)
MF_LABELS = ["Low", "Medium", "High"]
NEG_LABELS = ["Absent", "Present"]
NICE = {"pos_score": "positive-word attention", "neg_score": "negative-word attention"}

torch.manual_seed(SEED)
np.random.seed(SEED)


class ANFIS(nn.Module):
    def __init__(self, centres, sigmas):
        super().__init__()
        self.c = nn.Parameter(centres.clone())             # (2 inputs, 3 MFs)
        self.log_s = nn.Parameter(sigmas.clone().log())    # widths kept positive via exp
        grid = torch.cartesian_prod(torch.arange(3), torch.arange(3), torch.arange(2))
        self.register_buffer("grid", grid)                 # (18, 3): pos MF, neg MF, negation
        self.conseq = nn.Parameter(torch.zeros(len(grid), 4))
        self.register_buffer("mask", torch.ones(len(grid)))

    def memberships(self, x):
        """(N, 2, 3) degrees per MF index. Shoulders: the lowest-centre MF is 1 for x <= its
        centre, the highest-centre MF is 1 for x >= its centre; Gaussian elsewhere."""
        s = self.log_s.exp()
        xi = x[:, :2, None]
        mu = torch.exp(-0.5 * ((xi - self.c[None]) / s[None]) ** 2)
        rank = self.c.argsort(1).argsort(1)  # (2,3): 0 = Low, 1 = Medium, 2 = High
        shoulder = ((xi <= self.c[None]) & (rank == 0)[None]) | ((xi >= self.c[None]) & (rank == 2)[None])
        return torch.where(shoulder, torch.ones_like(mu), mu)

    def firing(self, x):
        mu = self.memberships(x)
        mu_neg = torch.stack([1 - x[:, 2], x[:, 2]], 1)                            # (N,2)
        g = self.grid
        w = mu[:, 0, g[:, 0]] * mu[:, 1, g[:, 1]] * mu_neg[:, g[:, 2]]             # (N,R)
        return w * self.mask

    def forward(self, x):
        w = self.firing(x)
        wn = w / (w.sum(1, keepdim=True) + 1e-12)
        f = torch.cat([x, torch.ones(len(x), 1)], 1) @ self.conseq.T              # (N,R)
        return (wn * f).sum(1), wn, f


def train(model, x, y, epochs, params=None):
    opt = torch.optim.Adam(params or model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(x)[0], y)
        loss.backward()
        opt.step()
        with torch.no_grad():  # keep MFs inside the observed data range so labels stay meaningful
            model.c.clamp_(min=torch.zeros(2, 1), max=x[:, :2].max(0).values[:, None])
            model.log_s.clamp_(min=float(np.log(MIN_SIGMA)))
    return loss.item()


@torch.no_grad()
def deciding_rule(model, x):
    """Per sentence: the rule whose contribution (normalised firing x consequent logit) is
    largest in the direction of the ANFIS output. Returns (logit, wn, contrib, deciding_idx).
    Shared by app.py and compare.py so the demo and the evaluation use one definition."""
    logit, wn, f = model(x)
    contrib = wn * f
    sign = torch.where(logit > 0, 1.0, -1.0)[:, None]
    return logit, wn, contrib, (sign * contrib).argmax(1)


def select_rules(ranking, grid, k=TOP_K, max_rules=MAX_RULES):
    """Top-k by firing strength, then greedily add the best-ranked rule covering the most
    still-uncovered membership labels until all are covered (or max_rules reached).
    Coverage is by MF index (pos 0-2, neg 0-2, negation 0-1). Returns (kept, uncovered)."""
    def labels(r):
        i, j, n = grid[r].tolist()
        return {("pos", i), ("neg", j), ("negation", n)}
    need = ({("pos", i) for i in range(3)} | {("neg", j) for j in range(3)}
            | {("negation", 0), ("negation", 1)})
    kept = list(ranking[:k])
    covered = set().union(*(labels(r) for r in kept))
    while need - covered and len(kept) < max_rules:
        cand = [r for r in ranking if r not in kept and labels(r) - covered]
        if not cand:
            break
        best = max(cand, key=lambda r: (len(labels(r) - covered), -ranking.index(r)))
        kept.append(best)
        covered |= labels(best)
    return kept, sorted(need - covered)


@torch.no_grad()
def fidelity(model, x, y):
    return float(((model(x)[0] > 0).float() == y).float().mean())


def mf_names(model):
    """Map MF index -> Low/Medium/High by centre rank (robust if centres cross during training)."""
    names = []
    for i in range(2):
        order = torch.argsort(model.c[i]).tolist()
        names.append({mf: MF_LABELS[r] for r, mf in enumerate(order)})
    return names


@torch.no_grad()
def describe_rules(model, x, y, rule_ids):
    names = mf_names(model)
    _, wn, _ = model(x)
    dominant = wn.argmax(1)
    out = []
    for r in rule_ids:
        i, j, k = model.grid[r].tolist()
        proto = torch.tensor([[model.c[0, i], model.c[1, j], float(k)]])
        p_pos = float(torch.sigmoid(torch.cat([proto, torch.ones(1, 1)], 1) @ model.conseq[r]))
        sel = dominant == r
        pred = (model(x)[0] > 0).float()
        out.append({
            "rule": r,
            "text": (f"IF {NICE['pos_score']} is {names[0][i]} AND {NICE['neg_score']} is "
                     f"{names[1][j]} AND negation is {NEG_LABELS[k]} THEN sentiment is "
                     f"{'POSITIVE' if p_pos >= 0.5 else 'NEGATIVE'} (p_pos={p_pos:.2f})"),
            "p_pos_at_prototype": round(p_pos, 3),
            "mean_firing": round(float(wn[:, r].mean()), 3),
            "dominant_n": int(sel.sum()),
            "dominant_agreement": round(float((pred[sel] == y[sel]).float().mean()), 3) if sel.any() else None,
        })
    return out


def main():
    df = pd.read_csv(os.path.join("results", "features.csv"))
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df.model_label)
    xtr = torch.tensor(tr[INPUTS].values, dtype=torch.float32)
    xte = torch.tensor(te[INPUTS].values, dtype=torch.float32)
    ytr = torch.tensor(tr.model_label.values, dtype=torch.float32)
    yte = torch.tensor(te.model_label.values, dtype=torch.float32)

    centres, sigmas = [], []
    for col in ["pos_score", "neg_score"]:
        nz = tr[col][tr[col] > 0]
        q = np.quantile(nz, [0.1, 0.5, 0.9])
        centres.append(q)
        sigmas.append(np.full(3, max((q[2] - q[0]) / 4, 1e-3)))
    centres = torch.tensor(np.array(centres), dtype=torch.float32)
    sigmas = torch.tensor(np.array(sigmas), dtype=torch.float32)
    print("Init centres (10/50/90 pct of non-zero):",
          {c: [round(v, 3) for v in centres[i].tolist()] for i, c in enumerate(["pos", "neg"])})

    full = ANFIS(centres, sigmas)
    loss = train(full, xtr, ytr, EPOCHS)
    fid_full_tr, fid_full_te = fidelity(full, xtr, ytr), fidelity(full, xte, yte)

    with torch.no_grad():
        mean_fire = full(xtr)[1].mean(0)
    ranking = torch.argsort(mean_fire, descending=True).tolist()

    def pruned_model(rule_ids):
        m = copy.deepcopy(full)
        m.mask.zero_()
        m.mask[rule_ids] = 1
        train(m, xtr, ytr, FINETUNE_EPOCHS)
        return m

    k_curve = {}
    for k in [2, 3, 4, 6, 8, 12, 18]:
        mk = pruned_model(ranking[:k])
        k_curve[k] = {"train": round(fidelity(mk, xtr, ytr), 3), "test": round(fidelity(mk, xte, yte), 3)}
    kept, uncovered = select_rules(ranking, full.grid)
    pruned = pruned_model(kept)
    fid_pr_tr, fid_pr_te = fidelity(pruned, xtr, ytr), fidelity(pruned, xte, yte)

    # Reference points on the same 3 inputs
    majority = float((yte == ytr.mode().values).float().mean())
    lr = LogisticRegression().fit(tr[INPUTS], tr.model_label)
    fid_lr = float(lr.score(te[INPUTS], te.model_label))

    rules = describe_rules(pruned, xte, yte, kept)
    names = mf_names(pruned)
    mfs = {col: {names[i][m]: {"centre": round(float(pruned.c[i, m]), 4),
                               "sigma": round(float(pruned.log_s[i, m].exp()), 4)} for m in range(3)}
           for i, col in enumerate(["pos_score", "neg_score"])}

    metrics = {
        "inputs": INPUTS, "n_train": len(tr), "n_test": len(te),
        "final_train_loss": round(loss, 4),
        "fidelity_full_18_rules": {"train": round(fid_full_tr, 3), "test": round(fid_full_te, 3)},
        "fidelity_pruned": {"train": round(fid_pr_tr, 3), "test": round(fid_pr_te, 3)},
        "fidelity_vs_k": k_curve,
        "baseline_majority_test": round(majority, 3),
        "baseline_logreg_test": round(fid_lr, 3),
        "n_rules": len(kept), "conditions_per_rule": 3,
        "uncovered_labels": uncovered, "top_k_before_coverage": TOP_K,
        "membership_functions": mfs,
        "rules": rules,
    }
    with open(os.path.join("results", "anfis_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join("results", "anfis_rules.txt"), "w", encoding="utf-8") as f:
        f.write(f"ANFIS (first-order TSK) - {len(kept)} rules (coverage-aware), test fidelity {fid_pr_te:.3f}\n\n")
        for n, r in enumerate(rules, 1):
            f.write(f"R{n}: {r['text']}\n    mean firing {r['mean_firing']:.3f} | dominant on "
                    f"{r['dominant_n']} test sentences, agreement {r['dominant_agreement']}\n")
        f.write("\nMembership functions (centre, sigma):\n" + json.dumps(mfs, indent=2) + "\n")
    torch.save({"state_dict": pruned.state_dict(), "inputs": INPUTS, "mf_names": names,
                "kept_rules": kept}, os.path.join("results", "anfis_model.pt"))
    with open(os.path.join("results", "split.json"), "w") as f:
        json.dump({"train_idx": tr.idx.tolist(), "test_idx": te.idx.tolist()}, f)

    print(f"Train/test: {len(tr)}/{len(te)} | final loss {loss:.4f}")
    print(f"Fidelity vs DistilBERT (test): full 18 rules {fid_full_te:.3f} | "
          f"pruned {len(kept)} rules {fid_pr_te:.3f}")
    print(f"Baselines (test): majority {majority:.3f} | logistic regression {fid_lr:.3f}")
    print("Fidelity vs #rules (test):", {k: v["test"] for k, v in k_curve.items()})
    print(f"\n{len(kept)} kept rules (top {TOP_K} by firing + coverage additions; "
          f"uncovered labels: {uncovered or 'none'}):")
    for n, r in enumerate(rules, 1):
        print(f"R{n}: {r['text']}\n    firing {r['mean_firing']:.3f} | dominant on "
              f"{r['dominant_n']} test sents, agreement {r['dominant_agreement']}")


if __name__ == "__main__":
    main()
