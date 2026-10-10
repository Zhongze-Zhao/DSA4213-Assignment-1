"""Part III error analysis from saved predictions (no model needed).

    python -m src.error_analysis
Writes results/downstream/errors.md and results/downstream/error_stats.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.utils import ROOT, save_json

D = ROOT / "results/downstream"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
PAIRS = [("transformer/mean", "lstm/mean"), ("lstm/mean", "rnn/mean"),
         ("lstm_w2v/mean", "lstm/mean"), ("lstm_glove/mean", "lstm/mean"),
         ("lstm_glove/mean", "lstm_w2v/mean"), ("lstm/emb", "lstm/mean"),
         ("lstm/mean", "lstm/last"), ("transformer/emb", "transformer/mean"),
         ("lstm_smallinit/mean", "lstm/mean"), ("lstm_smallinit/emb", "lstm/emb"),
         ("lstm_glove/mean", "lstm_smallinit/mean")]


def mcnemar(a_ok, b_ok):
    """Exact McNemar test on the discordant pairs."""
    b01 = int((a_ok & ~b_ok).sum())
    b10 = int((~a_ok & b_ok).sum())
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"a_only_correct": b01, "b_only_correct": b10, "p_value": float(p)}


def main():
    df = pd.read_csv(D / "test_predictions.csv")
    cols = [c for c in df.columns if "/" in c]
    ok = {c: (df[c] == df.label).to_numpy() for c in cols}
    PAIRS[:] = [(a, b) for a, b in PAIRS if a in ok and b in ok]
    stats = {"mcnemar": {f"{a} vs {b}": {"acc_a": float(ok[a].mean()), "acc_b": float(ok[b].mean()),
                                         **mcnemar(ok[a], ok[b])} for a, b in PAIRS}}

    # accuracy by input length (tokens after WikiText-style tokenisation)
    bins = pd.qcut(df.length, 4)
    stats["accuracy_by_length_quartile"] = {
        c: {str(k): round(v, 4) for k, v in pd.Series(ok[c]).groupby(bins.values, observed=False).mean().items()}
        for c in ["lstm/mean", "transformer/mean", "lstm_glove/mean"]}

    # per-class recall
    stats["per_class_recall"] = {c: {lab: float(ok[c][df.label == lab].mean()) for lab in LABELS}
                                 for c in ["rnn/mean", "lstm/mean", "transformer/mean",
                                           "lstm_w2v/mean", "lstm_glove/mean", "lstm_smallinit/mean"]
                                 if c in ok}
    # most common confusions
    conf = {}
    for c in ["lstm/mean", "transformer/mean", "lstm_glove/mean"]:
        wrong = df[~ok[c]]
        conf[c] = (wrong.label + " -> " + wrong[c]).value_counts().head(4).to_dict()
    stats["top_confusions"] = conf
    save_json(stats, D / "error_stats.json")

    rng = np.random.default_rng(0)
    lines = ["# Part III representative errors (AG News test)\n"]

    def show(title, mask, cols_show, k=6):
        idx = np.where(mask)[0]
        lines.append(f"\n## {title} ({len(idx)} examples)\n")
        for i in rng.choice(idx, size=min(k, len(idx)), replace=False):
            r = df.iloc[i]
            preds = ", ".join(f"{c}={r[c]}" for c in cols_show)
            lines.append(f"- [{r.label}] ({preds}; OOV {r.oov_rate:.0%}, {r.length} tok) {r.text[:220]}")

    show("GloVe-LSTM (mean) correct, random-LSTM (mean) wrong",
         ok["lstm_glove/mean"] & ~ok["lstm/mean"], ["lstm/mean", "lstm_glove/mean"])
    show("Transformer (mean) correct, LSTM (mean) wrong",
         ok["transformer/mean"] & ~ok["lstm/mean"], ["lstm/mean", "transformer/mean"])
    show("All five LMs wrong with mean pooling",
         np.all([~ok[f"{m}/mean"] for m in ["rnn", "lstm", "transformer", "lstm_w2v", "lstm_glove"]], 0),
         ["lstm_glove/mean", "transformer/mean"])
    show("Input-embedding mean correct but contextual mean wrong (random LSTM)",
         ok["lstm/emb"] & ~ok["lstm/mean"], ["lstm/mean", "lstm/emb"])
    (D / "errors.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
