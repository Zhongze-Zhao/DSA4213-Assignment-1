"""Build all report tables and figures from the saved json/csv results (no hand-copied numbers).

    python -m src.summarize
Writes results/figures/*.png and results/tables.md
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ROOT

R = ROOT / "results"
FIG = R / "figures"
NEURAL = ["rnn", "lstm", "transformer"]
PART2 = {"lstm": "random N(0,1) (trainable)", "lstm_smallinit": "random N(0,0.1²) (trainable)",
         "lstm_w2v": "own Skip-gram (frozen)",
         "lstm_glove": "GloVe 6B (frozen)"}
COLORS = {"rnn": "#1b9e77", "lstm": "#d95f02", "transformer": "#7570b3", "lstm_smallinit": "#e6ab02",
          "lstm_w2v": "#e7298a", "lstm_glove": "#66a61e"}


def load(name):
    return json.load(open(R / f"lm/{name}/metrics.json"))


def hist(name):
    return pd.read_csv(R / f"lm/{name}/history.csv")


def loss_figure(names, labels, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    for n in names:
        h = hist(n)
        axes[0].plot(h.epoch[1:], h.train_loss[1:], "--", color=COLORS[n], alpha=0.8)
        axes[0].plot(h.epoch, h.val_loss, "-", color=COLORS[n], label=labels[n])
        axes[1].plot(h.epoch[1:], h.val_loss[1:] - h.train_loss[1:], color=COLORS[n], label=labels[n])
    axes[0].set(xlabel="epoch", ylabel="cross-entropy (nats)", title=f"{title}: train (--) / val (—)",
                ylim=(4.0, 7.0))
    axes[1].axhline(0, color="grey", lw=0.6)
    axes[1].set(xlabel="epoch", ylabel="val − train loss", title="generalisation gap")
    axes[0].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    md = []

    # ---------- Part I ----------
    ng = load("ngram")
    rows = [{"model": f"KN {ng['order']}-gram", "test PPL": f"{ng['test_ppl']:.1f}",
             "test CE": f"{ng['test_loss']:.3f}", "val PPL": f"{ng['val_ppl']:.1f}",
             "params / counts (M)": f"{ng['stored_counts'] / 1e6:.2f}†", "non-emb params (M)": "–",
             "train time (s)": f"{ng['train_time_s']:.0f} (CPU)", "median s/epoch": "–",
             "best epoch": "–",
             "test scoring (k tok/s)": f"{ng['inference_tokens_per_s'] / 1e3:.0f} (CPU)"}]
    for n in NEURAL:
        m = load(n)
        rows.append({"model": n.upper() if n != "transformer" else "Transformer",
                     "test PPL": f"{m['test_ppl']:.1f}", "test CE": f"{m['test_loss']:.3f}",
                     "val PPL": f"{m['val_ppl']:.1f}",
                     "params / counts (M)": f"{m['params']['total'] / 1e6:.2f}",
                     "non-emb params (M)": f"{m['params']['non_embedding'] / 1e6:.2f}",
                     "train time (s)": f"{m['train_time_s']:.0f}",
                     "median s/epoch": f"{hist(n).epoch_time_s[1:].median():.1f}",
                     "best epoch": m["best_epoch"],
                     "test scoring (k tok/s)": f"{m['inference_tokens_per_s'] / 1e3:.0f}"})
    gen = json.load(open(R / "generations.json"))
    for r, n in zip(rows, ["ngram"] + NEURAL):
        g = gen["models"][n]
        r["gen ms/token"] = f"{g['ms_per_token_greedy']:.2f}"
    md += ["## Part I: language models", md_table(pd.DataFrame(rows)),
           "† n-gram: number of stored (context, word) count entries, not trainable parameters.\n"]
    loss_figure(NEURAL, {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer"},
                FIG / "part1_loss.png", "Part I")

    # generation repetition table
    grows = []
    for n, g in gen["models"].items():
        for mode in ("greedy", "sample"):
            gm = g["metrics"][mode]
            grows.append({"model": n, "decoding": mode, "distinct-1": f"{gm['distinct1']:.3f}",
                          "distinct-2": f"{gm['distinct2']:.3f}",
                          "repeated 3-gram rate": f"{gm['repeated_trigram_rate']:.3f}"})
    md += ["## Generation repetition metrics (40 tokens; greedy: 5 prompts, top-k: 5x10 samples)",
           md_table(pd.DataFrame(grows))]

    # ---------- Part II ----------
    emb = json.load(open(R / "embeddings.json"))
    rows = []
    for n, lab in [(n, l) for n, l in PART2.items() if (R / f"lm/{n}/metrics.json").exists()]:
        m, h = load(n), hist(n)
        target = h.val_loss.min() + 0.05
        ep_close = int(h[h.val_loss <= target].epoch.min())
        rows.append({"embedding": lab, "test PPL": f"{m['test_ppl']:.1f}",
                     "val PPL": f"{m['val_ppl']:.1f}", "best epoch": m["best_epoch"],
                     "val PPL @ep1": f"{np.exp(h.val_loss[1]):.0f}",
                     "val PPL @ep5": f"{np.exp(h.val_loss[5]):.0f}",
                     "epochs to within 0.05 of best": ep_close,
                     "final gap (val−train)": f"{h.val_loss.iloc[-1] - h.train_loss.iloc[-1]:.2f}",
                     "step-loss std": f"{h.step_loss_std[1:].mean():.3f}",
                     "grad-norm mean/max": f"{h.grad_norm_mean[1:].mean():.2f}/{h.grad_norm_max[1:].max():.2f}",
                     "trainable (M)": f"{m['params']['trainable'] / 1e6:.2f}",
                     "LM train time (s)": f"{m['train_time_s']:.0f}",
                     "median s/epoch": f"{h.epoch_time_s[1:].median():.1f}"})
    md += ["## Part II: LSTM embedding ablation", md_table(pd.DataFrame(rows)),
           f"Own Skip-gram training cost (separate from LM training): "
           f"{emb['word2vec']['train_time_s']:.1f}s on CPU ({emb['word2vec']['hardware']}), "
           f"{emb['word2vec']['epochs']} epochs over {emb['word2vec']['train_tokens']:,} tokens. "
           f"GloVe type coverage {emb['glove']['type_coverage']:.2%}, train-token coverage "
           f"{emb['glove']['token_coverage_train']:.2%}, missing types {emb['glove']['missing_types']}.\n"]
    seeds = sorted(R.glob("lm/*_seed*/metrics.json"))
    if seeds:
        srows = {}
        for p in seeds:
            m = json.load(open(p))
            base = m["name"].rsplit("_seed", 1)[0]
            srows.setdefault(base, [load(base)["test_ppl"]]).append(m["test_ppl"])
        md += ["### Seed variation (test PPL over seeds 42 + extra seeds)",
               md_table(pd.DataFrame([{"model": k, "n seeds": len(v), "mean": f"{np.mean(v):.1f}",
                                       "std": f"{np.std(v, ddof=1):.1f}",
                                       "values": ", ".join(f"{x:.1f}" for x in v)}
                                      for k, v in srows.items()]))]
    loss_figure([n for n in PART2 if (R / f"lm/{n}/history.csv").exists()], PART2,
                FIG / "part2_loss.png", "Part II")

    # ---------- Part III ----------
    ds = json.load(open(R / "downstream/summary.json"))
    df = pd.DataFrame(ds["results"])
    tab = df.pivot(index="model", columns="pooling", values="test_accuracy")
    f1 = df.pivot(index="model", columns="pooling", values="test_macro_f1")
    ppl = df.groupby("model").lm_test_ppl.first()
    rows = []
    for mname in [m for m in ["rnn", "lstm", "transformer", "lstm_smallinit", "lstm_w2v", "lstm_glove"]
                  if m in ppl.index]:
        r = {"LM": mname, "LM test PPL": f"{ppl[mname]:.1f}"}
        for pool in ["last", "mean", "emb"]:
            r[f"{pool} acc / F1"] = f"{tab.loc[mname, pool]:.3f} / {f1.loc[mname, pool]:.3f}"
        rows.append(r)
    md += ["## Part III: AG News (frozen LM features + logistic regression)", md_table(pd.DataFrame(rows)),
           f"Spearman(PPL, acc): {ds['ppl_vs_accuracy']}\n",
           f"OOV: {json.dumps(ds['data'], indent=None)}\n",
           f"Accuracy by per-example OOV-rate bucket: {json.dumps(ds['accuracy_by_oov_bucket'])}\n"]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for pool, mk in (("last", "o"), ("mean", "s")):
        sub = df[df.pooling == pool]
        for _, r in sub.iterrows():
            ax.scatter(r.lm_test_ppl, r.test_accuracy, marker=mk, color=COLORS[r.model], s=45)
        ax.plot([], [], mk, color="grey", label=f"{pool} pooling")
    for mname in ppl.index:
        ax.annotate(mname, (ppl[mname], df[(df.model == mname) & (df.pooling == "mean")].test_accuracy.item()),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set(xlabel="LM test perplexity (lower = better)", ylabel="AG News test accuracy",
           title="Does better LM => better features?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "part3_ppl_vs_acc.png", dpi=200)
    plt.close(fig)

    (R / "tables.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
