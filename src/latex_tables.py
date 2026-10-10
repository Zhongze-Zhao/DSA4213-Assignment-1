"""Generate the LaTeX tables / macros used by report/report.tex from saved results.

    python -m src.latex_tables       # writes report/generated/*.tex
"""
import json

import numpy as np
import pandas as pd

from src.utils import ROOT

R = ROOT / "results"
OUT = ROOT / "report/generated"


def J(p):
    return json.load(open(R / p))


def seed_stats(name):
    vals = [J(f"lm/{name}/metrics.json")["test_ppl"]]
    vals += [json.load(open(p))["test_ppl"] for p in sorted(R.glob(f"lm/{name}_seed*/metrics.json"))]
    return vals


def pm(vals):
    return f"{np.mean(vals):.1f} $\\pm$ {np.std(vals, ddof=1):.1f}" if len(vals) > 1 else "--"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bench = J("benchmark.json")["models"] if (R / "benchmark.json").exists() else {}
    gen = J("generations.json")["models"]
    macros = []

    # ---------------- Part I ----------------
    ng = J("lm/ngram/metrics.json")
    rows = []
    g = gen["ngram"]
    rows.append(f"KN {ng['order']}-gram & {ng['test_ppl']:.1f} & (deterministic) & "
                f"{ng['stored_counts'] / 1e6:.1f}$^\\dagger$ & -- & {ng['train_time_s']:.0f} (CPU) & -- & "
                f"{ng['inference_tokens_per_s'] / 1e3:.0f} (CPU) & {g['ms_per_token_greedy']:.1f} & "
                f"{g['metrics']['greedy']['repeated_trigram_rate']:.2f} & {g['metrics']['sample']['distinct2']:.2f} \\\\")
    for n, lab in [("rnn", "RNN"), ("lstm", "LSTM"), ("transformer", "Transformer")]:
        m, g = J(f"lm/{n}/metrics.json"), gen[n]
        h = pd.read_csv(R / f"lm/{n}/history.csv")
        b = bench.get(n, {})
        rows.append(f"{lab} & {m['test_ppl']:.1f} & {pm(seed_stats(n))} & {m['params']['total'] / 1e6:.1f} & "
                    f"{m['params']['non_embedding'] / 1e6:.2f} & {m['train_time_s']:.0f} "
                    f"({h.epoch_time_s[1:].median():.0f}/ep) & "
                    f"{b.get('train_tokens_per_s', float('nan')) / 1e3:.0f} & "
                    f"{b.get('infer_tokens_per_s', float('nan')) / 1e3:.0f} & {g['ms_per_token_greedy']:.1f} & "
                    f"{g['metrics']['greedy']['repeated_trigram_rate']:.2f} & {g['metrics']['sample']['distinct2']:.2f} \\\\")
    (OUT / "part1_table.tex").write_text(
        "\\begin{tabular}{lrrrrrrrrrr}\n\\toprule\n"
        "Model & Test PPL & Test PPL & Params & Non-emb & Train s & Train & Infer & Gen & Rep-3 & Dist-2 \\\\\n"
        " & seed 42 & 3 seeds & (M) & (M) & (s/epoch) & k tok/s & k tok/s & ms/tok & greedy & top-$k$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Part II ----------------
    emb = J("embeddings.json")
    rows = []
    part2 = [("lstm", "Random $\\mathcal{N}(0,1)$, trainable"), ("lstm_w2v", "Own Skip-gram, frozen"),
             ("lstm_glove", "GloVe 6B, frozen"),
             ("lstm_smallinit", "\\textit{Suppl.:} random $\\mathcal{N}(0,0.1^2)$, train.")]
    for n, lab in [(n, l) for n, l in part2 if (R / f"lm/{n}/metrics.json").exists()]:
        m = J(f"lm/{n}/metrics.json")
        h = pd.read_csv(R / f"lm/{n}/history.csv")
        best = h.val_loss.min()
        ep_close = int(h[h.val_loss <= best + 0.05].epoch.min())
        cov = {"lstm": "--", "lstm_smallinit": "--", "lstm_w2v": f"{emb['word2vec']['type_coverage']:.1%}",
               "lstm_glove": f"{emb['glove']['type_coverage']:.1%}"}[n].replace("%", "\\%")
        rows.append(f"{lab} & {cov} & {m['params']['trainable'] / 1e6:.1f} & {m['test_ppl']:.1f} & "
                    f"{pm(seed_stats(n))} & {np.exp(h.val_loss[1]):.0f} & {np.exp(h.val_loss[3]):.0f} & "
                    f"{ep_close} & {h.val_loss.iloc[-1] - h.train_loss.iloc[-1]:.2f} & "
                    f"{h.step_loss_std[1:].mean():.3f} \\\\")
    (OUT / "part2_table.tex").write_text(
        "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
        "Input embedding & Cover. & Train. & Test PPL & Test PPL & Val PPL & Val PPL & Ep. to & Gap & Step-loss \\\\\n"
        " & & (M) & seed 42 & 3 seeds & ep 1 & ep 3 & best+.05 & ep 20 & std \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- Part III ----------------
    ds = J("downstream/summary.json")
    df = pd.DataFrame(ds["results"])
    labs = {"rnn": "RNN", "lstm": "LSTM", "transformer": "Transformer",
            "lstm_w2v": "LSTM + Skip-gram (frozen)", "lstm_glove": "LSTM + GloVe (frozen)",
            "lstm_smallinit": "\\textit{Suppl.:} LSTM, $\\mathcal{N}(0,0.1^2)$ init"}
    rows = []
    for n, lab in [(n, l) for n, l in labs.items() if (df.model == n).any()]:
        cells = []
        for pool in ["last", "mean", "emb"]:
            r = df[(df.model == n) & (df.pooling == pool)].iloc[0]
            cells.append(f"{r.test_accuracy:.3f} / {r.test_macro_f1:.3f}")
        ppl = df[df.model == n].lm_test_ppl.iloc[0]
        rows.append(f"{lab} & {ppl:.1f} & " + " & ".join(cells) + " \\\\")
    (OUT / "part3_table.tex").write_text(
        "\\begin{tabular}{lrccc}\n\\toprule\n"
        "Frozen LM & LM test PPL & Last state & Mean of states & Mean of input emb. \\\\\n"
        " & & acc / macro-F1 & acc / macro-F1 & acc / macro-F1 \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- generation excerpts ----------------
    def esc(s):
        s = " ".join(t if all(ord(c) < 0x250 for c in t) else "[non-Latin]" for t in s.split(" "))
        for a, b in [("\\", ""), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                     ("<eos>", "\\textlangle eos\\textrangle"), ('"', "''"), ("@-@", "@-@")]:
            s = s.replace(a, b)
        return s

    gj = J("generations.json")
    pi, n_tok = 1, 24
    rows = []
    for n, lab in [("ngram", "KN 4-gram"), ("rnn", "RNN"), ("lstm", "LSTM"), ("transformer", "Transformer"),
                   ("lstm_glove", "LSTM+GloVe")]:
        rows.append(f"{lab} & {esc(' '.join(gj['models'][n]['sample'][pi][0].split()[:n_tok]))} \\ldots \\\\")
    for n, lab in [("ngram", "KN 4-gram"), ("transformer", "Transformer")]:
        rows.append(f"{lab} (greedy) & {esc(' '.join(gj['models'][n]['greedy'][pi].split()[:n_tok]))} \\ldots \\\\")
    (OUT / "gen_table.tex").write_text(
        f"\\begin{{tabularx}}{{\\linewidth}}{{lX}}\n\\toprule\n\\multicolumn{{2}}{{l}}{{Prompt: "
        f"\\textit{{{esc(gj['prompts'][pi])}}} \\quad (top-$k$=40, $T$=0.8 unless noted; first sample, "
        f"first {n_tok} tokens)}} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")

    # ---------------- nearest neighbours (appendix) ----------------
    import torch
    from src.data import load_corpus
    from src.embeddings import neighbours

    vocab, _ = load_corpus()
    words = ["king", "war", "football", "france", "album", "good"]
    W_rand = torch.load(ROOT / "checkpoints/lstm.pt", map_location="cpu")["state_dict"]["emb.weight"]
    W_small = torch.load(ROOT / "checkpoints/lstm_smallinit.pt", map_location="cpu")["state_dict"]["emb.weight"]
    nb = {"LSTM, $\\mathcal{N}(0,1)$ init": neighbours(W_rand, vocab, words, k=4),
          "LSTM, $\\mathcal{N}(0,0.1^2)$ init": neighbours(W_small, vocab, words, k=4),
          "Skip-gram (own)": emb["neighbours"]["word2vec"], "GloVe 6B": emb["neighbours"]["glove"]}
    rows = [f"\\textit{{{w}}} & " + " & ".join(esc(", ".join(nb[k][w][:4])) for k in nb) + " \\\\"
            for w in words]
    (OUT / "neighbours_table.tex").write_text(
        "\\begin{tabularx}{\\linewidth}{lXXXX}\n\\toprule\nWord & " + " & ".join(nb) + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")

    # ---------------- embedding drift: cos(initial, final) of trainable tables ----------------
    from src.models import build_model
    from src.utils import load_config, set_seed

    cnt = np.bincount(load_corpus()[1]["train"].numpy(), minlength=len(vocab))
    top = np.argsort(-cnt)[:1000]
    drift = {}
    for n in ["rnn", "lstm", "transformer", "lstm_smallinit"]:
        if not (ROOT / f"checkpoints/{n}.pt").exists():
            continue
        cfg = load_config(f"configs/lm_{n}.yaml")
        set_seed(cfg["seed"])  # embedding init is the first random draw in train_lm.main
        W0 = build_model(cfg, len(vocab), vocab.pad_id).emb.weight.detach()
        W1 = torch.load(ROOT / f"checkpoints/{n}.pt", map_location="cpu")["state_dict"]["emb.weight"]
        cos = torch.nn.functional.cosine_similarity(W0, W1, dim=1).numpy()
        drift[n] = {"all": float(cos[3:].mean()), "top1000": float(cos[top].mean())}
    json.dump(drift, open(R / "embedding_drift.json", "w"), indent=2)

    # ---------------- OOV-bucket accuracy (appendix) + error counts ----------------
    pred = pd.read_csv(R / "downstream/test_predictions.csv")
    five = ["rnn", "lstm", "transformer", "lstm_w2v", "lstm_glove"]
    all_wrong = int(np.all([pred[f"{m}/mean"] != pred.label for m in five], axis=0).sum())
    glove_errors = int((pred["lstm_glove/mean"] != pred.label).sum())
    ob = ds["accuracy_by_oov_bucket"]
    buckets = list(ob["count"])
    rows = [f"\\texttt{{<unk>}} share & " + " & ".join(b.replace("%", "\\%") for b in buckets) + " \\\\",
            "Articles & " + " & ".join(str(ob["count"][b]) for b in buckets) + " \\\\ \\midrule"]
    for m in ["lstm", "transformer", "lstm_glove"]:
        rows.append(f"{labs[m]} (mean) & " + " & ".join(f"{ob[m + '/mean'][b]:.3f}" for b in buckets) + " \\\\")
    (OUT / "oov_table.tex").write_text("\\begin{tabular}{l" + "r" * len(buckets) + "}\n\\toprule\n"
                                       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- inline numbers ----------------
    d = ds["data"]
    err = J("downstream/error_stats.json")
    w2v = emb["word2vec"]
    macros += [
        ("agOOV", f"{d['agnews_test']['token_oov_rate']:.1%}"),
        ("sstOOV", f"{d['sst2_validation']['token_oov_rate']:.1%}"),
        ("gloveCov", f"{emb['glove']['type_coverage']:.1%}"),
        ("gloveTokCov", f"{emb['glove']['token_coverage_train']:.1%}"),
        ("gloveMissing", f"{emb['glove']['missing_types']:,}"),
        ("wvTime", f"{w2v['train_time_s']:.0f}"),
        ("rhoMean", f"{ds['ppl_vs_accuracy']['mean']['spearman_ppl_vs_acc']:.1f}"),
        ("rhoMeanP", f"{ds['ppl_vs_accuracy']['mean']['p_value']:.2f}"),
        ("rhoLast", f"{ds['ppl_vs_accuracy']['last']['spearman_ppl_vs_acc']:.1f}"),
        ("mcLstmRnnP", f"{err['mcnemar']['lstm/mean vs rnn/mean']['p_value']:.2f}"),
        ("mcTrEmbP", f"{err['mcnemar']['transformer/emb vs transformer/mean']['p_value']:.2f}"),
        ("bizSci", str(err["top_confusions"]["lstm_glove/mean"]["Business -> Sci/Tech"]
                       + err["top_confusions"]["lstm_glove/mean"]["Sci/Tech -> Business"])),
        ("allWrong", str(all_wrong)),
        ("gloveErrors", f"{glove_errors:,}"),
    ]
    from scipy.stats import ttest_ind
    t_res = ttest_ind(seed_stats("transformer"), seed_stats("lstm"), equal_var=False)
    macros += [("pTrLstm", f"{t_res.pvalue:.3f}"),
               ("meanrnn", f"{np.mean(seed_stats('rnn')):.1f}"),
               ("meanlstm", f"{np.mean(seed_stats('lstm')):.1f}"),
               ("meantransformer", f"{np.mean(seed_stats('transformer')):.1f}")]
    for n, d in drift.items():
        key = n.replace("_", "").replace("smallinit", "small")
        macros += [(f"drift{key}", f"{d['all']:.2f}"), (f"drifttop{key}", f"{d['top1000']:.2f}")]
    for n in ["rnn", "lstm", "transformer", "lstm_w2v", "lstm_glove"]:
        v = seed_stats(n)
        macros.append((f"sd{n.replace('_', '').replace('2', 'two')}",
                       f"{np.std(v, ddof=1):.1f}" if len(v) > 1 else "(pending)"))
    tex = "\n".join(f"\\newcommand{{\\{k}}}{{{v.replace('%', chr(92) + '%')}}}" for k, v in macros)
    (OUT / "numbers.tex").write_text(tex + "\n")
    print((OUT / "part1_table.tex").read_text(), (OUT / "part2_table.tex").read_text(),
          (OUT / "part3_table.tex").read_text(), tex, sep="\n")


if __name__ == "__main__":
    main()
