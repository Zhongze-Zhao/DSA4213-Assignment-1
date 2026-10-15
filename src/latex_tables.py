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


def pfmt(p):
    """p-value for use inside math mode."""
    if p >= 0.01:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.0e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def pm(vals):
    return f"{np.mean(vals):.1f} $\\pm$ {np.std(vals, ddof=1):.1f}" if len(vals) > 1 else "--"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bench = J("benchmark.json")["models"] if (R / "benchmark.json").exists() else {}
    gen = J("generations.json")["models"]
    macros = []

    # ---------------- Part I ----------------
    # (a) prediction quality and fit; train CE re-scored in eval mode at the selected checkpoint
    fit = J("lm/clean_fit.json")
    ng = J("lm/ngram/metrics.json")
    names = [("ngram", f"KN {ng['order']}-gram"), ("rnn", "RNN"), ("lstm", "LSTM"), ("transformer", "Transformer")]
    rows = []
    for n, lab in names:
        m, f = (ng if n == "ngram" else J(f"lm/{n}/metrics.json")), fit[n]
        seeds = "(deterministic)" if n == "ngram" else pm(seed_stats(n))
        rows.append(f"{lab} & {m['test_loss']:.3f} & {m['test_ppl']:.1f} & {seeds} & {f['train_ce']:.2f} & "
                    f"{f['val_ce']:.2f} & {f['gap_val_minus_train']:.2f} \\\\")
    qual = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            "Model & Test CE & Test PPL & Test PPL & Train CE & Val CE & Gap \\\\\n"
            " & (nats) & seed 42 & 3 seeds & (eval mode) & & val$-$train \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    # (b) cost and generation. Training time: lockstep re-run (src/lockstep_train.py) if available, so that
    # all three models share the same thermal history; generation: src/gen_benchmark.py (batch 1, 120 tokens)
    lock = J("lockstep_timing.json")["models"] if (R / "lockstep_timing.json").exists() else None
    gb = J("gen_benchmark.json")["ms_per_token"] if (R / "gen_benchmark.json").exists() else None
    rows = []
    g = gen["ngram"]
    ng_gen = f"{gb['ngram']['120']['median']:.2f} (CPU)" if gb else f"{g['ms_per_token_greedy']:.1f}"
    rows.append(f"KN {ng['order']}-gram & {ng['stored_counts'] / 1e6:.1f}$^\\dagger$ & -- & {ng['train_time_s']:.0f} (CPU) & -- & "
                f"{ng['inference_tokens_per_s'] / 1e3:.0f} (CPU) & -- & {ng_gen} & "
                f"{g['metrics']['greedy']['repeated_trigram_rate']:.2f} & {g['metrics']['sample']['distinct2']:.2f} \\\\")
    for n, lab in names[1:]:
        m, g = J(f"lm/{n}/metrics.json"), gen[n]
        b = bench.get(n, {})
        if lock:
            tt = f"{lock[n]['train_time_s']:.0f} ({lock[n]['median_epoch_s']:.0f}/ep)"
        else:
            h = pd.read_csv(R / f"lm/{n}/history.csv")
            tt = f"{m['train_time_s']:.0f} ({h.epoch_time_s[1:].median():.0f}/ep)"
        gen_cells = (f"{gb[n]['recompute']['120']['median']:.2f} & {gb[n]['incremental']['120']['median']:.2f}"
                     if gb else f"{g['ms_per_token_greedy']:.1f} & --")
        rows.append(f"{lab} & {m['params']['total'] / 1e6:.1f} & {m['params']['non_embedding'] / 1e6:.2f} & {tt} & "
                    f"{b.get('train_tokens_per_s', float('nan')) / 1e3:.0f} & "
                    f"{b.get('infer_tokens_per_s', float('nan')) / 1e3:.0f} & {gen_cells} & "
                    f"{g['metrics']['greedy']['repeated_trigram_rate']:.2f} & {g['metrics']['sample']['distinct2']:.2f} \\\\")
    cost = ("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
            "Model & Params & Non-emb & Train time & Train & Infer & \\multicolumn{2}{c}{Gen ms/token} & Rep-3 & Dist-2 \\\\\n"
            " & (M) & (M) & s (s/epoch) & k tok/s & k tok/s & recomp. & increm. & greedy & top-$k$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    short = {"rnn": "Rnn", "lstm": "Lstm", "transformer": "Tr"}
    for n, k in short.items():
        macros += [(f"bTrain{k}", f"{bench[n]['train_tokens_per_s'] / 1e3:.0f}"),
                   (f"bInfer{k}", f"{bench[n]['infer_tokens_per_s'] / 1e3:.0f}")]
        if lock:
            macros += [(f"lockTime{k}", f"{lock[n]['train_time_s']:.0f}"),
                       (f"lockEp{k}", f"{lock[n]['median_epoch_s']:.1f}"),
                       (f"lockVal{k}", f"{lock[n]['best_val_loss']:.2f}")]
        if gb:
            for mode, mk in (("recompute", "Rec"), ("incremental", "Inc")):
                for L, lk in (("40", "Short"), ("120", "Long")):
                    macros.append((f"gen{k}{mk}{lk}", f"{gb[n][mode][L]['median']:.2f}"))
    if gb:
        macros.append(("genNgram", f"{gb['ngram']['120']['median']:.2f}"))
    macros.append(("ngFitTime", f"{ng['train_time_s']:.0f}"))
    (OUT / "part1_table.tex").write_text(qual + "\n")
    (OUT / "part1_cost_table.tex").write_text(cost + "\n")
    macros += [(f"gap{n.replace('transformer', 'tr')}", f"{fit[n]['gap_val_minus_train']:.2f}")
               for n in ["ngram", "rnn", "lstm", "transformer"]]
    macros += [(f"trainppl{n.replace('transformer', 'tr')}", f"{fit[n]['train_ppl']:.1f}")
               for n in ["ngram", "rnn", "lstm", "transformer"]]

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
                    f"{ep_close} & {fit[n]['gap_val_minus_train']:.2f} & "
                    f"{h.step_loss_std[1:].mean():.3f} \\\\")
    (OUT / "part2_table.tex").write_text(
        "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
        "Input embedding & Cover. & Train. & Test PPL & Test PPL & Val PPL & Val PPL & Ep. to & Gap & Step-loss \\\\\n"
        " & & (M) & seed 42 & 3 seeds & ep 1 & ep 3 & best+.05 & val$-$train & std \\\\\n\\midrule\n"
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

    # ---------------- Part III over LM seeds + controls (src/downstream_seeds.py) ----------------
    FAM = {"rnn": "Rnn", "lstm": "Lstm", "transformer": "Tr", "lstm_w2v": "Sg", "lstm_glove": "Glove",
           "lstm_smallinit": "Small", "untrained_lstm": "UnLstm", "untrained_transformer": "UnTr"}
    if (R / "downstream/seeds.csv").exists():
        sd = pd.read_csv(R / "downstream/seeds.csv")
        ss = J("downstream/seeds_summary.json")
        agg = sd.groupby(["family", "pooling"]).agg(acc=("test_accuracy", "mean"), sd=("test_accuracy", "std"),
                                                     f1=("test_macro_f1", "mean"), n=("seed", "count"),
                                                     ppl=("lm_test_ppl", "mean"))

        def cell(fam, pool):
            a = agg.loc[(fam, pool)]
            if a.n > 1 and a.sd == 0:  # frozen input table: identical features for every LM seed
                return f"{a.acc:.3f}$^\\ddagger$"
            return f"{a.acc:.3f}" + (f"\\,{{\\scriptsize$\\pm${a.sd:.3f}}}" if a.n > 1 else "")

        def row(fam, lab, seeds_lab):
            ppl = agg.loc[(fam, "mean")].ppl
            ppl_s = f"{ppl:,.0f}" if ppl > 1000 else f"{ppl:.1f}"
            return f"{lab} & {seeds_lab} & {ppl_s} & " + " & ".join(cell(fam, p) for p in ["last", "mean", "emb"]) + " \\\\"

        rows = [row(f, l, "3") for f, l in [("rnn", "RNN"), ("lstm", "LSTM"), ("transformer", "Transformer"),
                                             ("lstm_w2v", "LSTM + Skip-gram (frozen)"),
                                             ("lstm_glove", "LSTM + GloVe (frozen)")]]
        rows += ["\\midrule", row("lstm_smallinit", "\\textit{Suppl.:} LSTM, $\\mathcal{N}(0,0.1^2)$ init", "1"),
                 row("untrained_lstm", "\\textit{Control:} untrained LSTM", "--"),
                 row("untrained_transformer", "\\textit{Control:} untrained Transformer", "--"),
                 "\\midrule",
                 f"\\textit{{Reference:}} TF-IDF bag of words, no LM & -- & -- & "
                 f"\\multicolumn{{3}}{{c}}{{{ss['bow_tfidf']['test_accuracy']:.3f}}} \\\\"]
        (OUT / "part3_table.tex").write_text(
            "\\begin{tabular}{lcrccc}\n\\toprule\n"
            "Frozen LM & Seeds & LM test & Last state & Mean of states & Mean of input emb. \\\\\n"
            " & & PPL & \\multicolumn{3}{c}{test accuracy (mean $\\pm$ std over LM seeds)} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        for (fam, pool), a in agg.iterrows():
            macros.append((f"acc{FAM[fam]}{pool.capitalize()}", f"{a.acc:.3f}"))
            if a.n > 1:
                macros.append((f"sd{FAM[fam]}{pool.capitalize()}", f"{a.sd:.3f}"))
        for fam in ["untrained_lstm", "untrained_transformer"]:
            macros.append((f"ppl{FAM[fam]}", f"{agg.loc[(fam, 'mean')].ppl:,.0f}"))
        for pool in ["last", "mean", "emb"]:
            r = ss["spearman_all_trained"][pool]
            macros += [(f"rhoAll{pool.capitalize()}", f"{r['rho']:.2f}"),
                       (f"rhoAll{pool.capitalize()}P", pfmt(r["p_value"]))]
            r1 = ss["spearman_part1_only"][pool]
            macros += [(f"rhoPI{pool.capitalize()}", f"{r1['rho']:.2f}"),
                       (f"rhoPI{pool.capitalize()}P", pfmt(r1["p_value"]))]
        macros += [("nAll", str(ss["spearman_all_trained"]["mean"]["n"])),
                   ("bowAcc", f"{ss['bow_tfidf']['test_accuracy']:.3f}"),
                   ("maxFdiff", f"{(sd.test_accuracy - sd.test_macro_f1).abs().max():.3f}")]

    # ---------------- blind human rating of generations (src/human_eval.py) ----------------
    if (R / "human_eval.json").exists():
        he = J("human_eval.json")
        rows = []
        for n, lab in [("ngram", "KN 4-gram"), ("rnn", "RNN"), ("lstm", "LSTM"), ("transformer", "Transformer"),
                       ("lstm_w2v", "LSTM + Skip-gram"), ("lstm_glove", "LSTM + GloVe")]:
            sc = he["scores"][n]
            ppl = J(f"lm/{n}/metrics.json")["test_ppl"]
            cells = [f"{sc[d]['mean']:.2f} [{sc[d]['ci95'][0]:.1f}, {sc[d]['ci95'][1]:.1f}]"
                     for d in ("fluency", "consistency")]
            rows.append(f"{lab} & {ppl:.1f} & " + " & ".join(cells) + " \\\\")
        (OUT / "human_table.tex").write_text(
            "\\begin{tabular}{lrcc}\n\\toprule\nModel & Test PPL & Fluency & Consistency \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        p = he["mann_whitney_p"]
        macros += [("heTrNgCon", f"{p['transformer vs ngram / consistency']:.3f}"),
                   ("heTrLstmCon", f"{p['transformer vs lstm / consistency']:.2f}"),
                   ("heSgGloveCon", f"{p['lstm_w2v vs lstm_glove / consistency']:.2f}"),
                   ("heItems", str(he["items_per_model"]))]

    # ---------------- generation excerpts ----------------
    def esc(s):
        s = " ".join(t if all(ord(c) < 0x250 for c in t) else "[non-Latin]" for t in s.split(" "))
        for a, b in [("\\", ""), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                     ("<eos>", "\\textlangle eos\\textrangle{}"), ('"', "''"), ("@-@", "@-@")]:
            s = s.replace(a, b)
        return s

    gj = J("generations.json")
    pi, n_tok = 1, 24
    rows = []
    for n, lab in [("ngram", "KN 4-gram"), ("rnn", "RNN"), ("lstm", "LSTM"), ("transformer", "Transformer"),
                   ("lstm_w2v", "LSTM+Skip-gram"), ("lstm_glove", "LSTM+GloVe")]:
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
    # seed-1/2 runs alternated RNN and Transformer, so their epoch times are not biased by run order
    def seed_ep(n):
        return np.mean([pd.read_csv(R / f"lm/{n}_seed{s}/history.csv").epoch_time_s[1:].median() for s in (1, 2)])
    macros += [("transSeedEp", f"{seed_ep('transformer'):.1f}"), ("rnnSeedEp", f"{seed_ep('rnn'):.1f}")]
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
