"""Part III robustness: repeat the frozen-feature probe for every LM seed, plus two controls.

    python -m src.downstream_seeds --config configs/downstream.yaml

Same AG News subset, tokenisation, poolings and classifier as src/downstream.py, applied to
  * all trained checkpoints: rnn / lstm / transformer / lstm_w2v / lstm_glove x seeds 42, 1, 2
    (+ the supplementary lstm_smallinit), so accuracy gets a mean +- std over LM seeds and the
    PPL-vs-accuracy correlation uses every checkpoint;
  * untrained controls: the LSTM and Transformer architectures with the exact seed-42 initial
    weights but NO language-model training (does next-token training help the features at all?);
  * a bag-of-words reference: TF-IDF over the same WikiText-2 vocabulary + logistic regression
    with the same C selection (how far are LM features from a purely lexical classifier?).
Writes results/downstream/seeds.csv and results/downstream/seeds_summary.json.
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.data import load_corpus, make_windows, wiki_tokenize
from src.downstream import extract, fit_eval, load_agnews
from src.generate import load_neural
from src.models import build_model
from src.train_lm import run_epoch
from src.utils import ROOT, get_device, limit_gpu_memory, load_config, save_json, set_seed

FAMILIES = ["rnn", "lstm", "transformer", "lstm_w2v", "lstm_glove"]
SEEDS = [42, 1, 2]


def ckpt_name(family, seed):
    return family if seed == 42 else f"{family}_seed{seed}"


def untrained(family, vocab, device):
    """Seed-42 initial weights of `family`, exactly as train_lm.main draws them, untrained."""
    cfg = load_config(f"configs/lm_{family}.yaml")
    set_seed(cfg["seed"])
    return build_model(cfg, len(vocab), vocab.pad_id).to(device).eval(), cfg


def bow_baseline(ids_tr, ytr, ids_te, yte, V, cfg):
    def counts(id_lists):
        rows = np.repeat(np.arange(len(id_lists)), [len(x) for x in id_lists])
        cols = np.concatenate([np.asarray(x) for x in id_lists])
        return csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(id_lists), V))
    tf = TfidfTransformer(sublinear_tf=True).fit(counts(ids_tr))
    Xtr, Xte = tf.transform(counts(ids_tr)), tf.transform(counts(ids_te))
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=cfg["val_frac"], stratify=ytr,
                                      random_state=cfg["seed"])
    # sparse TF-IDF is already scaled, so a wider C grid than for the standardised LM features
    grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    best = max(grid, key=lambda C: f1_score(
        yv, LogisticRegression(C=C, max_iter=cfg["max_iter"]).fit(Xa, ya).predict(Xv), average="macro"))
    pred = LogisticRegression(C=best, max_iter=cfg["max_iter"]).fit(Xtr, ytr).predict(Xte)
    return {"C": best, "test_accuracy": accuracy_score(yte, pred),
            "test_macro_f1": f1_score(yte, pred, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/downstream.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    limit_gpu_memory()
    vocab, ids = load_corpus(cfg["data_config"])

    tr, te = load_agnews(cfg)
    ids_tr = [vocab.encode(wiki_tokenize(t)) for t in tr["text"]]
    ids_te = [vocab.encode(wiki_tokenize(t)) for t in te["text"]]
    ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()

    jobs = [(f, s, ckpt_name(f, s)) for f in FAMILIES for s in SEEDS] + [("lstm_smallinit", 42, "lstm_smallinit")]
    jobs += [("untrained_lstm", 42, None), ("untrained_transformer", 42, None)]
    test_win = make_windows(ids["test"], 128, vocab.pad_id)

    rows = []
    for family, seed, name in jobs:
        if name is None:
            model, lm_cfg = untrained(family.removeprefix("untrained_"), vocab, device)
            ppl = math.exp(run_epoch(model, test_win, lm_cfg, device, vocab.pad_id, bs=32))
        else:
            model, _ = load_neural(name, len(vocab), vocab.pad_id, device)
            ppl = json.load(open(ROOT / f"results/lm/{name}/metrics.json"))["test_ppl"]
        Ftr = extract(model, ids_tr, cfg["max_len"], vocab.pad_id, device)
        Fte = extract(model, ids_te, cfg["max_len"], vocab.pad_id, device)
        for pool in cfg["poolings"]:
            _, m = fit_eval(Ftr[pool], ytr, Fte[pool], yte, cfg)
            rows.append({"family": family, "seed": seed, "pooling": pool, "lm_test_ppl": ppl,
                         **{k: v for k, v in m.items() if k != "confusion"}})
            print(f"{family:22s} seed {seed:2d} {pool:5s} ppl {ppl:9.1f} acc {m['test_accuracy']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/downstream/seeds.csv", index=False)

    summary = {"by_family": {}, "spearman_all_trained": {}, "bow_tfidf": None}
    for (fam, pool), g in df.groupby(["family", "pooling"]):
        summary["by_family"].setdefault(fam, {})[pool] = {
            "n_seeds": len(g), "acc_mean": g.test_accuracy.mean(),
            "acc_std": g.test_accuracy.std(ddof=1) if len(g) > 1 else None,
            "f1_mean": g.test_macro_f1.mean(), "ppl_mean": g.lm_test_ppl.mean()}
    trained = df[~df.family.str.startswith("untrained")]
    for pool, g in trained.groupby("pooling"):
        rho, p = spearmanr(g.lm_test_ppl, g.test_accuracy)
        summary["spearman_all_trained"][pool] = {"rho": float(rho), "p_value": float(p), "n": len(g)}
        # within the Part I families only (same random-embedding setting)
        g1 = g[g.family.isin(["rnn", "lstm", "transformer"])]
        rho1, p1 = spearmanr(g1.lm_test_ppl, g1.test_accuracy)
        summary.setdefault("spearman_part1_only", {})[pool] = {"rho": float(rho1), "p_value": float(p1),
                                                               "n": len(g1)}
    summary["bow_tfidf"] = bow_baseline(ids_tr, ytr, ids_te, yte, len(vocab), cfg)
    print(json.dumps(summary, indent=1))
    save_json(summary, "results/downstream/seeds_summary.json")


if __name__ == "__main__":
    main()