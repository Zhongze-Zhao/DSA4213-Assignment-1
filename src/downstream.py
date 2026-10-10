"""Part III: frozen LM representations + logistic regression on AG News topic classification.

    python -m src.downstream --config configs/downstream.yaml

For every trained LM and pooling choice, each text becomes one fixed-length vector:
  last  : top-layer hidden state at the last real (non-pad) token
  mean  : average of top-layer hidden states over real tokens (padding excluded)
  emb   : average of the LM's INPUT embeddings over real tokens (no context; reference point)
The LMs are frozen; only a StandardScaler + LogisticRegression is trained, with C chosen on a
validation split carved from the training subset. Same classifier and procedure for all.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_corpus, wiki_tokenize
from src.generate import load_neural
from src.utils import ROOT, get_device, limit_gpu_memory, load_config, save_json, set_seed

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def load_agnews(cfg):
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], cache_dir=str(ROOT / "data/hf_cache"))
    tr = ds["train"].to_pandas()
    te = ds["test"].to_pandas()
    # stratified subset of the official train split; official test split kept whole
    tr, _ = train_test_split(tr, train_size=cfg["train_size"], stratify=tr["label"],
                             random_state=cfg["seed"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def oov_stats(token_lists, vocab):
    toks = [t for ts in token_lists for t in ts]
    types = set(toks)
    return {"tokens": len(toks),
            "token_oov_rate": sum(t not in vocab.stoi for t in toks) / len(toks),
            "type_oov_rate": sum(t not in vocab.stoi for t in types) / len(types),
            "mean_length": len(toks) / len(token_lists)}


@torch.inference_mode()
def extract(model, id_lists, max_len, pad_id, device, bs=64):
    """Return dict pooling -> (N, H) features for one frozen model."""
    feats = {"last": [], "mean": [], "emb": []}
    for i in range(0, len(id_lists), bs):
        chunk = [ids[:max_len] for ids in id_lists[i:i + bs]]
        lens = torch.tensor([len(c) for c in chunk], device=device)
        x = torch.full((len(chunk), max(lens).item()), pad_id, device=device)
        for j, c in enumerate(chunk):
            x[j, : len(c)] = torch.tensor(c, device=device)  # right padding
        h = model.encode(x)  # no (B, T, |V|) logits: features only
        mask = (x != pad_id).unsqueeze(-1).float()
        feats["last"].append(h[torch.arange(len(chunk), device=device), lens - 1])
        feats["mean"].append((h * mask).sum(1) / mask.sum(1))
        e = model.emb(x)
        feats["emb"].append((e * mask).sum(1) / mask.sum(1))
    return {k: torch.cat(v).float().cpu().numpy() for k, v in feats.items()}


def fit_eval(Xtr, ytr, Xte, yte, cfg):
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=cfg["val_frac"], stratify=ytr,
                                      random_state=cfg["seed"])
    best = None
    for C in cfg["C_grid"]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=cfg["max_iter"]))
        clf.fit(Xa, ya)
        f1 = f1_score(yv, clf.predict(Xv), average="macro")
        if best is None or f1 > best[1]:
            best = (C, f1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=best[0], max_iter=cfg["max_iter"]))
    clf.fit(Xtr, ytr)  # refit on the full training subset with the selected C
    pred = clf.predict(Xte)
    return pred, {"C": best[0], "val_macro_f1": best[1],
                  "test_accuracy": accuracy_score(yte, pred),
                  "test_macro_f1": f1_score(yte, pred, average="macro"),
                  "confusion": confusion_matrix(yte, pred).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/downstream.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    limit_gpu_memory()
    vocab, _ = load_corpus(cfg["data_config"])

    tr, te = load_agnews(cfg)
    tok_tr = [wiki_tokenize(t) for t in tr["text"]]
    tok_te = [wiki_tokenize(t) for t in te["text"]]
    ids_tr = [vocab.encode(t) for t in tok_tr]
    ids_te = [vocab.encode(t) for t in tok_te]
    ytr, yte = tr["label"].to_numpy(), te["label"].to_numpy()
    stats = {"agnews_train_subset": oov_stats(tok_tr, vocab), "agnews_test": oov_stats(tok_te, vocab),
             "train_size": len(tr), "test_size": len(te),
             "train_label_counts": np.bincount(ytr).tolist()}
    if cfg.get("compare_sst2"):  # justification for the dataset choice
        from datasets import load_dataset
        sst = load_dataset("stanfordnlp/sst2", cache_dir=str(ROOT / "data/hf_cache"))["validation"]
        stats["sst2_validation"] = oov_stats([wiki_tokenize(s) for s in sst["sentence"]], vocab)
    print(stats)

    lm_metrics = {}
    for name in cfg["models"]:
        import json
        lm_metrics[name] = json.load(open(ROOT / f"results/lm/{name}/metrics.json"))["test_ppl"]

    rows, preds = [], {}
    for name in cfg["models"]:
        model, _ = load_neural(name, len(vocab), vocab.pad_id, device)
        t0 = time.perf_counter()
        Ftr = extract(model, ids_tr, cfg["max_len"], vocab.pad_id, device)
        Fte = extract(model, ids_te, cfg["max_len"], vocab.pad_id, device)
        ext_s = time.perf_counter() - t0
        for pool in cfg["poolings"]:
            pred, m = fit_eval(Ftr[pool], ytr, Fte[pool], yte, cfg)
            preds[f"{name}/{pool}"] = pred
            rows.append({"model": name, "pooling": pool, "dim": Ftr[pool].shape[1],
                         "lm_test_ppl": lm_metrics[name], "extract_time_s": ext_s, **m})
            print(f"{name:12s} {pool:5s} acc {m['test_accuracy']:.4f} "
                  f"macroF1 {m['test_macro_f1']:.4f} (C={m['C']})")
        del model
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    (ROOT / "results/downstream").mkdir(parents=True, exist_ok=True)
    df.drop(columns=["confusion"]).to_csv(ROOT / "results/downstream/results.csv", index=False)

    # error analysis: per-example predictions + accuracy by OOV rate / length buckets
    oov_rate = np.array([np.mean([t == vocab.unk_id for t in ids]) for ids in ids_te])
    length = np.array([len(ids) for ids in ids_te])
    per_ex = pd.DataFrame({"text": te["text"], "label": [LABELS[y] for y in yte],
                           "oov_rate": oov_rate, "length": length})
    for k, p in preds.items():
        per_ex[k] = [LABELS[y] for y in p]
    per_ex.to_csv(ROOT / "results/downstream/test_predictions.csv", index=False)
    buckets = pd.cut(oov_rate, [-0.001, 0.0, 0.05, 0.1, 1.0], labels=["0", "(0,5%]", "(5,10%]", ">10%"])
    by_oov = {k: pd.Series(p == yte).groupby(buckets, observed=False).mean().round(4).to_dict()
              for k, p in preds.items()}
    by_oov["count"] = pd.Series(buckets).value_counts().sort_index().to_dict()
    from scipy.stats import spearmanr
    rank = {}
    for pool in cfg["poolings"]:
        sub = df[df.pooling == pool]
        rho, pval = spearmanr(sub["lm_test_ppl"], sub["test_accuracy"])
        rank[pool] = {"spearman_ppl_vs_acc": float(rho), "p_value": float(pval), "n_models": len(sub)}
    save_json({"data": stats, "results": rows, "accuracy_by_oov_bucket": by_oov,
               "ppl_vs_accuracy": rank, "labels": LABELS}, "results/downstream/summary.json")


if __name__ == "__main__":
    main()
