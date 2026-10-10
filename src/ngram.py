"""Interpolated Kneser-Ney n-gram language model (own implementation).

    python -m src.ngram --config configs/ngram.yaml [--quick]

The order n is selected on the validation set; the test set is scored with the same windows as
the neural models (each target only sees preceding tokens inside its 128-token window).
"""
import argparse
import math
import pickle
import time
from collections import Counter, defaultdict

import numpy as np

from src.data import load_corpus, make_windows
from src.utils import ROOT, hardware_info, load_config, save_json


class KneserNeyLM:
    """Interpolated Kneser-Ney with one absolute discount D_k = n1/(n1+2*n2) per order.

    Highest order used for a query: raw counts c(h,w).
    Lower orders: continuation counts N1+(. h w) (number of distinct left extensions).
    Base case: continuation unigram interpolated with a uniform distribution over the vocab.
    """

    def __init__(self, n, vocab_size):
        self.n, self.V = n, vocab_size

    def fit(self, tokens):
        n = self.n
        raw = {k: Counter(zip(*[tokens[i:] for i in range(k)])) for k in range(1, n + 1)}
        # continuation counts for orders 1..n-1: how many distinct words precede the k-gram
        cont = {k: Counter(g[1:] for g in raw[k + 1]) for k in range(1, n)}
        self.D = {}
        for k in range(1, n + 1):
            src = raw[k] if k == n else cont[k]
            cc = Counter(src.values())
            self.D[k] = cc[1] / (cc[1] + 2 * cc[2])
        # tables[kind][k][h] = (dict w->count, total, n_types) for k>=2
        self.tables = {"raw": {}, "cont": {}}
        for kind, src_all in (("raw", raw), ("cont", cont)):
            for k in range(2, n + 1):
                if k not in src_all:
                    continue
                t = defaultdict(dict)
                for g, c in src_all[k].items():
                    t[g[:-1]][g[-1]] = c
                self.tables[kind][k] = {h: (d, sum(d.values()), len(d)) for h, d in t.items()}
        # base: continuation unigram mixed with uniform
        uc = np.zeros(self.V)
        for (w,), c in cont[1].items() if n > 1 else raw[1].items():
            uc[w] = c
        tot, D1 = uc.sum(), self.D[1]
        self.p_uni = np.maximum(uc - D1, 0) / tot + D1 * (uc > 0).sum() / tot / self.V
        return self

    def _p(self, w, h, top):
        """P(w | h) with h a tuple of length k-1; `top` means h is the longest context used."""
        if not h:
            return self.p_uni[w]
        k = len(h) + 1
        lower = self._p(w, h[1:], False)
        entry = self.tables["raw" if top else "cont"][k].get(h)
        if entry is None:  # unseen context: all mass goes to the lower order
            return lower
        d, tot, types = entry
        D = self.D[k]
        return max(d.get(w, 0) - D, 0) / tot + D * types / tot * lower

    def prob(self, w, ctx):
        h = tuple(ctx[-(self.n - 1):]) if self.n > 1 else ()
        return self._p(w, h, True)

    def dist(self, ctx):
        """Full next-token distribution (numpy vector over the vocab), used for generation."""
        h_full = tuple(ctx[-(self.n - 1):]) if self.n > 1 else ()
        p = self.p_uni.copy()
        for k in range(2, len(h_full) + 2):
            h = h_full[len(h_full) - (k - 1):]
            entry = self.tables["raw" if k == len(h_full) + 1 else "cont"][k].get(h)
            if entry is None:
                continue
            d, tot, types = entry
            D = self.D[k]
            new = D * types / tot * p
            ws = np.fromiter(d.keys(), dtype=np.int64)
            cs = np.fromiter(d.values(), dtype=np.float64)
            new[ws] += np.maximum(cs - D, 0) / tot
            p = new
        return p


def score_windows(lm, windows, pad_id):
    """Mean NLL over all non-pad targets; context restricted to the target's own window."""
    nll, cnt = 0.0, 0
    for row in windows.tolist():
        for t in range(1, len(row)):
            y = row[t]
            if y == pad_id:
                break
            nll -= math.log(lm.prob(y, row[max(0, t - lm.n + 1):t]))
            cnt += 1
    return nll / cnt, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ngram.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    vocab, ids = load_corpus(cfg["data_config"])
    train = ids["train"].tolist()
    win = {s: make_windows(ids[s], cfg["seq_len"], vocab.pad_id) for s in ("validation", "test")}
    if args.quick:
        train, win = train[:200_000], {s: w[:50] for s, w in win.items()}

    # select n on validation
    sel = {}
    for n in cfg["orders"]:
        t0 = time.perf_counter()
        lm = KneserNeyLM(n, len(vocab)).fit(train)
        fit_s = time.perf_counter() - t0
        val, _ = score_windows(lm, win["validation"], vocab.pad_id)
        sel[n] = {"val_loss": val, "val_ppl": math.exp(val), "fit_time_s": fit_s, "D": lm.D}
        print(f"n={n}: val ppl {math.exp(val):.1f} (fit {fit_s:.1f}s) D={lm.D}")
    best_n = min(sel, key=lambda n: sel[n]["val_loss"])

    t0 = time.perf_counter()
    lm = KneserNeyLM(best_n, len(vocab)).fit(train)
    fit_s = time.perf_counter() - t0
    # sanity check: distribution sums to 1 and matches the scalar scorer
    ctx = train[100:100 + best_n - 1]
    p = lm.dist(ctx)
    assert abs(p.sum() - 1) < 1e-6, p.sum()
    assert abs(p[train[100 + best_n - 1]] - lm.prob(train[100 + best_n - 1], ctx)) < 1e-12
    t0 = time.perf_counter()
    test, n_targets = score_windows(lm, win["test"], vocab.pad_id)
    inf_s = time.perf_counter() - t0
    # "model size" = number of stored (context, word) count entries + unigram vector
    n_params = sum(len(d) for kind in lm.tables.values() for table in kind.values()
                   for d, _, _ in table.values()) + len(vocab)
    metrics = {
        "name": f"kn{best_n}gram", "order": best_n, "selection": sel,
        "val_loss": sel[best_n]["val_loss"], "val_ppl": sel[best_n]["val_ppl"],
        "test_loss": test, "test_ppl": math.exp(test), "test_targets": n_targets,
        "stored_counts": n_params, "train_time_s": fit_s,
        "inference_test_s": inf_s, "inference_tokens_per_s": n_targets / inf_s,
        "dist_sum_check": float(p.sum()), "hardware": hardware_info() | {"device": "cpu (1 thread)"},
    }
    print(f"[kn{best_n}] test ppl {metrics['test_ppl']:.1f} | fit {fit_s:.1f}s | "
          f"score {n_targets / inf_s:,.0f} tok/s")
    out = "results/quick/ngram" if args.quick else "results/lm/ngram"
    save_json(metrics, f"{out}/metrics.json")
    if not args.quick:
        (ROOT / "checkpoints").mkdir(exist_ok=True)
        import src.ngram  # pickle under the importable name, not __main__
        lm.__class__ = src.ngram.KneserNeyLM
        with open(ROOT / "checkpoints/ngram.pkl", "wb") as f:
            pickle.dump(lm, f)


if __name__ == "__main__":
    main()
