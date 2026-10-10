"""Part II: build the two fixed embedding tables aligned with the shared vocabulary.

    python -m src.embeddings --config configs/embeddings.yaml

1. Own embeddings: gensim Skip-gram trained on the WikiText-2 TRAIN split only (timed separately).
2. Public embeddings: GloVe 6B 300d (Wikipedia 2014 + Gigaword 5, 6B tokens, uncased).
Missing words get N(0, sigma^2) vectors (sigma = std of the pretrained table, fixed seed);
<pad> is the zero vector. Saves data/emb/{w2v,glove}_300.pt and results/embeddings.json.
"""
import argparse
import os
import time

import numpy as np
import torch

from src.data import EOS, PAD, load_corpus
from src.utils import ROOT, load_config, save_json, set_seed

os.environ.setdefault("GENSIM_DATA_DIR", str(ROOT / "data/gensim"))  # must precede gensim import

# WikiText number/hyphen markers -> closest GloVe token
GLOVE_ALIASES = {"@-@": "-", "@,@": ",", "@.@": ".", "'t": "n't", "−": "-"}


def train_sentences(vocab, ids):
    """Split the train stream into sentences at <eos> (keeping <eos> so it gets a vector)."""
    sents, cur = [], []
    for w in vocab.decode(ids["train"].tolist()):
        cur.append(w)
        if w == EOS:
            sents.append(cur)
            cur = []
    return sents


def align(vocab, lookup, dim, sigma, seed):
    """Build a |V| x dim table; lookup(word) returns a vector or None."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, sigma, size=(len(vocab), dim)).astype(np.float32)
    found = np.zeros(len(vocab), dtype=bool)
    for i, w in enumerate(vocab.itos):
        v = lookup(w)
        if v is not None:
            W[i] = v
            found[i] = True
    W[vocab.pad_id] = 0.0
    return torch.from_numpy(W), found


def coverage(vocab, found, train_ids):
    counts = np.bincount(train_ids.numpy(), minlength=len(vocab))
    real = np.ones(len(vocab), dtype=bool)
    real[vocab.pad_id] = False
    return {
        "type_coverage": float(found[real].sum() / real.sum()),
        "token_coverage_train": float(counts[found].sum() / counts.sum()),
        "missing_types": int((~found & real).sum()),
    }


def neighbours(W, vocab, words, k=5):
    Wn = torch.nn.functional.normalize(W, dim=1)
    out = {}
    for w in words:
        if w not in vocab.stoi:
            continue
        sims = Wn @ Wn[vocab.stoi[w]]
        sims[vocab.stoi[w]] = -1
        out[w] = [vocab.itos[i] for i in sims.topk(k).indices.tolist()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/embeddings.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    vocab, ids = load_corpus(cfg["data_config"])
    out_dir = ROOT / "data/emb"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ---- 1. own Skip-gram on the training split ----
    from gensim.models import Word2Vec

    sents = train_sentences(vocab, ids)
    w = cfg["word2vec"]
    t0 = time.perf_counter()
    w2v = Word2Vec(sents, vector_size=cfg["dim"], window=w["window"], min_count=1, sg=1,
                   negative=w["negative"], sample=w["sample"], epochs=w["epochs"],
                   workers=w["workers"], seed=cfg["seed"])
    w2v_time = time.perf_counter() - t0
    kv = w2v.wv
    sigma = float(kv.vectors.std())
    W, found = align(vocab, lambda t: kv[t] if t in kv else None, cfg["dim"], sigma, cfg["seed"])
    torch.save({"weights": W, "found": torch.from_numpy(found)}, out_dir / "w2v_300.pt")
    results["word2vec"] = {
        "method": "Skip-gram (gensim), negative sampling", **w,
        "train_sentences": len(sents), "train_tokens": int(len(ids["train"])),
        "train_time_s": w2v_time, "vector_std": sigma,
        **coverage(vocab, found, ids["train"]),
        "hardware": f"CPU, {w['workers']} worker thread(s)",
    }
    print(f"word2vec: {w2v_time:.1f}s, coverage {results['word2vec']['type_coverage']:.4f}")

    # ---- 2. public GloVe ----
    import gensim.downloader as api

    t0 = time.perf_counter()
    glove = api.load(cfg["glove"]["name"])
    load_time = time.perf_counter() - t0
    sigma_g = float(glove.vectors.std())

    def glove_lookup(t):
        t = GLOVE_ALIASES.get(t, t)
        return glove[t] if t in glove else None

    Wg, found_g = align(vocab, glove_lookup, cfg["dim"], sigma_g, cfg["seed"])
    torch.save({"weights": Wg, "found": torch.from_numpy(found_g)}, out_dir / "glove_300.pt")
    missing = [vocab.itos[i] for i in np.where(~found_g)[0] if i != vocab.pad_id]
    counts = np.bincount(ids["train"].numpy(), minlength=len(vocab))
    missing_by_freq = sorted(missing, key=lambda t: -counts[vocab.stoi[t]])
    results["glove"] = {
        "name": cfg["glove"]["name"], "source": "GloVe 6B (Wikipedia 2014 + Gigaword 5), uncased",
        "glove_vocab": len(glove.key_to_index), "aliases": GLOVE_ALIASES,
        "load_time_s": load_time, "vector_std": sigma_g,
        **coverage(vocab, found_g, ids["train"]),
        "most_frequent_missing": [(t, int(counts[vocab.stoi[t]])) for t in missing_by_freq[:25]],
        "missing_init": "N(0, sigma^2) with sigma = std of GloVe entries, seed 42; <pad> = 0",
    }
    print(f"glove: coverage {results['glove']['type_coverage']:.4f} types, "
          f"{results['glove']['token_coverage_train']:.4f} train tokens")

    words = cfg["probe_words"]
    results["neighbours"] = {"word2vec": neighbours(W, vocab, words),
                             "glove": neighbours(Wg, vocab, words)}
    save_json(results, "results/embeddings.json")


if __name__ == "__main__":
    main()
