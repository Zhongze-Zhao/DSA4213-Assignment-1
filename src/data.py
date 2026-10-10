"""WikiText-2 loading, preprocessing and vocabulary shared by all language models.

Usage:
    python -m src.data --config configs/data.yaml   # builds cache and prints statistics
"""
import argparse
from collections import Counter

import torch

from src.utils import ROOT, load_config, save_json

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
SPECIALS = [PAD, UNK, EOS]  # PAD gets index 0


class Vocab:
    def __init__(self, counter: Counter, min_freq: int = 1):
        words = sorted((w for w, c in counter.items() if c >= min_freq and w not in SPECIALS),
                       key=lambda w: (-counter[w], w))
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.unk_id, self.eos_id = 0, 1, 2

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def tokenize_lines(lines, cfg):
    """Turn raw dataset lines into a flat token list (whitespace split; v1 is pre-tokenized)."""
    tokens = []
    for line in lines:
        line = line.strip()
        if not line and cfg["skip_empty_lines"]:
            continue
        if cfg["lowercase"]:
            line = line.lower()
        tokens.extend(line.split())
        if cfg["add_eos"]:
            tokens.append(EOS)
    return tokens


def tokenize_text(text: str, lowercase: bool = True):
    """Tokenize free text (prompts, downstream data) consistently with the LM corpus."""
    return (text.lower() if lowercase else text).split()


def build(cfg):
    from datasets import load_dataset

    ds = load_dataset(cfg["dataset"], cfg["config"], cache_dir=str(ROOT / cfg["cache_dir"]))
    toks = {s: tokenize_lines(ds[s]["text"], cfg) for s in ["train", "validation", "test"]}
    vocab = Vocab(Counter(toks["train"]), cfg["min_freq"])  # vocabulary from train only
    ids = {s: torch.tensor(vocab.encode(t), dtype=torch.long) for s, t in toks.items()}
    return vocab, ids, toks


def load_corpus(config_path="configs/data.yaml"):
    """Return (vocab, {'train','validation','test'} -> 1D LongTensor). Builds cache on first call."""
    cfg = load_config(config_path)
    cache = ROOT / cfg["processed_dir"] / "wikitext2.pt"
    if cache.exists():
        obj = torch.load(cache)
        vocab = Vocab(Counter())
        vocab.itos = obj["itos"]
        vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
        return vocab, obj["ids"]
    vocab, ids, _ = build(cfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"itos": vocab.itos, "ids": ids}, cache)
    return vocab, ids


def corpus_stats(vocab, ids):
    stats = {"vocab_size": len(vocab)}
    for split, t in ids.items():
        n = len(t)
        stats[split] = {
            "tokens": n,
            "unk_rate": round((t == vocab.unk_id).sum().item() / n, 4),
            "eos_count": (t == vocab.eos_id).sum().item(),
        }
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    args = p.parse_args()
    vocab, ids = load_corpus(args.config)
    stats = corpus_stats(vocab, ids)
    save_json(stats, "results/data_stats.json")
    print(stats)
    print("most frequent:", vocab.itos[3:23])
    print("sample decode:", " ".join(vocab.decode(ids["train"][:40].tolist())))
