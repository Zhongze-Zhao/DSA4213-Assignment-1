"""Clean train / val / test cross-entropy at the selected (best-validation) checkpoints.

    python -m src.clean_fit

The training loss logged in history.csv is averaged over each epoch while the weights change and
with dropout active, so it over-states the training loss (and makes val - train negative early on).
Here every split is re-scored in eval mode (no dropout) with the final selected weights and the
same 128-token windows, giving a like-for-like generalisation gap. Writes results/lm/clean_fit.json.
"""
import json
import math
import pickle

import torch

from src.data import load_corpus, make_windows
from src.generate import load_neural
from src.ngram import score_windows
from src.train_lm import run_epoch
from src.utils import ROOT, get_device, limit_gpu_memory, save_json

NEURAL = ["rnn", "lstm", "transformer", "lstm_w2v", "lstm_glove", "lstm_smallinit"]


def main():
    device = get_device()
    limit_gpu_memory()
    vocab, ids = load_corpus()
    win = {s: make_windows(ids[s], 128, vocab.pad_id) for s in ("train", "validation", "test")}
    out = {}
    names = NEURAL + [f"{m}_seed{s}" for m in NEURAL[:5] for s in (1, 2)]
    for name in names:
        if not (ROOT / f"checkpoints/{name}.pt").exists():
            continue
        model, _ = load_neural(name, len(vocab), vocab.pad_id, device)
        ce = {s: run_epoch(model, w, {}, device, vocab.pad_id, bs=32) for s, w in win.items()}
        out[name] = {"train_ce": ce["train"], "val_ce": ce["validation"], "test_ce": ce["test"],
                     "gap_val_minus_train": ce["validation"] - ce["train"],
                     "train_ppl": math.exp(ce["train"])}
        print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    with open(ROOT / "checkpoints/ngram.pkl", "rb") as f:
        lm = pickle.load(f)
    ce = {s: score_windows(lm, w, vocab.pad_id)[0] for s, w in win.items()}
    out["ngram"] = {"train_ce": ce["train"], "val_ce": ce["validation"], "test_ce": ce["test"],
                    "gap_val_minus_train": ce["validation"] - ce["train"], "train_ppl": math.exp(ce["train"])}
    print("ngram", {k: round(v, 3) for k, v in out["ngram"].items()})
    save_json(out, "results/lm/clean_fit.json")


if __name__ == "__main__":
    main()