"""Training-time measurement under shared conditions (Part I cost).

    python -m src.lockstep_train

The Part I models were trained one after another, so their wall-clock times also reflect when each
ran (the laptop GPU throttles as it heats up). Here the RNN, LSTM and Transformer are re-trained
from the same seed-42 initialisation with the full Part I procedure (20 epochs, same configs), but
in lockstep: epoch e of every model runs before epoch e+1 of any model, in alternating order, so
all three see the same thermal history. Only the timings are used; the checkpoints and results of
Parts I-III are not touched. Validation loss is logged as a sanity check that the runs train as
before. Writes results/lockstep_timing.json.
"""
import math
import statistics
import time

import torch

from src.data import load_corpus, make_windows
from src.models import build_model
from src.train_lm import lr_lambda, run_epoch, sync
from src.utils import get_device, hardware_info, limit_gpu_memory, load_config, save_json, set_seed

NAMES = ["rnn", "lstm", "transformer"]


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = get_device()
    limit_gpu_memory()
    vocab, ids = load_corpus()
    runs = {}
    for n in NAMES:
        cfg = load_config(f"configs/lm_{n}.yaml")
        set_seed(cfg["seed"])  # same initial weights as the Part I run
        model = build_model(cfg, len(vocab), vocab.pad_id).to(device)
        win = {s: make_windows(ids[s], cfg["seq_len"], vocab.pad_id) for s in ("train", "validation")}
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps = math.ceil(len(win["train"]) / cfg["batch_size"]) * cfg["epochs"]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, steps))
        runs[n] = {"cfg": cfg, "model": model, "win": win, "opt": opt, "sched": sched,
                   "epoch_s": [], "val_loss": []}
    epochs = runs[NAMES[0]]["cfg"]["epochs"]
    for ep in range(1, epochs + 1):
        order = NAMES if ep % 2 else NAMES[::-1]
        for n in order:
            r = runs[n]
            sync(device)
            t0 = time.perf_counter()
            run_epoch(r["model"], r["win"]["train"], r["cfg"], device, vocab.pad_id, r["opt"], r["sched"])
            sync(device)
            r["epoch_s"].append(time.perf_counter() - t0)
            r["val_loss"].append(run_epoch(r["model"], r["win"]["validation"], r["cfg"], device, vocab.pad_id,
                                           bs=r["cfg"]["eval_batch_size"]))
        print(f"epoch {ep:2d}: " + " | ".join(f"{n} {runs[n]['epoch_s'][-1]:.1f}s val {runs[n]['val_loss'][-1]:.3f}"
                                             for n in NAMES), flush=True)
    out = {"protocol": "lockstep, alternating order per epoch, seed 42, Part I configs", "hardware": hardware_info(),
           "models": {}}
    for n in NAMES:
        r = runs[n]
        ntok = r["cfg"]["epochs"] * int((r["win"]["train"][:, 1:] != vocab.pad_id).sum())
        out["models"][n] = {"train_time_s": sum(r["epoch_s"]), "median_epoch_s": statistics.median(r["epoch_s"]),
                            "min_epoch_s": min(r["epoch_s"]), "max_epoch_s": max(r["epoch_s"]),
                            "train_tokens_per_s": ntok / sum(r["epoch_s"]),
                            "epoch_s": r["epoch_s"], "val_loss": r["val_loss"], "best_val_loss": min(r["val_loss"])}
        print(f"{n:12s} total {sum(r['epoch_s']):.0f}s | median {statistics.median(r['epoch_s']):.1f} s/epoch | "
              f"best val {min(r['val_loss']):.3f}")
    save_json(out, "results/lockstep_timing.json")


if __name__ == "__main__":
    main()