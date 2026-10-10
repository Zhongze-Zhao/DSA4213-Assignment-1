"""Controlled cost benchmark. Sequential training wall-clock on a laptop GPU drifts with thermal
throttling, so here all models are timed in an interleaved order (round-robin, several rounds,
median), using the same batch shape as training and evaluation.

    python -m src.benchmark --config configs/benchmark.yaml
"""
import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from src.data import load_corpus, make_windows
from src.models import build_model, count_params
from src.train_lm import load_emb_weights, sync
from src.utils import get_device, hardware_info, load_config, save_json, set_seed


def time_steps(model, batches, pad_id, train, opt=None):
    model.train(train)
    sync(batches[0].device)
    t0 = time.perf_counter()
    ntok = 0
    with torch.set_grad_enabled(train):
        for b in batches:
            x, y = b[:, :-1], b[:, 1:]
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=pad_id)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            ntok += (y != pad_id).sum().item()
    sync(batches[0].device)
    return ntok / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmark.yaml")
    args = ap.parse_args()
    bcfg = load_config(args.config)
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = get_device()
    vocab, ids = load_corpus()
    models = {}
    for name in bcfg["models"]:
        cfg = load_config(f"configs/lm_{name}.yaml")
        if "embedding" in cfg:
            cfg["model"]["freeze_emb"] = cfg["embedding"]["freeze"]
        m = build_model(cfg, len(vocab), vocab.pad_id, load_emb_weights(cfg)).to(device)
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
        win = make_windows(ids["train"], cfg["seq_len"], vocab.pad_id)
        batches = [win[i * cfg["batch_size"]:(i + 1) * cfg["batch_size"]].to(device)
                   for i in range(bcfg["steps"])]
        time_steps(m, batches[:bcfg["warmup"]], vocab.pad_id, True, opt)  # warm-up
        models[name] = (m, opt, batches, count_params(m))
    res = {n: {"train": [], "infer": []} for n in models}
    for r in range(bcfg["rounds"]):
        order = list(models) if r % 2 == 0 else list(models)[::-1]  # alternate order
        for n in order:
            m, opt, batches, _ = models[n]
            res[n]["train"].append(time_steps(m, batches, vocab.pad_id, True, opt))
            res[n]["infer"].append(time_steps(m, batches, vocab.pad_id, False))
        print(f"round {r + 1}: " + ", ".join(f"{n} {res[n]['train'][-1]:,.0f}" for n in models))
    out = {"settings": bcfg, "hardware": hardware_info(), "models": {}}
    for n, (_, _, _, params) in models.items():
        out["models"][n] = {"train_tokens_per_s": statistics.median(res[n]["train"]),
                            "infer_tokens_per_s": statistics.median(res[n]["infer"]),
                            "train_all": res[n]["train"], "infer_all": res[n]["infer"],
                            "params": params}
        print(f"{n:12s} train {out['models'][n]['train_tokens_per_s']:9,.0f} tok/s | "
              f"infer {out['models'][n]['infer_tokens_per_s']:9,.0f} tok/s")
    save_json(out, "results/benchmark.json")


if __name__ == "__main__":
    main()
