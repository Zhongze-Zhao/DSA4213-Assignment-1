"""Train and evaluate one neural language model.

    python -m src.train_lm --config configs/lm_lstm.yaml           # full run
    python -m src.train_lm --config configs/lm_lstm.yaml --quick   # ~1 min smoke test

Outputs (full run): checkpoints/<name>.pt, results/lm/<name>/{metrics.json, history.csv, config.json}
"""
import argparse
import math
import time

import pandas as pd
import torch
import torch.nn.functional as F

from src.data import load_corpus, make_windows
from src.models import build_model, count_params
from src.utils import ROOT, get_device, hardware_info, load_config, save_json, set_seed


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_epoch(model, windows, cfg, device, pad_id, opt=None, sched=None, bs=None, log=None):
    """One pass over `windows`. Training if `opt` is given. Returns mean NLL per non-pad target."""
    train = opt is not None
    model.train(train)
    bs = bs or cfg["batch_size"]
    order = torch.randperm(len(windows)) if train else torch.arange(len(windows))
    tot_loss = torch.zeros((), device=device)
    tot_tok = 0
    with torch.set_grad_enabled(train):
        for i in range(0, len(windows), bs):
            batch = windows[order[i:i + bs]].to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            # sum of NLL over real targets only: padding positions are ignored
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                       ignore_index=pad_id, reduction="sum")
            ntok = (y != pad_id).sum().item()
            if train:
                opt.zero_grad(set_to_none=True)
                (loss_sum / ntok).backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
                sched.step()
                if log is not None:
                    log.append(((loss_sum / ntok).detach(), gn.detach()))
            tot_loss += loss_sum.detach()
            tot_tok += ntok
    return tot_loss.item() / tot_tok


def lr_lambda(cfg, total_steps):
    def f(step):
        if step < cfg["warmup_steps"]:
            return (step + 1) / cfg["warmup_steps"]
        prog = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


@torch.no_grad()
def time_inference(model, windows, cfg, device, pad_id, repeats=3):
    """Common inference workload: score the whole test set (same windows, batch size for all
    models). Returns median wall-clock seconds and target tokens/second."""
    model.eval()
    run_epoch(model, windows[: cfg["eval_batch_size"] * 2], cfg, device, pad_id,
              bs=cfg["eval_batch_size"])  # warm-up (cuDNN autotune, allocator)
    times = []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        run_epoch(model, windows, cfg, device, pad_id, bs=cfg["eval_batch_size"])
        sync(device)
        times.append(time.perf_counter() - t0)
    sec = sorted(times)[len(times) // 2]
    ntok = (windows[:, 1:] != pad_id).sum().item()
    return sec, ntok / sec


def load_emb_weights(cfg):
    if "embedding" not in cfg:
        return None
    return torch.load(ROOT / cfg["embedding"]["path"])["weights"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    ap.add_argument("--seed", type=int, default=None, help="override seed (stability runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["name"] = f"{cfg['name']}_seed{args.seed}"
    if args.quick:
        cfg.update(epochs=2, warmup_steps=20)
    set_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = get_device()

    vocab, ids = load_corpus(cfg["data_config"])
    win = {s: make_windows(ids[s], cfg["seq_len"], vocab.pad_id) for s in ids}
    if args.quick:
        win = {"train": win["train"][:300], "validation": win["validation"][:50],
               "test": win["test"][:50]}

    emb_w = load_emb_weights(cfg)
    if "embedding" in cfg:
        cfg["model"]["freeze_emb"] = cfg["embedding"]["freeze"]
    model = build_model(cfg, len(vocab), vocab.pad_id, emb_w).to(device)
    params = count_params(model)
    print(f"[{cfg['name']}] params: {params}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = math.ceil(len(win["train"]) / cfg["batch_size"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, steps_per_epoch * cfg["epochs"]))

    out_dir = ROOT / ("results/quick" if args.quick else "results/lm") / cfg["name"]
    ckpt_path = ROOT / ("checkpoints/quick" if args.quick else "checkpoints") / f"{cfg['name']}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    val0 = run_epoch(model, win["validation"], cfg, device, vocab.pad_id, bs=cfg["eval_batch_size"])
    history = [{"epoch": 0, "train_loss": None, "val_loss": val0, "epoch_time_s": 0.0}]
    print(f"epoch 0 | val {val0:.3f}")
    best_val, best_epoch, train_time = float("inf"), 0, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(1, cfg["epochs"] + 1):
        log = []
        sync(device)
        t0 = time.perf_counter()
        tr = run_epoch(model, win["train"], cfg, device, vocab.pad_id, opt, sched, log=log)
        sync(device)
        dt = time.perf_counter() - t0  # training time excludes validation
        train_time += dt
        va = run_epoch(model, win["validation"], cfg, device, vocab.pad_id, bs=cfg["eval_batch_size"])
        step_losses = torch.stack([l for l, _ in log]).float()
        grad_norms = torch.stack([g for _, g in log]).float()
        history.append({"epoch": ep, "train_loss": tr, "val_loss": va, "epoch_time_s": dt,
                        "step_loss_std": step_losses.std().item(),
                        "grad_norm_mean": grad_norms.mean().item(),
                        "grad_norm_max": grad_norms.max().item(),
                        "lr_end": sched.get_last_lr()[0]})
        print(f"epoch {ep:2d} | train {tr:.3f} | val {va:.3f} (ppl {math.exp(va):7.1f}) | "
              f"{dt:5.1f}s | gnorm {grad_norms.mean():.2f}")
        if va < best_val:  # model selection on validation only
            best_val, best_epoch = va, ep
            torch.save({"state_dict": model.state_dict(), "config": cfg, "itos_len": len(vocab)},
                       ckpt_path)

    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    test = run_epoch(model, win["test"], cfg, device, vocab.pad_id, bs=cfg["eval_batch_size"])
    inf_sec, inf_tps = time_inference(model, win["test"], cfg, device, vocab.pad_id)
    metrics = {
        "name": cfg["name"], "best_epoch": best_epoch,
        "val_loss": best_val, "val_ppl": math.exp(best_val),
        "test_loss": test, "test_ppl": math.exp(test),
        "test_targets": int((win["test"][:, 1:] != vocab.pad_id).sum()),
        "params": params,
        "train_time_s": train_time, "sec_per_epoch": train_time / cfg["epochs"],
        "epochs": cfg["epochs"],
        "train_tokens_per_s": cfg["epochs"] * int((win["train"][:, 1:] != vocab.pad_id).sum()) / train_time,
        "inference_test_s": inf_sec, "inference_tokens_per_s": inf_tps,
        "peak_gpu_mem_MB": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "hardware": hardware_info(),
    }
    print(f"[{cfg['name']}] best epoch {best_epoch} | val ppl {metrics['val_ppl']:.1f} | "
          f"test ppl {metrics['test_ppl']:.1f} | train {train_time:.0f}s | "
          f"infer {inf_tps:,.0f} tok/s")
    save_json(metrics, out_dir / "metrics.json")
    save_json(cfg, out_dir / "config.json")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)


if __name__ == "__main__":
    main()
