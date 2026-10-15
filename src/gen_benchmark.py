"""Controlled generation-latency benchmark (Part I inference cost).

    python -m src.gen_benchmark --config configs/generate.yaml

Times batch-1 greedy decoding of the model itself (forward pass + argmax on the GPU; no CPU-side
sampling code), for two decoding implementations:
  * recompute   : every step re-encodes the whole context (what src/generate.py does)
  * incremental : RNN/LSTM carry their recurrent state, the Transformer uses a key/value cache,
                  so each step only processes the newest token
and two generation lengths (40 and 120 new tokens; prompt + 120 stays within the 128-token
context, so positions are never truncated). Protocol: warm-up, then `rounds` round-robin rounds
over models in alternating order, median over rounds. Before timing, incremental decoding is
checked against the full forward pass (max |logit difference|). The KN 4-gram (CPU, 1 thread) is
timed with the same loop as a reference. Writes results/gen_benchmark.json.
"""
import argparse
import pickle
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_corpus, tokenize_text
from src.generate import load_neural
from src.utils import ROOT, get_device, hardware_info, limit_gpu_memory, load_config, save_json, set_seed

NEURAL = ["rnn", "lstm", "transformer"]
LENGTHS = [40, 120]
ROUNDS = 5


# ---------------- incremental decoding ----------------
class RecurrentState:
    def __init__(self, model):
        self.m, self.state = model, None

    def step(self, x):  # x: (1, t) new tokens -> logits of the last one
        h, self.state = self.m.rnn(self.m.emb(x), self.state)
        return self.m.out(h[:, -1])


class TransformerKVCache:
    """Key/value cache for the pre-LN TransformerLM (src/models.py), eval mode (no dropout)."""

    def __init__(self, model):
        self.m, self.k, self.v, self.pos = model, None, None, 0

    def step(self, x):
        m = self.m
        t = x.size(1)
        h = m.emb(x) + m.pos(torch.arange(self.pos, self.pos + t, device=x.device))
        new_k, new_v = [], []
        for i, layer in enumerate(m.blocks.layers):
            sa = layer.self_attn
            nh = sa.num_heads
            q, k, v = F.linear(layer.norm1(h), sa.in_proj_weight, sa.in_proj_bias).chunk(3, -1)
            q, k, v = (z.view(1, t, nh, -1).transpose(1, 2) for z in (q, k, v))  # (1, nh, t, hd)
            if self.k is not None:
                k, v = torch.cat([self.k[i], k], 2), torch.cat([self.v[i], v], 2)
            new_k.append(k)
            new_v.append(v)
            # prefill (empty cache): causal mask; single new token: it may attend to everything cached
            o = F.scaled_dot_product_attention(q, k, v, is_causal=self.k is None and t > 1)
            h = h + sa.out_proj(o.transpose(1, 2).reshape(1, t, -1))
            h = h + layer.linear2(layer.activation(layer.linear1(layer.norm2(h))))
        self.k, self.v, self.pos = new_k, new_v, self.pos + t
        return m.out(m.norm(h)[:, -1])


def make_stepper(name, model):
    return TransformerKVCache(model) if name == "transformer" else RecurrentState(model)


@torch.no_grad()
def check_incremental(name, model, vocab_size, device, T=100, prefill=20):
    """Max |logit difference| between incremental decoding and the full forward pass."""
    g = torch.Generator().manual_seed(0)
    x = torch.randint(3, vocab_size, (1, T), generator=g).to(device)
    full = model(x)[0]
    st = make_stepper(name, model)
    diffs = [(st.step(x[:, :prefill])[0] - full[prefill - 1]).abs().max()]
    for t in range(prefill, T):
        diffs.append((st.step(x[:, t:t + 1])[0] - full[t]).abs().max())
    return float(torch.stack(diffs).max()), float(full.abs().max())


# ---------------- timed greedy decoding ----------------
@torch.no_grad()
def greedy(name, model, prompt, length, mode, device):
    ids = list(prompt)
    if mode == "recompute":
        for _ in range(length):
            nxt = model(torch.tensor([ids], device=device))[0, -1].argmax()
            ids.append(int(nxt))  # .item(): the next step needs the token, so this sync is inherent
    else:
        st = make_stepper(name, model)
        logits = st.step(torch.tensor([ids], device=device))
        for _ in range(length):
            nxt = int(logits[0].argmax())
            ids.append(nxt)
            logits = st.step(torch.tensor([[nxt]], device=device))
    return ids[len(prompt):]


def time_run(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generate.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(42)
    torch.set_num_threads(1)  # n-gram reference on one CPU thread, as in Part I
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = get_device()
    limit_gpu_memory()
    vocab, _ = load_corpus(cfg["data_config"])
    prompts = [vocab.encode(tokenize_text(p)) for p in cfg["prompts"]]
    assert max(map(len, prompts)) + max(LENGTHS) <= 128

    models = {n: load_neural(n, len(vocab), vocab.pad_id, device)[0] for n in NEURAL}
    out = {"settings": {"lengths": LENGTHS, "rounds": ROUNDS, "prompts": cfg["prompts"], "batch_size": 1,
                        "decoding": "greedy, argmax on device"}, "hardware": hardware_info(),
           "check": {}, "agreement": {}, "ms_per_token": {}}

    # 1) correctness of incremental decoding
    for n, m in models.items():
        diff, scale = check_incremental(n, m, len(vocab), device)
        out["check"][n] = {"max_abs_logit_diff": diff, "max_abs_logit": scale}
        print(f"check {n:12s} max |logit diff| {diff:.2e} (logit scale {scale:.1f})", flush=True)
        assert diff < 1e-2 * scale, f"incremental decoding of {n} does not match the full forward pass"
    for n, m in models.items():  # same greedy tokens from both implementations?
        same = [greedy(n, m, p, 40, "recompute", device) == greedy(n, m, p, 40, "incremental", device)
                for p in prompts]
        out["agreement"][n] = sum(same) / len(same)

    # 2) timing: warm-up, then round-robin rounds in alternating order
    configs = [(n, mode, L) for n in NEURAL for mode in ("recompute", "incremental") for L in LENGTHS]
    for n, mode, L in configs:
        greedy(n, models[n], prompts[0], 10, mode, device)
    times = {c: [] for c in configs}
    for r in range(ROUNDS):
        order = configs if r % 2 == 0 else configs[::-1]
        for n, mode, L in order:
            sec, _ = time_run(lambda: [greedy(n, models[n], p, L, mode, device) for p in prompts], device)
            times[(n, mode, L)].append(1000 * sec / (L * len(prompts)))
        print(f"round {r + 1} done", flush=True)
    for (n, mode, L), v in times.items():
        out["ms_per_token"].setdefault(n, {}).setdefault(mode, {})[str(L)] = {
            "median": statistics.median(v), "min": min(v), "max": max(v), "all": v}
        print(f"{n:12s} {mode:11s} L={L:3d}: median {statistics.median(v):.2f} ms/token "
              f"(range {min(v):.2f}-{max(v):.2f})")

    # 3) n-gram reference (CPU): its context is only the last 3 tokens, so it is incremental by nature
    with open(ROOT / "checkpoints/ngram.pkl", "rb") as f:
        ng = pickle.load(f)

    def ng_greedy(p, L):
        ids = list(p)
        for _ in range(L):
            ids.append(int(np.argmax(ng.dist(ids))))
        return ids

    ng_greedy(prompts[0], 5)
    out["ms_per_token"]["ngram"] = {}
    for L in LENGTHS:
        v = [1000 * time_run(lambda: [ng_greedy(p, L) for p in prompts], torch.device("cpu"))[0]
             / (L * len(prompts)) for _ in range(ROUNDS)]
        out["ms_per_token"]["ngram"][str(L)] = {"median": statistics.median(v), "min": min(v), "max": max(v),
                                                "all": v}
        print(f"ngram (CPU)  L={L:3d}: median {statistics.median(v):.2f} ms/token")
    save_json(out, "results/gen_benchmark.json")


if __name__ == "__main__":
    main()