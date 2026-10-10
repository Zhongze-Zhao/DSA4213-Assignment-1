"""Generate continuations from every LM with matched prompts and decoding settings.

    python -m src.generate --config configs/generate.yaml

Writes results/generations.json (all samples + repetition metrics + generation latency) and
results/generations.md (human-readable side-by-side table).
"""
import argparse
import pickle
import time
from collections import Counter

import numpy as np
import torch

from src.data import load_corpus, tokenize_text
from src.models import build_model
from src.utils import ROOT, get_device, load_config, save_json, set_seed


def load_neural(name, vocab_size, pad_id, device):
    ck = torch.load(ROOT / f"checkpoints/{name}.pt", map_location=device)
    model = build_model(ck["config"], vocab_size, pad_id).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["config"]["seq_len"]


def pick(probs, mode, top_k, temperature, banned, rng):
    """probs: numpy vector of next-token probabilities. Same decoding rule for every model."""
    p = probs.astype(np.float64).copy()
    p[banned] = 0.0
    if mode == "greedy":
        return int(p.argmax())
    logp = np.log(np.maximum(p, 1e-30)) / temperature
    top = np.argpartition(-logp, top_k)[:top_k]
    q = np.exp(logp[top] - logp[top].max())
    return int(top[rng.choice(len(top), p=q / q.sum())])


def make_next_dist(kind, model, ctx_len, device):
    if kind == "ngram":
        return lambda ids: model.dist(ids)

    @torch.no_grad()
    def f(ids):
        x = torch.tensor([ids[-ctx_len:]], device=device)  # no state/KV caching for any model
        return torch.softmax(model(x)[0, -1].float(), -1).cpu().numpy()
    return f


def generate(next_dist, prompt_ids, length, mode, top_k, temperature, banned, rng):
    ids = list(prompt_ids)
    for _ in range(length):
        ids.append(pick(next_dist(ids), mode, top_k, temperature, banned, rng))
    return ids[len(prompt_ids):]


def rep_metrics(tokens):
    """Repetition statistics of one continuation."""
    def ngrams(n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    g2, g3 = ngrams(2), ngrams(3)
    rep3 = sum(c - 1 for c in Counter(g3).values()) / max(1, len(g3))
    return {"distinct1": len(set(tokens)) / len(tokens), "distinct2": len(set(g2)) / max(1, len(g2)),
            "repeated_trigram_rate": rep3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generate.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    vocab, _ = load_corpus(cfg["data_config"])
    banned = [vocab.pad_id, vocab.unk_id]  # never emit <pad>/<unk> (same for all models)
    prompts = [tokenize_text(p) for p in cfg["prompts"]]
    out = {"settings": {k: cfg[k] for k in ("length", "top_k", "temperature", "samples_per_prompt")},
           "prompts": cfg["prompts"], "models": {}}

    for name in cfg["models"]:
        if name == "ngram":
            with open(ROOT / "checkpoints/ngram.pkl", "rb") as f:
                model = pickle.load(f)
            nd = make_next_dist("ngram", model, None, device)
        else:
            model, ctx = load_neural(name, len(vocab), vocab.pad_id, device)
            nd = make_next_dist("neural", model, ctx, device)
        res = {"greedy": [], "sample": [], "metrics": {}}
        rng = np.random.default_rng(cfg["seed"])
        # latency: greedy generation of `length` tokens for each prompt, batch size 1
        nd([vocab.eos_id] + vocab.encode(prompts[0]))  # warm-up
        t0 = time.perf_counter()
        for p in prompts:
            g = generate(nd, vocab.encode(p), cfg["length"], "greedy", 0, 1.0, banned, rng)
            res["greedy"].append(" ".join(vocab.decode(g)))
        if device.type == "cuda":
            torch.cuda.synchronize()
        res["ms_per_token_greedy"] = 1000 * (time.perf_counter() - t0) / (len(prompts) * cfg["length"])
        mets = {m: [] for m in ("greedy", "sample")}
        for p, g in zip(prompts, res["greedy"]):
            mets["greedy"].append(rep_metrics(g.split()))
        for p in prompts:
            samples = []
            for _ in range(cfg["samples_per_prompt"]):
                s = generate(nd, vocab.encode(p), cfg["length"], "sample", cfg["top_k"],
                             cfg["temperature"], banned, rng)
                samples.append(" ".join(vocab.decode(s)))
                mets["sample"].append(rep_metrics(vocab.decode(s)))
            res["sample"].append(samples)
        res["metrics"] = {m: {k: float(np.mean([d[k] for d in v])) for k in v[0]}
                          for m, v in mets.items()}
        out["models"][name] = res
        print(f"{name:12s} {res['ms_per_token_greedy']:.2f} ms/token | {res['metrics']}")

    save_json(out, "results/generations.json")
    lines = [f"# Generated continuations\n\nSettings: {out['settings']}; "
             "<unk>/<pad> banned for all models; first sample shown.\n"]
    for i, p in enumerate(cfg["prompts"]):
        lines.append(f"\n## Prompt {i + 1}: \"{p}\"\n")
        for name, r in out["models"].items():
            lines.append(f"- **{name} (greedy)**: {r['greedy'][i]}")
            lines.append(f"- **{name} (top-k)**: {r['sample'][i][0]}")
    (ROOT / "results/generations.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
