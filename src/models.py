"""Neural language models: vanilla RNN, LSTM and causal Transformer.

All models share the interface
    logits = model(x)                       # x: (B, T) token ids -> (B, T, V)
    logits, h = model(x, return_hidden=True) # h: (B, T, H) top-layer states (for Part III)
and predict position t only from tokens <= t (i.e. the target x[t+1] uses only preceding tokens).

Run `python -m src.models` for a causality and parameter-count check.
"""
import torch
import torch.nn as nn


def make_embedding(vocab_size, emb_dim, pad_id, weights=None, freeze=False):
    emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
    if weights is not None:  # Part II: pretrained table (word2vec / GloVe)
        emb.weight.data.copy_(weights)
    emb.weight.requires_grad = not freeze  # freeze=True keeps the ENTIRE table fixed
    return emb


class RecurrentLM(nn.Module):
    """Embedding -> multi-layer RNN or LSTM -> linear softmax layer."""

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout, cell, pad_id,
                 emb_weights=None, freeze_emb=False, **_):
        super().__init__()
        self.emb = make_embedding(vocab_size, emb_dim, pad_id, emb_weights, freeze_emb)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(emb_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x, return_hidden=False):
        h, _ = self.rnn(self.drop(self.emb(x)))  # unidirectional => causal by construction
        logits = self.out(self.drop(h))
        return (logits, h) if return_hidden else logits


class TransformerLM(nn.Module):
    """Decoder-only Transformer (pre-LN) with learned positions and a causal attention mask."""

    def __init__(self, vocab_size, emb_dim, num_heads, num_layers, ff_dim, dropout, max_len,
                 pad_id, emb_weights=None, freeze_emb=False, **_):
        super().__init__()
        self.emb = make_embedding(vocab_size, emb_dim, pad_id, emb_weights, freeze_emb)
        self.pos = nn.Embedding(max_len, emb_dim)
        layer = nn.TransformerEncoderLayer(emb_dim, num_heads, ff_dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, vocab_size)
        self.max_len = max_len
        self.hidden_dim = emb_dim

    def forward(self, x, return_hidden=False):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        # Upper-triangular -inf mask: position t can only attend to positions <= t.
        # Sequences are right-padded, so real tokens never attend to padding.
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.drop(self.emb(x) + self.pos(pos))
        h = self.norm(self.blocks(h, mask=mask, is_causal=True))
        logits = self.out(self.drop(h))
        return (logits, h) if return_hidden else logits


def build_model(cfg, vocab_size, pad_id, emb_weights=None):
    m = cfg["model"]
    if m["type"] in ("rnn", "lstm"):
        return RecurrentLM(vocab_size, cell=m["type"], pad_id=pad_id, emb_weights=emb_weights, **m)
    if m["type"] == "transformer":
        return TransformerLM(vocab_size, pad_id=pad_id, emb_weights=emb_weights,
                             max_len=cfg["seq_len"], **m)
    raise ValueError(m["type"])


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.emb.weight.numel()
    out = sum(p.numel() for p in model.out.parameters())
    return {"total": total, "trainable": trainable, "input_embedding": emb,
            "output_layer": out, "non_embedding": total - emb - out}


if __name__ == "__main__":
    # Causality check: changing future tokens must not change earlier logits.
    torch.manual_seed(0)
    V, T = 1000, 16
    cfgs = {
        "rnn": {"type": "rnn", "emb_dim": 32, "hidden_dim": 64, "num_layers": 2, "dropout": 0.0},
        "lstm": {"type": "lstm", "emb_dim": 32, "hidden_dim": 64, "num_layers": 2, "dropout": 0.0},
        "transformer": {"type": "transformer", "emb_dim": 32, "num_heads": 4, "num_layers": 2,
                        "ff_dim": 64, "dropout": 0.0},
    }
    for name, m in cfgs.items():
        model = build_model({"model": m, "seq_len": T}, V, 0).eval()
        x = torch.randint(3, V, (2, T))
        x2 = x.clone()
        x2[:, 8:] = torch.randint(3, V, (2, T - 8))
        with torch.no_grad():
            diff = (model(x)[:, :8] - model(x2)[:, :8]).abs().max().item()
        print(f"{name:12s} max |logit change| at t<8 after editing t>=8: {diff:.2e}  "
              f"-> causal={diff < 1e-5}")
