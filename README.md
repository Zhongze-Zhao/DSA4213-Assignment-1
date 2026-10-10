# DSA4213 Assignment 1 — Language Models, Embeddings, and Representations

Core question: **does better next-token prediction (lower perplexity) give better representations
for a downstream task?** Part I compares an n-gram, RNN, LSTM and Transformer LM on WikiText-2;
Part II ablates the LSTM's input embeddings (random/trainable vs own Skip-gram/frozen vs GloVe/frozen);
Part III freezes the LMs and classifies AG News with logistic regression on their representations.

## Environment

Tested on WSL2 Ubuntu, Python 3.12.3, PyTorch 2.11.0 + CUDA 12.8, NVIDIA RTX 5070 Ti Laptop GPU (12 GB),
24 CPU threads, 15 GB RAM.

```bash
python3 -m venv .venv && source .venv/bin/activate      # or: uv venv .venv
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx needs cu128+
pip install -r requirements.txt
```

## Data (downloaded automatically, nothing is redistributed in this repo)

| Use | Source | Access |
|---|---|---|
| Parts I–II corpus | WikiText-2 (`Salesforce/wikitext`, config `wikitext-2-v1`), CC BY-SA | HuggingFace `datasets` |
| Part II public vectors | GloVe 6B 300d (Pennington et al., 2014) | `gensim.downloader`, name `glove-wiki-gigaword-300` |
| Part III task | AG News (`fancyzhx/ag_news`, Zhang et al., 2015) | HuggingFace `datasets` |
| OOV comparison only | SST-2 validation (`stanfordnlp/sst2`) | HuggingFace `datasets` |

Caches go to `data/` (git-ignored).

### Preprocessing (`configs/data.yaml`, `src/data.py`)

- `wikitext-2-v1` official train/validation/test splits; already tokenized, rare words already `<unk>`.
- Lowercased (matches uncased GloVe 6B); whitespace split; empty lines dropped; `<eos>` appended to every line.
- Vocabulary = all train tokens (min_freq 1) + `<pad>`(0), `<unk>`, `<eos>` → **28,913** types, shared by all LMs.
  Lowercasing introduces no new validation/test OOV.
- LM evaluation (common to all four models): each split is cut into non-overlapping 128-token windows
  (`make_windows`); every token except the first is a target exactly once (244,101 test targets);
  each target is predicted only from preceding tokens **inside its window**; the last window is
  right-padded and padded targets are excluded from the loss (`ignore_index=<pad>`).
- AG News text is mapped to WikiText conventions by `wiki_tokenize` (lowercase, split punctuation,
  `didn't → didn 't`, `3.5 → 3 @.@ 5`, `well-known → well @-@ known`, AG-News escapes such as `#36;` → `$`).

## Reproducing everything

```bash
scripts/run_all.sh            # ≈3 h on the GPU above; or run the parts separately:
python -m src.data            # corpus statistics -> results/data_stats.json
python -m src.models          # causality unit check for RNN/LSTM/Transformer
scripts/run_part1.sh          # KN n-gram + 3 neural LMs
scripts/run_part2.sh          # Skip-gram + GloVe tables, 2 frozen-embedding LSTMs
scripts/run_part3.sh          # generations, AG News downstream, error analysis, benchmark, tables & figures
scripts/run_seeds.sh          # seeds 1, 2 for the three Part II LSTMs
python -m src.train_lm --config configs/lm_lstm_smallinit.yaml   # supplementary Part II run
cd report && pdflatex report.tex && pdflatex report.tex            # build the PDF report
```

**WSL2 note.** On Windows the NVIDIA driver can silently back CUDA allocations with shared system RAM; a
3.5 GB logits tensor once froze the whole WSL VM. All scripts therefore cap PyTorch at 70% of VRAM
(`utils.limit_gpu_memory`) and feature extraction never materialises vocabulary logits (`model.encode`).
Setting *NVIDIA Control Panel → CUDA – Sysmem Fallback Policy → Prefer No Sysmem Fallback* also helps.

Smoke test (≈1 min): `python -m src.train_lm --config configs/lm_lstm.yaml --quick` and
`python -m src.ngram --quick` (outputs go to `results/quick/`).

**Seeds**: 42 everywhere (Python, NumPy, PyTorch via `src/utils.set_seed`; gensim `seed=42, workers=1`;
sklearn `random_state=42`). Extra seeds 1 and 2 only in `run_seeds.sh`. cuDNN kernels are not forced to be
deterministic, so GPU reruns can differ in the last decimals.

## Configurations (`configs/`)

| File | What |
|---|---|
| `data.yaml` | preprocessing |
| `ngram.yaml` | interpolated Kneser-Ney; order chosen from {2,3,4} on validation |
| `lm_base.yaml` | shared neural training procedure: context 128, batch 32, 20 epochs, AdamW lr 1e-3, wd 0.01, 300 warm-up steps + cosine to 10%, grad-clip 1.0, best-validation checkpoint |
| `lm_rnn.yaml`, `lm_lstm.yaml`, `lm_transformer.yaml` | Part I architectures (all 300-d random trainable embeddings, dropout 0.3) |
| `lm_lstm_w2v.yaml`, `lm_lstm_glove.yaml` | Part II frozen-embedding LSTMs (identical to `lm_lstm.yaml` otherwise) |
| `lm_lstm_smallinit.yaml` | supplementary: trainable random embeddings initialised N(0, 0.1²) instead of N(0, 1) |
| `benchmark.yaml` | round-robin cost benchmark (5 rounds × 60 steps) |
| `embeddings.yaml` | Skip-gram (dim 300, window 5, neg 5, 10 epochs) and GloVe settings |
| `generate.yaml` | 5 prompts, 40 tokens, greedy and top-k 40 / T 0.8, `<unk>`/`<pad>` banned |
| `downstream.yaml` | AG News 20k stratified train subset / 7.6k test, poolings, LR C grid |

## Code (`src/`)

| File | Purpose |
|---|---|
| `data.py` | loading, vocabulary, windows, `wiki_tokenize` |
| `ngram.py` | own interpolated Kneser-Ney implementation (scoring + full distributions) |
| `models.py` | `RecurrentLM` (RNN/LSTM) and `TransformerLM` (pre-LN, causal mask) + causality check |
| `train_lm.py` | training loop, loss curves, timing, checkpointing, test evaluation |
| `embeddings.py` | Skip-gram training (timed), GloVe alignment, coverage, neighbours |
| `generate.py` | matched decoding for all LMs, repetition metrics, generation latency |
| `downstream.py` | frozen feature extraction (last / mean / input-embedding mean), logistic regression, error analysis |
| `error_analysis.py` | McNemar tests, confusions, per-class / length / OOV accuracy, example errors |
| `benchmark.py` | interleaved training/inference throughput benchmark |
| `summarize.py` | builds `results/tables.md` and `results/figures/*.png` from saved results |
| `latex_tables.py` | builds `report/generated/*.tex` (all report tables and inline numbers) and embedding-drift stats |

## Outputs (`results/`)

`data_stats.json`, `lm/<model>/{metrics.json,history.csv,config.json}`, `embeddings.json`,
`generations.{json,md}`, `downstream/{summary.json,results.csv,test_predictions.csv,error_stats.json,errors.md}`,
`benchmark.json`, `embedding_drift.json`, `tables.md`, `figures/`.

## Timing procedure

Training time = wall-clock of the training passes only (validation excluded), with
`torch.cuda.synchronize()` around each epoch. Because runs were sequential on a laptop GPU that throttles
when hot, the cost comparison in the report uses `src/benchmark.py`: all models timed round-robin
(alternating order, 5 rounds × 60 batches of 32×128 tokens, median). Inference workload (identical for all models) = scoring the
full test set in 128-token windows with batch size 32; median of 3 timed passes after a warm-up.
Generation latency = greedy decoding of 40 tokens for 5 prompts at batch size 1, recomputing the full
context each step (no state/KV caching for any model). The n-gram model runs on 1 CPU thread.

## Attribution and tool use

- Neural LM code written for this assignment in PyTorch, following the structure of the course
  notebooks (DSA4213 Lectures 3–7). Kneser-Ney implemented from the standard formulation
  (Chen & Goodman, 1999). Libraries: PyTorch, HuggingFace `datasets`, gensim, scikit-learn, NumPy, pandas, matplotlib.
- **AI assistance**: code scaffolding, debugging and the README were produced with the help of
  Claude Code (Anthropic). All results were generated by running this code; analysis in the report is my own.
