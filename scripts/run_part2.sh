#!/usr/bin/env bash
# Part II: own Skip-gram + GloVe tables, then two frozen-embedding LSTMs.
# Setting 1 (random, trainable) reuses the Part I LSTM (identical config/procedure).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python -u -m src.embeddings --config configs/embeddings.yaml 2>&1 | grep --line-buffered -v Warning | tee logs/embeddings.log
for m in lstm_w2v lstm_glove; do
  python -u -m src.train_lm --config configs/lm_$m.yaml 2>&1 | grep --line-buffered -v Warning | tee logs/train_$m.log
done
