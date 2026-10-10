#!/usr/bin/env bash
# Part I: n-gram + RNN/LSTM/Transformer (neural runs sequential so GPU timings are clean)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python -u -m src.ngram --config configs/ngram.yaml | tee logs/ngram.log
for m in rnn lstm transformer; do
  python -u -m src.train_lm --config configs/lm_$m.yaml 2>&1 | grep --line-buffered -v Warning | tee logs/train_$m.log
done
