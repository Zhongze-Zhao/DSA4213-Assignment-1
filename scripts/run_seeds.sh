#!/usr/bin/env bash
# Optional: extra seeds for the three Part II LSTMs to quantify run-to-run variation
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 1 2; do
  for m in lstm lstm_w2v lstm_glove; do
    python -u -m src.train_lm --config configs/lm_$m.yaml --seed $s 2>&1 | grep --line-buffered -v Warning | tee logs/train_${m}_seed$s.log
  done
done
