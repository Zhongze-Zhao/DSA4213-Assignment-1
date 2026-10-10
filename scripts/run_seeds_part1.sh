#!/usr/bin/env bash
# Extra seeds for the Part I RNN and Transformer (LSTM seeds come from run_seeds.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 1 2; do
  for m in rnn transformer; do
    python -u -m src.train_lm --config configs/lm_$m.yaml --seed $s 2>&1 | grep --line-buffered -v Warning > logs/train_${m}_seed$s.log
  done
done
echo "part1 seeds done $(date)"
