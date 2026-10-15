#!/usr/bin/env bash
# Reproduce every result in the report (~4.5 h on an RTX 5070 Ti Laptop GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data
python -m src.models          # causality check
scripts/run_part1.sh
scripts/run_part2.sh
python -u -m src.train_lm --config configs/lm_lstm_smallinit.yaml   # supplementary run
scripts/run_seeds.sh          # seeds 1, 2 for the three Part II LSTMs
scripts/run_seeds_part1.sh    # seeds 1, 2 for the Part I RNN and Transformer
scripts/run_part3.sh          # needs all checkpoints above
