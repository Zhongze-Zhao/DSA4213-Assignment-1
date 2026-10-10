#!/usr/bin/env bash
# Reproduce every result in the report (~3 h on an RTX 5070 Ti Laptop GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data
python -m src.models          # causality check
scripts/run_part1.sh
scripts/run_part2.sh
python -u -m src.train_lm --config configs/lm_lstm_smallinit.yaml   # supplementary run
scripts/run_part3.sh
scripts/run_seeds.sh && scripts/run_seeds_part1.sh && python -m src.summarize && python -m src.latex_tables
