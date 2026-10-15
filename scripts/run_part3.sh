#!/usr/bin/env bash
# Generation samples for all LMs, Part III downstream classification (seed 42 + all LM seeds and
# untrained/bag-of-words controls), error analysis, cost benchmark, then tables/figures
set -euo pipefail
cd "$(dirname "$0")/.."
python -u -m src.generate --config configs/generate.yaml 2>&1 | grep --line-buffered -v Warning | tee logs/generate.log
python -u -m src.downstream --config configs/downstream.yaml 2>&1 | grep --line-buffered -v Warning | tee logs/downstream.log
python -u -m src.downstream_seeds --config configs/downstream.yaml 2>&1 | grep --line-buffered -v Warning | tee logs/downstream_seeds.log
python -u -m src.clean_fit 2>&1 | grep --line-buffered -v Warning | tee logs/clean_fit.log
python -u -m src.error_analysis > logs/error_analysis.log
python -u -m src.benchmark 2>&1 | grep --line-buffered -v Warning | tee logs/benchmark.log
python -u -m src.gen_benchmark --config configs/generate.yaml 2>&1 | grep --line-buffered -v Warning | tee logs/gen_benchmark.log
python -u -m src.lockstep_train 2>&1 | grep --line-buffered -v Warning | tee logs/lockstep_train.log   # ~25 min
python -m src.summarize
python -m src.latex_tables
