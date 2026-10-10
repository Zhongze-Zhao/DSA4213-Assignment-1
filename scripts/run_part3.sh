#!/usr/bin/env bash
# Generation samples for all LMs, then Part III downstream classification, then tables/figures
set -euo pipefail
cd "$(dirname "$0")/.."
python -u -m src.generate --config configs/generate.yaml 2>&1 | grep --line-buffered -v Warning | tee logs/generate.log
python -u -m src.downstream --config configs/downstream.yaml 2>&1 | grep --line-buffered -v Warning | tee logs/downstream.log
python -u -m src.error_analysis > logs/error_analysis.log
python -u -m src.benchmark 2>&1 | grep --line-buffered -v Warning | tee logs/benchmark.log
python -m src.summarize
python -m src.latex_tables
