# BTCBOT Progress Report

## Snapshot scope

This repository contains the earlier BTCBOT implementation and its historical research scripts. It is a source-and-documentation snapshot only. The later `v2` project is intentionally excluded because it is unfinished and remains in the local workspace.

## Version history

- Initial BTCBOT research began with market-data preparation, backtesting, benchmarking, and model experiments.
- Subsequent iterations explored entry timing, direction quality, regime filters, risk controls, trailing exits, and cost-sensitive validation.
- The root-level experiment scripts document the progression of those earlier investigations.

## Current state

- Legacy source is preserved across `backtest`, `benchmark`, `data`, `evaluation`, `execution`, `models`, and `training`.
- Configuration and data verification entry points are preserved at the project root.
- Large training datasets, runtime logs, caches, and generated artifacts are excluded from this public snapshot because they are reproducible or unsuitable for a source repository.
- The active unfinished V2 line remains local and is not archived or uploaded here.

## Reproducibility note

The scripts in this snapshot may refer to local datasets or historical paths that are not included. Treat this repository as a progress record and source baseline for the previous BTCBOT versions, not as a fully self-contained live-trading package.
