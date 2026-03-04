# Run C Summary

## Run Setup

- Instrument: `IMOEXF`
- Timeframe: `10m`
- Period: `2020-11-30` to `2025-10-18`
- Initial capital: `100,000`
- Commission: `0.017%` per trade (`0.00017`)
- Slippage: not explicitly modeled in this run

## Key Metrics

Extracted from `reports/run_c/stats_mean_rev.html`:

- Cumulative Return: `38.55%`
- CAGR: `19.33%`
- Sharpe: `4.44`
- Sortino: `9.02`
- Max Drawdown: `-3.1%`
- Volatility (ann.): `5.73%`
- Calmar: `6.23`

## Artifacts

- Equity curve (main): `reports/equity_run_c.png`
- Full QuantStats report: `reports/run_c/stats_mean_rev.html`
- Trade journal snapshot: `reports/run_c/trade_journal.csv`
- Extra charts: `reports/run_c/model_equity.png`, `reports/run_c/pre_prod_test.png`

## Notes

- This repository is a research showcase, not production trading software.
- Metrics are in-sample for the specified period and may degrade out-of-sample.
