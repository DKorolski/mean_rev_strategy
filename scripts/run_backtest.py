import argparse
import datetime
import logging
from pathlib import Path

import backtrader as bt
import pandas as pd

from custom_data import CustomData
from data_prep import prepare_daily_info
from mean_rev_app import CombinedLongShortStrategy, load_config
from moex_parser2 import (
    moex_candles,
    moex_candles_index,
    moex_candles_option,
    moex_candles_stock,
)

DATA_LOADERS = {
    "moex_candles": moex_candles,
    "moex_candles_stock": moex_candles_stock,
    "moex_candles_index": moex_candles_index,
    "moex_candles_option": moex_candles_option,
}


def _normalize_strategy_config(strategy_cfg: dict) -> dict:
    normalized = dict(strategy_cfg)

    session_end_time = normalized.get("session_end_time")
    if isinstance(session_end_time, str):
        hours, minutes, seconds = map(int, session_end_time.split(":"))
        normalized["session_end_time"] = datetime.time(hours, minutes, seconds)

    exit_offset = normalized.get("exit_offset")
    if isinstance(exit_offset, (int, float)):
        normalized["exit_offset"] = datetime.timedelta(minutes=exit_offset)

    return normalized


def run_backtest(config_path: str) -> tuple[float, float]:
    config = load_config(config_path)

    broker_cfg = config["broker"]
    strategy_cfg = _normalize_strategy_config(config["strategy"])
    data_cfg = config["data"]

    loader_name = data_cfg.get("data_source", "moex_candles")
    loader = DATA_LOADERS.get(loader_name)
    if loader is None:
        supported = ", ".join(sorted(DATA_LOADERS))
        raise ValueError(f"Unsupported data_source '{loader_name}'. Use one of: {supported}")

    df = loader(
        data_cfg["ticker"],
        data_cfg["timeframe"],
        data_cfg["start_date"],
        data_cfg["end_date"],
    )
    df_prep = prepare_daily_info(df)

    cerebro = bt.Cerebro()
    cerebro.adddata(CustomData(dataname=df_prep))
    cerebro.addstrategy(CombinedLongShortStrategy, **strategy_cfg)
    cerebro.broker.setcash(broker_cfg["cash"])
    cerebro.broker.setcommission(commission=broker_cfg["commission"])

    start_value = float(cerebro.broker.getvalue())
    strategies = cerebro.run()
    end_value = float(cerebro.broker.getvalue())

    Path("reports/tmp").mkdir(parents=True, exist_ok=True)
    if strategies:
        trade_journal = getattr(strategies[0], "trade_journal", [])
        pd.DataFrame(trade_journal).to_csv("reports/tmp/trade_journal_latest.csv", index=False)

    return start_value, end_value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MOEX mean-reversion backtest.")
    parser.add_argument(
        "--config",
        default="configs/run_c.json",
        help="Path to JSON config file (default: configs/run_c.json).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    start_value, end_value = run_backtest(args.config)

    print(f"Start portfolio value: {start_value:.2f}")
    print(f"End portfolio value:   {end_value:.2f}")


if __name__ == "__main__":
    main()
