import asyncio
from typing import Any

import aiohttp
import aiomoex
import pandas as pd


def _normalize_candles_df(data: list[dict[str, Any]]) -> pd.DataFrame:
    if not data:
        empty = pd.DataFrame(columns=["Open", "Close", "High", "Low", "Volume", "Adj Close"])
        empty.index = pd.DatetimeIndex([], name="Date")
        return empty

    df = pd.DataFrame(data)
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.set_index("begin").sort_index()
    df.index.name = "Date"
    df = df.rename(
        columns={
            "open": "Open",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "volume": "Volume",
        }
    )
    df = df[["Open", "Close", "High", "Low", "Volume"]].copy()
    df["Adj Close"] = df["Close"]
    return df


async def _fetch_board_candles(
    security: str,
    interval: str,
    start: str,
    end: str,
    **kwargs: Any,
) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        data = await aiomoex.get_board_candles(
            session=session,
            security=security,
            interval=interval,
            start=start,
            end=end,
            **kwargs,
        )
    return _normalize_candles_df(data)


def moex_candles(security, interval, start, end):
    return asyncio.run(
        _fetch_board_candles(
            security=security,
            interval=interval,
            start=start,
            end=end,
            engine="futures",
            market="forts",
            board="TQBR",
        )
    )


def moex_candles_stock(security, interval, start, end):
    return asyncio.run(
        _fetch_board_candles(
            security=security,
            interval=interval,
            start=start,
            end=end,
            board="TQBR",
        )
    )


def candles_resample(df, interval):
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df_resampled = df.resample(interval).agg(
        {"Open": "first", "Close": "last", "High": "max", "Low": "min", "Volume": "sum"}
    )
    # Drop rows with all NaN values if any
    df_resampled.dropna(how="all", inplace=True)
    return df_resampled


def moex_candles_index(security, interval, start, end):
    return asyncio.run(
        _fetch_board_candles(
            security=security,
            interval=interval,
            start=start,
            end=end,
            engine="stock",
            market="index",
        )
    )


def moex_candles_option(security, interval, start, end):
    return asyncio.run(
        _fetch_board_candles(
            security=security,
            interval=interval,
            start=start,
            end=end,
            engine="futures",
            market="options",
        )
    )
