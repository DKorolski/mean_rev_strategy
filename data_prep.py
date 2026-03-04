# подготовка дневных данных
import pandas as pd


def prepare_daily_info(intraday_df):
    """
    На вход подаётся DataFrame с 10-минутными барами (индекс Datetime).
    На выходе — DataFrame тех же баров с добавленными колонками:
      - 'Close_prev': закрытие предыдущего дня
      - 'DayRangePrev': диапазон предыдущего дня (High_prev - Low_prev)
    """
    intraday_df = intraday_df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    intraday_df.index.names = ["datetime"]
    intraday_df.columns = ["open", "high", "low", "close", "volume"]
    df_intra = intraday_df.copy()
    df_intra["date"] = df_intra.index.date

    daily_info = (
        df_intra.groupby("date", as_index=False)
        .agg({"high": "max", "low": "min", "close": "last"})
        .rename(columns={"high": "high_day", "low": "low_day", "close": "close_day"})
    )
    daily_info["high_prev"] = daily_info["high_day"].shift(1)
    daily_info["low_prev"] = daily_info["low_day"].shift(1)
    daily_info["close_prev"] = daily_info["close_day"].shift(1)
    daily_info["dayrangeprev"] = daily_info["high_prev"] - daily_info["low_prev"]

    df_merged = pd.merge(
        df_intra.reset_index(),  # каждая свеча с отдельным временем
        daily_info[["date", "close_prev", "dayrangeprev"]],
        on="date",
        how="left",
    )
    # Протягиваем значения до конца дня (при необходимости)
    df_merged[["close_prev", "dayrangeprev"]] = df_merged[["close_prev", "dayrangeprev"]].ffill()
    df_merged.set_index("datetime", inplace=True)
    df_merged.drop("date", axis=1, inplace=True)
    return df_merged
