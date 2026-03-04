import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from moex_parser2 import moex_candles

def worker(num):
    """Функция, выполняющая задачу в процессе."""
    print(f"Worker {num} начал работу")

def plot_equity_curve(df_res, title="Equity Curve"):
    """
    Строим график кумулятивной прибыли (equity) в разрезе сделок (trade #).
    df_res — DataFrame после apply_strategy.
    """
    closed_trades = df_res[df_res['TradePnL'] != 0].copy()
    if len(closed_trades) == 0:
        print("Нет закрытых сделок — график эквити невозможен")
        return

    closed_trades.sort_index(inplace=True)
    closed_trades['cumPnL'] = closed_trades['TradePnL'].cumsum()
    closed_trades.reset_index(drop=True, inplace=True)
    closed_trades['TradeNumber'] = closed_trades.index + 1

    plt.figure(figsize=(10, 5))
    plt.plot(closed_trades['TradeNumber'], closed_trades['cumPnL'], marker='o')
    plt.title(title)
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.show()

def prepare_daily_info(intraday_df):
    """
    На вход подаётся DataFrame с 10-минутными барами (индекс Datetime).
    На выходе — DataFrame тех же баров с добавленными колонками:
      - 'Close_prev': закрытие предыдущего дня
      - 'DayRangePrev': диапазон предыдущего дня (High_prev - Low_prev)
    """
    df_intra = intraday_df.copy()
    df_intra['date'] = df_intra.index.date

    daily_info = (
        df_intra
        .groupby('date', as_index=False)
        .agg({
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        .rename(columns={
            'High': 'High_day',
            'Low': 'Low_day',
            'Close': 'Close_day'
        })
    )
    daily_info['High_prev'] = daily_info['High_day'].shift(1)
    daily_info['Low_prev'] = daily_info['Low_day'].shift(1)
    daily_info['Close_prev'] = daily_info['Close_day'].shift(1)
    daily_info['DayRangePrev'] = daily_info['High_prev'] - daily_info['Low_prev']

    df_merged = pd.merge(
        df_intra.reset_index(),  # каждая свеча с отдельным временем
        daily_info[['date', 'Close_prev', 'DayRangePrev']],
        on='date',
        how='left'
    )
    # Протягиваем значения до конца дня (при необходимости)
    df_merged[['Close_prev', 'DayRangePrev']] = df_merged[['Close_prev', 'DayRangePrev']].ffill()
    df_merged.set_index('Datetime', inplace=True)
    df_merged.drop('date', axis=1, inplace=True)
    return df_merged

def max_consecutive_runs(bool_array, value=True):
    """
    Считает максимальное количество подряд идущих значений (True или False)
    в булевом массиве.
    """
    max_streak = 0
    current_streak = 0
    for x in bool_array:
        if x == value:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak

def strategy_statistics(df_res):
    """
    Принимает DataFrame после apply_strategy (где есть столбец 'TradePnL')
    и возвращает словарь со сводными метриками, включая максимальное количество
    подряд выигрышных и проигрышных сделок.
    """
    closed_trades = df_res[df_res['TradePnL'] != 0].copy()
    n_trades = len(closed_trades)
    if n_trades == 0:
        return {
            'n_trades': 0,
            'total_pnl': 0.0,
            'mean_pnl': 0.0,
            'win_rate': 0.0,
            't_stat': 0.0,
            't_pvalue': 1.0,
            'sharpe': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0
        }

    total_pnl = closed_trades['TradePnL'].sum()
    mean_pnl = closed_trades['TradePnL'].mean()
    win_rate = np.mean(closed_trades['TradePnL'] > 0)
    t_stat, t_pvalue = stats.ttest_1samp(closed_trades['TradePnL'], 0)
    std_pnl = closed_trades['TradePnL'].std()
    sharpe = mean_pnl / std_pnl if std_pnl != 0 else np.nan

    closed_trades['winning'] = closed_trades['TradePnL'] > 0
    max_consecutive_wins = max_consecutive_runs(closed_trades['winning'].values, True)
    max_consecutive_losses = max_consecutive_runs(closed_trades['winning'].values, False)

    return {
        'n_trades': n_trades,
        'total_pnl': total_pnl,
        'mean_pnl': mean_pnl,
        'win_rate': win_rate,
        't_stat': t_stat,
        't_pvalue': t_pvalue,
        'sharpe': sharpe,
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses
    }

def apply_combined_long_short(
    df,
    # ПАРАМЕТРЫ ДЛЯ ЛОНГА
    MinRangeLong=0.01,
    MaxRangeLong=0.05,
    KLong=0.05,
    takeKLong=0.2,
    stopKLong=0.1,
    commLong=0.00017,
    # ПАРАМЕТРЫ ДЛЯ ШОРТА
    MinRangeShort=0.01,
    MaxRangeShort=0.05,
    KShort=0.05,
    takeKShort=0.2,
    stopKShort=0.1,
    commShort=0.00017
):
    """
    ЕДИНАЯ функция, совмещающая лонг и шорт.
    Вычисляются сигналы и производится открытие/закрытие позиции.
    Дополнительно журналируются сделки с EntryDatetime и ExitDatetime.
    """
    df_res = df.copy()
    df_res = df_res.sort_index()
    df_res.dropna(subset=['Close_prev', 'DayRangePrev'], inplace=True)
    
    # --- 1) Считаем триггеры для лонга ---
    df_res['RelDayRange'] = df_res['DayRangePrev'] / df_res['Close']
    df_res['TimeFilterLong'] = (df_res.index.hour < 12)
    df_res['Trigger1Long'] = (df_res['RelDayRange'] > MinRangeLong) & (df_res['RelDayRange'] < MaxRangeLong)
    df_res['Trigger2Long'] = df_res['Close'] < df_res['Close_prev']
    df_res['Trigger3Long'] = df_res['Close'] > (df_res['Close_prev'] - KLong * df_res['DayRangePrev'])
    df_res['LongSignal'] = (
        df_res['TimeFilterLong'] &
        df_res['Trigger1Long'] &
        df_res['Trigger2Long'] &
        df_res['Trigger3Long']
    ).astype(int)
    
    # --- 2) Считаем триггеры для шорта ---
    df_res['TimeFilterShort'] = (df_res.index.hour < 12)
    df_res['Trigger1Short'] = (df_res['RelDayRange'] > MinRangeShort) & (df_res['RelDayRange'] < MaxRangeShort)
    df_res['Trigger2Short'] = df_res['Close'] > df_res['Close_prev']
    df_res['Trigger3Short'] = df_res['Close'] < (df_res['Close_prev'] + KShort * df_res['DayRangePrev'])
    df_res['ShortSignal'] = (
        df_res['TimeFilterShort'] &
        df_res['Trigger1Short'] &
        df_res['Trigger2Short'] &
        df_res['Trigger3Short']
    ).astype(int)
    
    # Инициализируем колонки для результатов
    df_res['Position'] = 0
    df_res['TradeEntryPrice'] = np.nan
    df_res['TradeExitPrice'] = np.nan
    df_res['TradePnL'] = 0.0
    df_res['date'] = df_res.index.date
    
    n = len(df_res)
    position_list = [0] * n
    entry_price_list = [np.nan] * n
    exit_price_list = [np.nan] * n
    pnl_list = [0.0] * n

    # Список для журналирования сделок
    trade_journal = []

    close_vals = df_res['Close'].values
    dayrange_vals = df_res['DayRangePrev'].values
    long_signal_vals = df_res['LongSignal'].values
    short_signal_vals = df_res['ShortSignal'].values
    date_vals = df_res['date'].values

    position = 0    # 0 - флэт, +1 - лонг, -1 - шорт
    entry_price = np.nan
    current_day = None
    entry_datetime = None  # сохраняем время входа

    for i in range(n):
        current_close = close_vals[i]
        drange = dayrange_vals[i]

        if position == 0:
            # Нет позиции — ищем сигнал на вход
            if long_signal_vals[i] == 1:
                position = +1
                entry_price = current_close
                current_day = date_vals[i]
                entry_datetime = df_res.index[i]
                position_list[i] = +1
                entry_price_list[i] = entry_price
            elif short_signal_vals[i] == 1:
                position = -1
                entry_price = current_close
                current_day = date_vals[i]
                entry_datetime = df_res.index[i]
                position_list[i] = -1
                entry_price_list[i] = entry_price
            else:
                position_list[i] = 0

        elif position == +1:
            # Обработка лонг-позиции
            position_list[i] = +1
            take_price_long = entry_price + takeKLong * drange
            stop_price_long = entry_price - stopKLong * drange

            if current_close >= take_price_long:
                exit_price_list[i] = current_close
                comm_cost = commLong * (entry_price + current_close)
                pnl = (current_close - entry_price) - comm_cost
                pnl_list[i] = pnl
                trade_journal.append({
                    'EntryDatetime': entry_datetime,
                    'ExitDatetime': df_res.index[i],
                    'TradeType': 'Long',
                    'EntryPrice': entry_price,
                    'ExitPrice': current_close,
                    'TradePnL': pnl,
                    'EntryDate': current_day
                })
                position = 0
                entry_price = np.nan
                current_day = None
                entry_datetime = None
            elif current_close <= stop_price_long:
                exit_price_list[i] = current_close
                comm_cost = commLong * (entry_price + current_close)
                pnl = (current_close - entry_price) - comm_cost
                pnl_list[i] = pnl
                trade_journal.append({
                    'EntryDatetime': entry_datetime,
                    'ExitDatetime': df_res.index[i],
                    'TradeType': 'Long',
                    'EntryPrice': entry_price,
                    'ExitPrice': current_close,
                    'TradePnL': pnl,
                    'EntryDate': current_day
                })
                position = 0
                entry_price = np.nan
                current_day = None
                entry_datetime = None
            else:
                if i == n - 1 or date_vals[i+1] != current_day:
                    exit_price_list[i] = current_close
                    comm_cost = commLong * (entry_price + current_close)
                    pnl = (current_close - entry_price) - comm_cost
                    pnl_list[i] = pnl
                    trade_journal.append({
                        'EntryDatetime': entry_datetime,
                        'ExitDatetime': df_res.index[i],
                        'TradeType': 'Long',
                        'EntryPrice': entry_price,
                        'ExitPrice': current_close,
                        'TradePnL': pnl,
                        'EntryDate': current_day
                    })
                    position = 0
                    entry_price = np.nan
                    current_day = None
                    entry_datetime = None

        else:
            # Обработка шорт-позиции (position == -1)
            position_list[i] = -1
            take_price_short = entry_price - takeKShort * drange
            stop_price_short = entry_price + stopKShort * drange

            if current_close <= take_price_short:
                exit_price_list[i] = current_close
                comm_cost = commShort * (entry_price + current_close)
                pnl = (entry_price - current_close) - comm_cost
                pnl_list[i] = pnl
                trade_journal.append({
                    'EntryDatetime': entry_datetime,
                    'ExitDatetime': df_res.index[i],
                    'TradeType': 'Short',
                    'EntryPrice': entry_price,
                    'ExitPrice': current_close,
                    'TradePnL': pnl,
                    'EntryDate': current_day
                })
                position = 0
                entry_price = np.nan
                current_day = None
                entry_datetime = None
            elif current_close >= stop_price_short:
                exit_price_list[i] = current_close
                comm_cost = commShort * (entry_price + current_close)
                pnl = (entry_price - current_close) - comm_cost
                pnl_list[i] = pnl
                trade_journal.append({
                    'EntryDatetime': entry_datetime,
                    'ExitDatetime': df_res.index[i],
                    'TradeType': 'Short',
                    'EntryPrice': entry_price,
                    'ExitPrice': current_close,
                    'TradePnL': pnl,
                    'EntryDate': current_day
                })
                position = 0
                entry_price = np.nan
                current_day = None
                entry_datetime = None
            else:
                if i == n - 1 or date_vals[i+1] != current_day:
                    exit_price_list[i] = current_close
                    comm_cost = commShort * (entry_price + current_close)
                    pnl = (entry_price - current_close) - comm_cost
                    pnl_list[i] = pnl
                    trade_journal.append({
                        'EntryDatetime': entry_datetime,
                        'ExitDatetime': df_res.index[i],
                        'TradeType': 'Short',
                        'EntryPrice': entry_price,
                        'ExitPrice': current_close,
                        'TradePnL': pnl,
                        'EntryDate': current_day
                    })
                    position = 0
                    entry_price = np.nan
                    current_day = None
                    entry_datetime = None

    df_res['Position'] = position_list
    df_res['TradeEntryPrice'] = entry_price_list
    df_res['TradeExitPrice'] = exit_price_list
    df_res['TradePnL'] = pnl_list

    # Сохранение журнала сделок в CSV
    journal_df = pd.DataFrame(trade_journal)
    journal_df.to_csv("trade_journal.csv", index=False)
    print("Журнал сделок сохранён в файл: trade_journal.csv")

    return df_res


def save_trade_journal(df_res, filename="trade_journal.csv"):
    """
    Сохраняет журнал закрытых сделок в CSV.
    """
    journal = df_res[df_res['TradePnL'] != 0].copy()
    journal.to_csv(filename, index=True)
    print(f"Журнал сделок сохранён в файл: {filename}")

def plot_trades_and_equity(df_res, price_title="График цены с сделками", equity_title="Equity Curve"):
    """
    Строит два графика:
      1. График цены (Close) с отметками точек входа (зеленые треугольники) и выхода (красные треугольники вниз)
      2. График кумулятивной прибыли (equity curve) по закрытым сделкам
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # График цены с отметками сделок
    ax1.plot(df_res.index, df_res['Close'], label='Цена', color='blue')
    entry_df = df_res[~df_res['TradeEntryPrice'].isna()]
    exit_df = df_res[~df_res['TradeExitPrice'].isna()]
    ax1.scatter(entry_df.index, entry_df['TradeEntryPrice'], marker='^', color='green', s=100, label='Вход')
    ax1.scatter(exit_df.index, exit_df['TradeExitPrice'], marker='v', color='red', s=100, label='Выход')
    ax1.set_title(price_title)
    ax1.set_xlabel("Дата")
    ax1.set_ylabel("Цена")
    ax1.legend()
    ax1.grid(True)

    # График equity curve
    closed_trades = df_res[df_res['TradePnL'] != 0].copy()
    if len(closed_trades) > 0:
        closed_trades.sort_index(inplace=True)
        closed_trades['cumPnL'] = closed_trades['TradePnL'].cumsum()
        closed_trades.reset_index(drop=True, inplace=True)
        closed_trades['TradeNumber'] = closed_trades.index + 1
        ax2.plot(closed_trades['TradeNumber'], closed_trades['cumPnL'], marker='o', label='Equity', color='purple')
        ax2.set_title(equity_title)
        ax2.set_xlabel("Номер сделки")
        ax2.set_ylabel("Кумулятивный PnL")
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, "Нет закрытых сделок", horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Загрузка и подготовка данных
    df2 = moex_candles('IMOEXF', '10', '2020-11-30', '2025-10-18')
    df = df2[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index.names = ['Datetime']
    print(df)

    # Подготовка данных (расчёт Close_prev и DayRangePrev)
    df_prep = prepare_daily_info(df)

    # Применение комбинированной стратегии
    df_combined = apply_combined_long_short(df_prep, 
                                            # ЛОНГ параметры
                                            MinRangeLong=0.022,
                                            MaxRangeLong=0.045,
                                            KLong=0.045,
                                            takeKLong=0.12,
                                            stopKLong=0.48,
                                            commLong=0.00017,
                                            # ШОРТ параметры
                                            MinRangeShort=0.018,
                                            MaxRangeShort=0.05,
                                            KShort=0.06,
                                            takeKShort=0.17,
                                            stopKShort=0.5,
                                            commShort=0.00017
                                            )

    # Рассчитываем статистику
    stats_result = strategy_statistics(df_combined)
    print("Статистика стратегии:\n", stats_result)

    # Сохраняем журнал сделок в CSV
    save_trade_journal(df_combined, filename="trade_journal.csv")

    # Рисуем график цены с отметками сделок и график equity
    plot_trades_and_equity(df_combined, price_title="График цены с отметками сделок", equity_title="Equity Curve")
