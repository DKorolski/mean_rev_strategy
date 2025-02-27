import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from moex_parser2 import *
from functools import partial
from multiprocessing import Pool

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
    На каждом баре вычисляем:
      - Сигнал лонга (по своим параметрам)
      - Сигнал шорта (по своим параметрам)
    Если нет позиции:
      - Если сигнал лонга == 1 => открываем лонг (Position=+1)
      - ИНАЧЕ если сигнал шорта == 1 => открываем шорт (Position=-1)
    Если есть позиция:
      - Если лонг => проверяем выход лонга (тейк, стоп, конец дня)
      - Если шорт => проверяем выход шорта (тейк, стоп, конец дня)
    
    ПАРАМЕТРЫ:
      - MinRangeLong, MaxRangeLong, KLong, takeKLong, stopKLong, commLong
      - MinRangeShort, MaxRangeShort, KShort, takeKShort, stopKShort, commShort
    """

    df_res = df.copy()
    df_res = df_res.sort_index()
    df_res.dropna(subset=['Close_prev','DayRangePrev'], inplace=True)
    
    # --- 1) СЧИТАЕМ ТРИГГЕРЫ ДЛЯ ЛОНГА ---
    # (пример: покупка "на откате", как ранее)
    df_res['RelDayRange'] = df_res['DayRangePrev'] / df_res['Close']  # для обоих
    df_res['TimeFilterLong'] = (df_res.index.hour < 12)

    df_res['Trigger1Long'] = (df_res['RelDayRange'] > MinRangeLong) & (df_res['RelDayRange'] < MaxRangeLong)
    df_res['Trigger2Long'] = df_res['Close'] < df_res['Close_prev'] 
    df_res['Trigger3Long'] = df_res['Close'] > (df_res['Close_prev'] - KLong*df_res['DayRangePrev'])

    df_res['LongSignal'] = (
        df_res['TimeFilterLong'] &
        df_res['Trigger1Long'] &
        df_res['Trigger2Long'] &
        df_res['Trigger3Long']
    ).astype(int)

    # --- 2) СЧИТАЕМ ТРИГГЕРЫ ДЛЯ ШОРТА ---
    # (пример: продажа, как ранее)
    df_res['TimeFilterShort'] = (df_res.index.hour < 12)
    df_res['Trigger1Short'] = (df_res['RelDayRange'] > MinRangeShort) & (df_res['RelDayRange'] < MaxRangeShort)
    df_res['Trigger2Short'] = df_res['Close'] > df_res['Close_prev']
    df_res['Trigger3Short'] = df_res['Close'] < (df_res['Close_prev'] + KShort*df_res['DayRangePrev'])

    df_res['ShortSignal'] = (
        df_res['TimeFilterShort'] &
        df_res['Trigger1Short'] &
        df_res['Trigger2Short'] &
        df_res['Trigger3Short']
    ).astype(int)

    df_res['Position'] = 0
    df_res['TradeEntryPrice'] = np.nan
    df_res['TradeExitPrice'] = np.nan
    df_res['TradePnL'] = 0.0

    # Для удобства
    df_res['date'] = df_res.index.date
    close_vals = df_res['Close'].values
    dayrange_vals = df_res['DayRangePrev'].values
    long_signal_vals = df_res['LongSignal'].values
    short_signal_vals = df_res['ShortSignal'].values
    date_vals = df_res['date'].values

    n = len(df_res)
    position_list = [0]*n
    entry_price_list = [np.nan]*n
    exit_price_list = [np.nan]*n
    pnl_list = [0.0]*n

    position = 0    # 0 - flat, +1 - лонг, -1 - шорт
    entry_price = np.nan
    current_day = None

    for i in range(n):
        current_close = close_vals[i]
        drange = dayrange_vals[i]

        if position == 0:
            # Нет позиции
            # 1) Проверяем лонг-сигнал
            if long_signal_vals[i] == 1:
                position = +1
                entry_price = current_close
                current_day = date_vals[i]
                position_list[i] = +1
                entry_price_list[i] = entry_price

            # 2) ИНАЧЕ, проверяем шорт-сигнал
            elif short_signal_vals[i] == 1:
                position = -1
                entry_price = current_close
                current_day = date_vals[i]
                position_list[i] = -1
                entry_price_list[i] = entry_price

            # иначе остаёмся во флэте
            else:
                position_list[i] = 0

        elif position == +1:
            # Лонг
            position_list[i] = +1
            # Уровни тейка/стопа для лонга
            take_price_long = entry_price + takeKLong * drange
            stop_price_long = entry_price - stopKLong * drange

            # 1) Тейк, если Close >= take_price_long
            if current_close >= take_price_long:
                exit_price_list[i] = current_close
                comm_cost = commLong * (entry_price + current_close)
                pnl_list[i] = (current_close - entry_price) - comm_cost
                position = 0
                entry_price = np.nan
                current_day = None

            # 2) Стоп, если Close <= stop_price_long
            elif current_close <= stop_price_long:
                exit_price_list[i] = current_close
                comm_cost = commLong * (entry_price + current_close)
                pnl_list[i] = (current_close - entry_price) - comm_cost
                position = 0
                entry_price = np.nan
                current_day = None

            else:
                # 3) Конец дня
                # Если это последний бар или следующий бар - следующий день
                if i == n-1:
                    exit_price_list[i] = current_close
                    comm_cost = commLong * (entry_price + current_close)
                    pnl_list[i] = (current_close - entry_price) - comm_cost
                    position = 0
                    entry_price = np.nan
                    current_day = None
                else:
                    if date_vals[i+1] != current_day:
                        exit_price_list[i] = current_close
                        comm_cost = commLong * (entry_price + current_close)
                        pnl_list[i] = (current_close - entry_price) - comm_cost
                        position = 0
                        entry_price = np.nan
                        current_day = None

        else:
            # position == -1 => Шорт
            position_list[i] = -1
            # Уровни тейка/стопа для шорта
            take_price_short = entry_price - takeKShort * drange
            stop_price_short = entry_price + stopKShort * drange

            # 1) Тейк, если Close <= take_price_short
            if current_close <= take_price_short:
                exit_price_list[i] = current_close
                comm_cost = commShort * (entry_price + current_close)
                # шорт PnL = (entry - exit)
                pnl_list[i] = (entry_price - current_close) - comm_cost
                position = 0
                entry_price = np.nan
                current_day = None

            # 2) Стоп, если Close >= stop_price_short
            elif current_close >= stop_price_short:
                exit_price_list[i] = current_close
                comm_cost = commShort * (entry_price + current_close)
                pnl_list[i] = (entry_price - current_close) - comm_cost
                position = 0
                entry_price = np.nan
                current_day = None

            else:
                # 3) Конец дня
                if i == n-1:
                    exit_price_list[i] = current_close
                    comm_cost = commShort * (entry_price + current_close)
                    pnl_list[i] = (entry_price - current_close) - comm_cost
                    position = 0
                    entry_price = np.nan
                    current_day = None
                else:
                    if date_vals[i+1] != current_day:
                        exit_price_list[i] = current_close
                        comm_cost = commShort * (entry_price + current_close)
                        pnl_list[i] = (entry_price - current_close) - comm_cost
                        position = 0
                        entry_price = np.nan
                        current_day = None

    # Записываем результаты
    df_res['Position'] = position_list
    df_res['TradeEntryPrice'] = entry_price_list
    df_res['TradeExitPrice'] = exit_price_list
    df_res['TradePnL'] = pnl_list

    return df_res


   

if __name__ == "__main__":
    # Загрузка и подготовка данных
    df2 = moex_candles('IMOEXF','10','2020-11-30','2025-10-18')
    df = df2[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        #data0=ticker0.dropna()
    df.index.names = ['Datetime']
    print(df)

    # Подготовка данных (расчёт Close_prev и DayRangePrev)
    df_prep = prepare_daily_info(df)

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
    print("Combined Strategy Stats:\n", stats_result)

    # Рисуем общую кривую эквити
    plot_equity_curve(df_combined, title="Combined Long+Short Strategy")
    