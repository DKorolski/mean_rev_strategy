import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from moex_parser2 import *
from itertools import product
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

def apply_strategy(df, 
                   MinRange=0.01, 
                   MaxRange=0.05, 
                   K=0.05, 
                   holding_bars=15,
                   commission=0.0):
    """
    Применяет стратегию на основе переданных параметров.
    df — 10-минутный DataFrame (после prepare_daily_info) с колонками 'DayRangePrev', 'Close_prev', 'Close'.
    holding_bars — количество баров для удержания сделки.
    commission — комиссия брокера (доля, например, 0.001 означает 0.1%).
    
    Возвращает DataFrame с рассчитанными сигналами, ценами входа/выхода и PnL.
    """
    df_res = df.copy()
    df_res.dropna(subset=['Close_prev', 'DayRangePrev'], inplace=True)

    # 1) Расчёт относительного диапазона дня
    df_res['RelDayRange'] = df_res['DayRangePrev'] / df_res['Close']
    # 2) Отбор баров до 12:00
    df_res['TimeFilter'] = (df_res.index.hour < 12)
    # 3) Триггер по диапазону
    df_res['Trigger1'] = (df_res['RelDayRange'] > MinRange) & (df_res['RelDayRange'] < MaxRange)
    # 4) Триггер по сравнению закрытия с предыдущим днем
    df_res['Trigger2'] = df_res['Close'] > df_res['Close_prev']
    # 5) Триггер по цене (с поправкой на K)
    df_res['Trigger3'] = df_res['Close'] < (df_res['Close_prev'] + K * df_res['DayRangePrev'])

    df_res['EntrySignal'] = (
        df_res['TimeFilter'] &
        df_res['Trigger1'] &
        df_res['Trigger2'] &
        df_res['Trigger3']
    ).astype(int)

    df_res['EntryPrice'] = np.nan
    df_res.loc[df_res['EntrySignal'] == 1, 'EntryPrice'] = df_res['Close']

    n = len(df_res)
    position = [0] * n         # 0 — вне позиции, 1 — в позиции (шорт)
    bars_in_position = [0] * n   # сколько баров удерживается позиция на каждом баре
    entry_price = [np.nan] * n
    exit_price = [np.nan] * n
    pnl = [0] * n

    in_pos = False
    bars_held = 0
    current_entry_price = np.nan
    close_values = df_res['Close'].values

    for i, row in enumerate(df_res.itertuples()):
        if not in_pos:
            if row.EntrySignal == 1:
                in_pos = True
                bars_held = 0
                current_entry_price = close_values[i]
                position[i] = 1
                entry_price[i] = current_entry_price
                bars_in_position[i] = bars_held
        else:
            position[i] = 1
            bars_held += 1
            bars_in_position[i] = bars_held
            if bars_held >= holding_bars:
                exit_price[i] = close_values[i]
                # Расчёт PnL с учётом комиссии
                # Комиссия списывается и при входе, и при выходе:
                commission_cost = commission * (current_entry_price + close_values[i])
                pnl[i] = (current_entry_price - close_values[i]) - commission_cost
                in_pos = False
                bars_held = 0
                current_entry_price = np.nan

    df_res['Position'] = position
    df_res['BarsHeld'] = bars_in_position
    df_res['TradeEntryPrice'] = entry_price
    df_res['TradeExitPrice'] = exit_price
    df_res['TradePnL'] = pnl

    return df_res

def process_params(params, df_prep):
    """
    Обрабатывает одну комбинацию параметров.
    Если условие (MinRange < MaxRange) не выполняется, возвращается None.
    Теперь параметры включают также commission.
    """
    min_r, max_r, k_, hb, comm = params
    if min_r >= max_r:
        return None

    df_signals = apply_strategy(df_prep, 
                                MinRange=min_r, 
                                MaxRange=max_r, 
                                K=k_, 
                                holding_bars=hb,
                                commission=comm)
    stats_result = strategy_statistics(df_signals)
    row = {
        'MinRange': min_r,
        'MaxRange': max_r,
        'K': k_,
        'holding_bars': hb,
        'commission': comm,
        'n_trades': stats_result['n_trades'],
        'total_pnl': stats_result['total_pnl'],
        'mean_pnl': stats_result['mean_pnl'],
        'win_rate': stats_result['win_rate'],
        't_stat': stats_result['t_stat'],
        't_pvalue': stats_result['t_pvalue'],
        'sharpe': stats_result['sharpe'],
        'max_consecutive_wins': stats_result['max_consecutive_wins'],
        'max_consecutive_losses': stats_result['max_consecutive_losses']
    }
    return row

def optimize_parameters_mp(df_prep,
                           min_range_values=[0.01, 0.015, 0.02],
                           max_range_values=[0.03, 0.04, 0.05],
                           k_values=[0.02, 0.04, 0.06],
                           holding_bars_values=[10, 15, 20],
                           commission_values=[0.0, 0.0005, 0.001],
                           processes=4):
    """
    Распараллеленная оптимизация параметров.
    Теперь перебор включает параметр commission.
    """
    param_combinations = list(product(min_range_values, 
                                      max_range_values, 
                                      k_values, 
                                      holding_bars_values,
                                      commission_values))
    func = partial(process_params, df_prep=df_prep)

    with Pool(processes=processes) as pool:
        results = pool.map(func, param_combinations)

    results = [r for r in results if r is not None]
    return pd.DataFrame(results)

def analyze_short_price_behavior(df_res, 
                                 holding_bars=15,
                                 bars_forward=6):
    """
    Для шорт-сделок (от apply_strategy), смотрит:
      - На 6 баров вперёд (bars_forward)
      - До тайм-аута (holding_bars)
      - До конца дня
    И считает:
      - fraction_below: доля баров, где Close < EntryPrice
      - min_diff_scaled: (EntryPrice - minClose) / DayRangePrev
        (т.е. насколько глубоко цена ушла вниз (в пользу шорта),
         в долях от вчерашнего рэнжа)

    Возвращает DataFrame, каждая строка = одна сделка (один вход).
    """

    # Убедимся, что нужные колонки есть
    required_cols = ['TradeEntryPrice','Close','DayRangePrev','Position','BarsHeld']
    for c in required_cols:
        if c not in df_res.columns:
            raise ValueError(f"Не найдена колонка {c} в DataFrame.")
    
    # Преобразуем DataFrame в сортированный по времени
    df_sorted = df_res.sort_index().copy()
    df_sorted['date'] = df_sorted.index.date  # для определения конца дня

    entry_indices = df_sorted.index[ ~df_sorted['TradeEntryPrice'].isna() ]  
    # Это индексы, где мы ВОШЛИ в шорт (TradeEntryPrice != NaN)

    results = []
    n = len(df_sorted)
    all_ix = df_sorted.index

    # Векторные массивы
    close_vals = df_sorted['Close'].values
    dayrange_vals = df_sorted['DayRangePrev'].values
    date_vals = df_sorted['date'].values

    # Для быстрого поиска loc:
    # df_sorted.index.get_loc(idx) даёт позицию i в 0..(n-1)
    
    for idx in entry_indices:
        i_loc = df_sorted.index.get_loc(idx)

        entry_price = df_sorted['TradeEntryPrice'].iloc[i_loc]
        day_range   = dayrange_vals[i_loc]
        entry_day   = date_vals[i_loc]  # дата входа
        
        # ===================== 1) Ближайшие 6 баров =====================
        i_loc_end_6 = i_loc + int(bars_forward)
        if i_loc_end_6 >= n:
            i_loc_end_6 = n - 1

        subset_6 = df_sorted.iloc[i_loc+1 : i_loc_end_6+1]
        # fraction_below_6: доля баров в этом интервале, где Close < EntryPrice
        if len(subset_6) == 0:
            fraction_below_6 = np.nan
            min_diff_6 = np.nan
        else:
            closes_6 = subset_6['Close'].values
            fraction_below_6 = np.mean(closes_6 < entry_price)
            min_close_6 = np.min(closes_6)
            # Для шорта "насколько ниже" = (entry_price - min_close)
            # Хотим отн. DayRangePrev
            if day_range != 0:
                min_diff_6 = (entry_price - min_close_6) / day_range
            else:
                min_diff_6 = np.nan

        # ===================== 2) До тайм-аута (holding_bars) =====================
        i_loc_end_timeout = i_loc + int(holding_bars)
        if i_loc_end_timeout >= n:
            i_loc_end_timeout = n - 1

        subset_timeout = df_sorted.iloc[i_loc+1 : i_loc_end_timeout+1]
        if len(subset_timeout) == 0:
            fraction_below_timeout = np.nan
            min_diff_timeout = np.nan
        else:
            closes_timeout = subset_timeout['Close'].values
            fraction_below_timeout = np.mean(closes_timeout < entry_price)
            min_close_timeout = np.min(closes_timeout)
            if day_range != 0:
                min_diff_timeout = (entry_price - min_close_timeout) / day_range
            else:
                min_diff_timeout = np.nan

        # ===================== 3) До конца текущего дня =====================
        # Найдём индекс последнего бара текущего дня
        same_day_mask = (date_vals == entry_day)
        # индексы всех баров этого дня
        day_ix = df_sorted.index[same_day_mask]
        # последний бар дня
        last_bar_day = day_ix[-1]
        i_loc_day_end = df_sorted.index.get_loc(last_bar_day)

        # Срез от i_loc+1 до i_loc_day_end
        if i_loc_day_end <= i_loc:
            # значит, если вход был в последний бар дня
            subset_eod = pd.DataFrame()
        else:
            subset_eod = df_sorted.iloc[i_loc+1 : i_loc_day_end+1]

        if len(subset_eod) == 0:
            fraction_below_eod = np.nan
            min_diff_eod = np.nan
        else:
            closes_eod = subset_eod['Close'].values
            fraction_below_eod = np.mean(closes_eod < entry_price)
            min_close_eod = np.min(closes_eod)
            if day_range != 0:
                min_diff_eod = (entry_price - min_close_eod) / day_range
            else:
                min_diff_eod = np.nan

        results.append({
            'entry_idx': idx,
            'entry_price': entry_price,
            'DayRangePrev': day_range,

            'fraction_below_6': fraction_below_6,
            'min_diff_6_scaled': min_diff_6,

            'fraction_below_timeout': fraction_below_timeout,
            'min_diff_timeout_scaled': min_diff_timeout,

            'fraction_below_eod': fraction_below_eod,
            'min_diff_eod_scaled': min_diff_eod,
        })
    
    df_stats = pd.DataFrame(results)
    return df_stats

def apply_strategy_short_eod_takeprofit(
    df, 
    MinRange=0.01, 
    MaxRange=0.05, 
    K=0.05, 
    commission=0.0,
    takeK=0.2
):
    """
    Шорт-стратегия:
      - Вход (EntrySignal):
         1) До 12:00 (TimeFilter).
         2) Относительный диапазон (DayRangePrev / Close) в [MinRange, MaxRange].
         3) Текущее Close > Close_prev (Trigger2).
         4) Текущее Close < Close_prev + K * DayRangePrev (Trigger3).
      - Выход:
         1) Если цена упала ниже (entry_price - takeK*DayRangePrev).
            (В шорте это "тейк-профит": мы берем прибыль, когда цена упала)
         2) Иначе в конце дня (последний бар сессии).

    Параметры:
      - MinRange, MaxRange : фильтр волатильности.
      - K : "припуск" к предыдущему закрытию.
      - commission : комиссия (доля, например 0.00017).
      - takeK : доля от DayRangePrev для тейк-профита.
    """
    df_res = df.copy()
    df_res.dropna(subset=['Close_prev', 'DayRangePrev'], inplace=True)

    # 1) Расчёт относительного диапазона
    df_res['RelDayRange'] = df_res['DayRangePrev'] / df_res['Close']

    # 2) Фильтр по времени
    df_res['TimeFilter'] = (df_res.index.hour < 12)

    # 3) Триггер по диапазону
    df_res['Trigger1'] = (df_res['RelDayRange'] > MinRange) & (df_res['RelDayRange'] < MaxRange)

    # 4) Триггер2: текущая цена выше вчерашнего закрытия
    df_res['Trigger2'] = df_res['Close'] > df_res['Close_prev']

    # 5) Триггер3: цена всё ещё < (Close_prev + K * DayRangePrev)
    df_res['Trigger3'] = df_res['Close'] < (df_res['Close_prev'] + K * df_res['DayRangePrev'])

    # Итоговый сигнал на вход (шорт)
    df_res['EntrySignal'] = (
        df_res['TimeFilter'] &
        df_res['Trigger1'] &
        df_res['Trigger2'] &
        df_res['Trigger3']
    ).astype(int)

    # Подготовка для результатов
    df_res['EntryPrice'] = np.nan
    df_res.loc[df_res['EntrySignal'] == 1, 'EntryPrice'] = df_res['Close']

    df_res['Position'] = 0
    df_res['TradeEntryPrice'] = np.nan
    df_res['TradeExitPrice'] = np.nan
    df_res['TradePnL'] = 0.0

    # Для удобства: колонка date
    df_res['date'] = df_res.index.date

    n = len(df_res)
    close_values = df_res['Close'].values
    dayrange_values = df_res['DayRangePrev'].values
    date_values = df_res['date'].values

    position = 0            # 0 - вне позиции, 1 - в шорте
    current_entry_price = np.nan
    current_day = None
    entry_price_list = [np.nan]*n
    exit_price_list = [np.nan]*n
    pnl_list = [0.0]*n
    pos_list = [0]*n

    for i in range(n):
        if position == 0:
            # Проверка входа
            if df_res['EntrySignal'].iloc[i] == 1:
                # входим в шорт по Close[i]
                position = 1
                current_entry_price = close_values[i]
                current_day = date_values[i]

                entry_price_list[i] = current_entry_price
                pos_list[i] = 1
        else:
            # Уже в шорте
            pos_list[i] = 1

            current_close = close_values[i]
            # Расчет "цены тейка" для шорта:
            #   take_price = entry_price - takeK * DayRangePrev
            take_price = current_entry_price - (takeK * dayrange_values[i])

            # 1) Проверка тейка (если Close <= take_price)
            if current_close <= take_price:
                # закрываем шорт
                exit_price_list[i] = current_close
                comm_cost = commission * (current_entry_price + current_close)
                # PnL в шорте = (Entry - Exit) - commission
                pnl_list[i] = (current_entry_price - current_close) - comm_cost

                position = 0
                current_entry_price = np.nan
                current_day = None

            else:
                # 2) Проверка конца дня
                if i == n - 1:
                    # последний бар вообще
                    exit_price_list[i] = current_close
                    comm_cost = commission * (current_entry_price + current_close)
                    pnl_list[i] = (current_entry_price - current_close) - comm_cost

                    position = 0
                    current_entry_price = np.nan
                    current_day = None
                else:
                    # Если следующий бар другой день => этот бар последний в дне
                    if date_values[i+1] != current_day:
                        # выходим по close текущего бара
                        exit_price_list[i] = current_close
                        comm_cost = commission * (current_entry_price + current_close)
                        pnl_list[i] = (current_entry_price - current_close) - comm_cost

                        position = 0
                        current_entry_price = np.nan
                        current_day = None

    # Заполняем в DataFrame
    df_res['Position'] = pos_list
    df_res['TradeEntryPrice'] = entry_price_list
    df_res['TradeExitPrice'] = exit_price_list
    df_res['TradePnL'] = pnl_list

    return df_res

def process_params_eod_takeprofit(params, df_prep):
    """
    Обрабатывает одну комбинацию параметров:
      (MinRange, MaxRange, K, commission, takeK)
    Возвращает dict с результатами статистики.
    """
    min_r, max_r, k_, comm, take_k = params
    if min_r >= max_r:
        return None

    df_signals = apply_strategy_short_eod_takeprofit(
        df_prep, 
        MinRange=min_r, 
        MaxRange=max_r, 
        K=k_, 
        commission=comm,
        takeK=take_k
    )
    stats_result = strategy_statistics(df_signals)

    row = {
        'MinRange': min_r,
        'MaxRange': max_r,
        'K': k_,
        'commission': comm,
        'takeK': take_k,
        'n_trades': stats_result['n_trades'],
        'total_pnl': stats_result['total_pnl'],
        'mean_pnl': stats_result['mean_pnl'],
        'win_rate': stats_result['win_rate'],
        't_stat': stats_result['t_stat'],
        't_pvalue': stats_result['t_pvalue'],
        'sharpe': stats_result['sharpe'],
        'max_consecutive_wins': stats_result['max_consecutive_wins'],
        'max_consecutive_losses': stats_result['max_consecutive_losses']
    }
    return row

def optimize_parameters_eod_takeprofit(
    df_prep,
    min_range_values=[0.01, 0.02],
    max_range_values=[0.04, 0.05],
    k_values=[0.02, 0.04],
    commission_values=[0.00017],
    takek_values=[0.2, 0.4],
    processes=4
):
    """
    Перебирает параметры (MinRange, MaxRange, K, commission, takeK),
    без holding_bars.
    """
    param_combinations = list(product(
        min_range_values,
        max_range_values,
        k_values,
        commission_values,
        takek_values
    ))
    
    func = partial(process_params_eod_takeprofit, df_prep=df_prep)
    
    with Pool(processes=processes) as pool:
        results = pool.map(func, param_combinations)
    
    # Убираем None
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)

def apply_strategy_short_eod_takeprofit_stop(
    df, 
    MinRange=0.01, 
    MaxRange=0.05, 
    K=0.05, 
    commission=0.0,
    takeK=0.2,
    stopK=0.1
):
    """
    Шорт-стратегия:
      - Вход (EntrySignal), как прежде:
         1) До 12:00 (TimeFilter).
         2) Относительный диапазон DayRangePrev/Close в [MinRange, MaxRange].
         3) Close > Close_prev (Trigger2).
         4) Close < (Close_prev + K*DayRangePrev) (Trigger3).
      - Выход:
         1) Если цена упала ниже (entry_price - takeK*DayRangePrev) => тейк-профит (для шорта).
         2) Если цена поднялась выше (entry_price + stopK*DayRangePrev) => стоп-лосс.
         3) Иначе в конце дня (последний бар текущей сессии).
    
    Параметры:
      - MinRange, MaxRange : фильтр волатильности (DayRangePrev/Close).
      - K : «припуск» к Close_prev.
      - commission : комиссия (доля, напр. 0.00017).
      - takeK : тейк (доля от DayRangePrev).
      - stopK : стоп (доля от DayRangePrev).
    """
    df_res = df.copy()
    df_res.dropna(subset=['Close_prev','DayRangePrev'], inplace=True)

    # 1) Расчёт относительного диапазона дня
    df_res['RelDayRange'] = df_res['DayRangePrev'] / df_res['Close']

    # 2) Фильтр по времени (до 12:00)
    df_res['TimeFilter'] = (df_res.index.hour < 12)

    # 3) Триггер1 по диапазону
    df_res['Trigger1'] = (df_res['RelDayRange'] > MinRange) & (df_res['RelDayRange'] < MaxRange)

    # 4) Триггер2: цена выше вчерашнего закрытия
    df_res['Trigger2'] = df_res['Close'] > df_res['Close_prev']

    # 5) Триггер3: цена всё ещё < (Close_prev + K * DayRangePrev)
    df_res['Trigger3'] = df_res['Close'] < (df_res['Close_prev'] + K*df_res['DayRangePrev'])

    # Итоговый сигнал на вход (шорт)
    df_res['EntrySignal'] = (
        df_res['TimeFilter'] &
        df_res['Trigger1'] &
        df_res['Trigger2'] &
        df_res['Trigger3']
    ).astype(int)

    df_res['Position'] = 0
    df_res['TradeEntryPrice'] = np.nan
    df_res['TradeExitPrice'] = np.nan
    df_res['TradePnL'] = 0.0

    # Для удобства
    df_res['EntryPrice'] = np.nan
    df_res.loc[df_res['EntrySignal'] == 1, 'EntryPrice'] = df_res['Close']
    df_res['date'] = df_res.index.date

    n = len(df_res)
    close_vals = df_res['Close'].values
    dayrange_vals = df_res['DayRangePrev'].values
    date_vals = df_res['date'].values

    position = 0  # 0 - вне позиции, 1 - в шорте
    current_entry_price = np.nan
    current_day = None

    pos_list = [0]*n
    entry_price_list = [np.nan]*n
    exit_price_list = [np.nan]*n
    pnl_list = [0.0]*n

    for i in range(n):
        if position == 0:
            # Проверяем сигнал на вход
            if df_res['EntrySignal'].iloc[i] == 1:
                # входим в шорт по Close[i]
                position = 1
                current_entry_price = close_vals[i]
                current_day = date_vals[i]

                pos_list[i] = 1
                entry_price_list[i] = current_entry_price
        else:
            # Уже в позиции (шорт)
            pos_list[i] = 1
            current_close = close_vals[i]
            drange = dayrange_vals[i]

            # Уровень тейка (шорт): entry_price - takeK * DayRangePrev
            take_price = current_entry_price - (takeK * drange)
            # Уровень стопа (шорт): entry_price + stopK * DayRangePrev
            stop_price = current_entry_price + (stopK * drange)

            # 1) Тейк-профит (цена упала в нашу пользу)
            if current_close <= take_price:
                exit_price_list[i] = current_close
                commission_cost = commission * (current_entry_price + current_close)
                # PnL (шорт) = Entry - Exit - коммиссия
                pnl_list[i] = (current_entry_price - current_close) - commission_cost
                position = 0
                current_entry_price = np.nan
                current_day = None

            # 2) Стоп-лосс (цена растёт против нас)
            elif current_close >= stop_price:
                exit_price_list[i] = current_close
                commission_cost = commission * (current_entry_price + current_close)
                pnl_list[i] = (current_entry_price - current_close) - commission_cost
                position = 0
                current_entry_price = np.nan
                current_day = None

            else:
                # 3) Конец дня
                if i == n-1:
                    # последний бар вообще
                    exit_price_list[i] = current_close
                    commission_cost = commission * (current_entry_price + current_close)
                    pnl_list[i] = (current_entry_price - current_close) - commission_cost
                    position = 0
                    current_entry_price = np.nan
                    current_day = None
                else:
                    if date_vals[i+1] != current_day:
                        # значит следующий бар => другой день
                        exit_price_list[i] = current_close
                        commission_cost = commission * (current_entry_price + current_close)
                        pnl_list[i] = (current_entry_price - current_close) - commission_cost
                        position = 0
                        current_entry_price = np.nan
                        current_day = None

    df_res['Position'] = pos_list
    df_res['TradeEntryPrice'] = entry_price_list
    df_res['TradeExitPrice'] = exit_price_list
    df_res['TradePnL'] = pnl_list

    return df_res

def process_params_stop_take(params, df_prep):
    """
    Обрабатывает одну комбинацию параметров
    для шорт-стратегии: EoD + TakeProfit + StopLoss.
    
    Параметры:
      (MinRange, MaxRange, K, commission, takeK, stopK)
    """
    min_r, max_r, k_, comm, take_k, stop_k = params
    if min_r >= max_r:
        return None
    
    df_signals = apply_strategy_short_eod_takeprofit_stop(
        df_prep,
        MinRange=min_r,
        MaxRange=max_r,
        K=k_,
        commission=comm,
        takeK=take_k,
        stopK=stop_k
    )
    stats_res = strategy_statistics(df_signals)
    
    row = {
        'MinRange': min_r,
        'MaxRange': max_r,
        'K': k_,
        'commission': comm,
        'takeK': take_k,
        'stopK': stop_k,
        'n_trades': stats_res['n_trades'],
        'total_pnl': stats_res['total_pnl'],
        'mean_pnl': stats_res['mean_pnl'],
        'win_rate': stats_res['win_rate'],
        't_stat': stats_res['t_stat'],
        't_pvalue': stats_res['t_pvalue'],
        'sharpe': stats_res['sharpe'],
        'max_consecutive_wins': stats_res['max_consecutive_wins'],
        'max_consecutive_losses': stats_res['max_consecutive_losses']
    }
    return row

def optimize_parameters_stop_take(
    df_prep,
    min_range_values=[0.01, 0.02],
    max_range_values=[0.04, 0.05],
    k_values=[0.02, 0.04],
    commission_values=[0.00017],
    takek_values=[0.2, 0.4],
    stopk_values=[0.1, 0.2],
    processes=4
):
    """
    Перебираем:
      - MinRange, MaxRange
      - K
      - commission
      - takeK
      - stopK
    И без holding_bars (выход либо по тейку, либо по стопу, либо в конце дня).
    """
    param_combinations = list(product(
        min_range_values,
        max_range_values,
        k_values,
        commission_values,
        takek_values,
        stopk_values
    ))

    func = partial(process_params_stop_take, df_prep=df_prep)

    with Pool(processes=processes) as pool:
        results = pool.map(func, param_combinations)

    # Отфильтруем None
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Загрузка и подготовка данных
    df2 = moex_candles('IMOEXF','10','2020-11-30','2025-02-18')
    df = df2[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        #data0=ticker0.dropna()
    df.index.names = ['Datetime']

    # Подготовка данных (расчёт Close_prev и DayRangePrev)
    df_prep = prepare_daily_info(df)

    # Запуск оптимизации с использованием распараллеливания
    """df_opt_res = optimize_parameters_mp(
        df_prep,
        min_range_values=[0.08, 0.01, 0.011, 0.014,0.016, 0.018, 0.02, 0.022,0.024],
        max_range_values=[ 0.035,0.04, 0.045, 0.05, 0.052],
        k_values=[ 0.04, 0.045, 0.05,0.055],
        holding_bars_values=[6, 8, 10, 15, 20, 25],
        commission_values=[0.00017],  # набор значений комиссии
        processes=4
    )"""
    
    #IMOEXF
    df_opt_res = optimize_parameters_stop_take(
        df_prep,
        min_range_values=[  0.018],
        max_range_values=[ 0.05],
        k_values=[ 0.06],
        commission_values=[0.00017],  # набор значений комиссии
        takek_values=[0.17],
        stopk_values=[0.5],
        processes=4
    )

    print("Первые 10 результатов:\n", df_opt_res.head(10))

    # Ищем лучший по total_pnl
    best_pnl_row = df_opt_res.loc[df_opt_res['total_pnl'].idxmax()]
    print("\nЛучший по total_pnl:\n", best_pnl_row)

    # Применяем стратегию с лучшими параметрами
    best_params = {
        'MinRange':   best_pnl_row['MinRange'],
        'MaxRange':   best_pnl_row['MaxRange'],
        'K':          best_pnl_row['K'],
        'commission': best_pnl_row['commission'],
        'takeK':      best_pnl_row['takeK'],
        'stopK':      best_pnl_row['stopK']
    }

    df_best = apply_strategy_short_eod_takeprofit_stop(df_prep, **best_params)
    
    # Смотрим статистику
    stats_best = strategy_statistics(df_best)
    print("\nСтатистика лучшего результата:\n", stats_best)

    # Рисуем эквити-кривую
    plot_equity_curve(df_best, title=f"Short EoD + Take({best_pnl_row['takeK']}) + Stop({best_pnl_row['stopK']})")
"""
    df_short_stats = analyze_short_price_behavior(
        df_best,
        holding_bars=best_params['holding_bars'],  # чтобы совпадало с реальным
        bars_forward=6
    )

    print(df_short_stats.head(10))

    # Смотрим средние или медианные значения
    print("\nСредние значения по всем шорт-входам:")
    print(df_short_stats.mean(numeric_only=True))

    print("\nОписательная статистика (describe):")
    print(df_short_stats.describe())"""
