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

def process_params_eod(params, df_prep):
    """
    Обрабатывает одну комбинацию параметров для стратегии "лонг до конца дня или по тейку (опц. стоп)".
    """
    # Предположим, что порядок параметров в product(...) именно такой:
    # (min_r, max_r, k_, comm, stop_k, take_k)
    min_r, max_r, k_, comm, stop_k, take_k = params
    
    # Проверка "MinRange < MaxRange"
    if min_r >= max_r:
        return None

    # Вызываем новую функцию
    df_signals = apply_strategy_long_eod_takeprofit(
        df_prep,
        MinRange=min_r,
        MaxRange=max_r,
        K=k_,
        commission=comm,
        takeK=take_k,
        stopK=stop_k  # если не хотите стоп, можно передавать None
    )
    stats_result = strategy_statistics(df_signals)
    
    row = {
        'MinRange':  min_r,
        'MaxRange':  max_r,
        'K':         k_,
        'commission':comm,
        'stopK':     stop_k,
        'takeK':     take_k,
        'n_trades':  stats_result['n_trades'],
        'total_pnl': stats_result['total_pnl'],
        'mean_pnl':  stats_result['mean_pnl'],
        'win_rate':  stats_result['win_rate'],
        't_stat':    stats_result['t_stat'],
        't_pvalue':  stats_result['t_pvalue'],
        'sharpe':    stats_result['sharpe'],
        'max_consecutive_wins':   stats_result['max_consecutive_wins'],
        'max_consecutive_losses': stats_result['max_consecutive_losses']
    }
    return row



def optimize_parameters_eod(df_prep,
                            min_range_values=[0.022],
                            max_range_values=[0.045],
                            k_values=[0.045],
                            commission_values=[0.00017],
                            stopk_values=[0.48],  # Пример: None или конкретные значения
                            takek_values=[0.12, 0.16],
                            processes=4):
    """
    Перебираем параметры:
      - MinRange, MaxRange, K, commission, stopK, takeK
    Без holding_bars.
    """
    param_combinations = list(product(
        min_range_values,
        max_range_values,
        k_values,
        commission_values,
        stopk_values,
        takek_values
    ))

    func = partial(process_params_eod, df_prep=df_prep)

    with Pool(processes=processes) as pool:
        results = pool.map(func, param_combinations)

    results = [r for r in results if r is not None]
    return pd.DataFrame(results)

def apply_strategy_long(df, 
                        MinRange=0.01, 
                        MaxRange=0.05, 
                        K=0.05, 
                        holding_bars=15,
                        commission=0.0,
                        stopK=0.02):
    """
    Лонговая версия стратегии со стоп-лоссом.
    
    Параметры:
      - MinRange, MaxRange: фильтр по волатильности предыдущего дня (DayRangePrev/Close).
      - K: «уровень отката» относительно вчерашнего закрытия.
      - holding_bars: максимальное количество баров, которые держим позицию.
      - commission: комиссия в долях (0.001 = 0.1%).
      - stopK: стоп-лосс (доля от вчерашнего диапазона, DayRangePrev).
        stop_price = entry_price - stopK * DayRangePrev(на момент входа).

    Логика входа (примерная):
      1) До 12:00 (TimeFilter)
      2) Достаточная волатильность (RelDayRange в [MinRange; MaxRange])
      3) Текущая цена ниже вчерашнего закрытия (Trigger2)
      4) Но не слишком низко (Trigger3)
    
    Логика выхода:
      1) По достижению holding_bars баров, 
         ИЛИ
      2) По стоп-лоссу, если Close < stop_price.
    """

    df_res = df.copy()
    # Отбрасываем бары, где нет нужных значений
    df_res.dropna(subset=['Close_prev', 'DayRangePrev'], inplace=True)

    # 1) Относительный диапазон
    df_res['RelDayRange'] = df_res['DayRangePrev'] / df_res['Close']

    # 2) Фильтр по времени (до 12:00)
    df_res['TimeFilter'] = (df_res.index.hour < 12)

    # 3) Триггер по диапазону
    df_res['Trigger1'] = (df_res['RelDayRange'] > MinRange) & (df_res['RelDayRange'] < MaxRange)

    # 4) Триггер2: покупка «на откате» (цена ниже вчерашнего закрытия)
    df_res['Trigger2'] = df_res['Close'] < df_res['Close_prev']

    # 5) Триггер3: не покупаем слишком глубокий провал
    df_res['Trigger3'] = df_res['Close'] > (df_res['Close_prev'] - K * df_res['DayRangePrev'])

    # Итоговый сигнал на вход (1 = вход лонг)
    df_res['EntrySignal'] = (
        df_res['TimeFilter'] &
        df_res['Trigger1'] &
        df_res['Trigger2'] &
        df_res['Trigger3']
    ).astype(int)

    df_res['EntryPrice'] = np.nan
    df_res.loc[df_res['EntrySignal'] == 1, 'EntryPrice'] = df_res['Close']

    # Подготовим списки для результатов
    n = len(df_res)
    position = [0] * n
    bars_in_position = [0] * n
    entry_price_list = [np.nan] * n
    exit_price_list = [np.nan] * n
    pnl_list = [0] * n
    stop_price_list = [np.nan] * n  # чтобы записать для контроля

    # Внутренние переменные цикла
    in_pos = False
    bars_held = 0
    current_entry_price = np.nan
    current_stop_price = np.nan  # динамический, но здесь сделаем его «фиксированным» на момент входа
    close_values = df_res['Close'].values
    dayrange_prev_values = df_res['DayRangePrev'].values

    for i, row in enumerate(df_res.itertuples()):
        if not in_pos:
            # Проверяем вход
            if row.EntrySignal == 1:
                in_pos = True
                bars_held = 0
                current_entry_price = close_values[i]
                # фиксируем DayRangePrev, чтобы стоп был неизменен всё время сделки
                entry_dayrange = dayrange_prev_values[i]  
                current_stop_price = current_entry_price - stopK * entry_dayrange

                position[i] = 1
                entry_price_list[i] = current_entry_price
                bars_in_position[i] = bars_held
                stop_price_list[i] = current_stop_price

        else:
            # Уже в позиции
            position[i] = 1
            bars_held += 1
            bars_in_position[i] = bars_held
            stop_price_list[i] = current_stop_price

            # Проверяем два условия выхода:
            # 1) Дошли до лимита баров
            if bars_held >= holding_bars:
                exit_price_list[i] = close_values[i]
                commission_cost = commission * (current_entry_price + close_values[i])
                pnl_list[i] = (close_values[i] - current_entry_price) - commission_cost
                in_pos = False
                bars_held = 0

            # 2) Cтоп-лосс: если Close < current_stop_price
            elif close_values[i] < current_stop_price:
                exit_price_list[i] = close_values[i]
                commission_cost = commission * (current_entry_price + close_values[i])
                pnl_list[i] = (close_values[i] - current_entry_price) - commission_cost
                in_pos = False
                bars_held = 0

            # Если вышли, сбрасываем текущие цены
            if not in_pos:
                current_entry_price = np.nan
                current_stop_price = np.nan

    # Записываем всё обратно в DataFrame
    df_res['Position'] = position
    df_res['BarsHeld'] = bars_in_position
    df_res['TradeEntryPrice'] = entry_price_list
    df_res['TradeExitPrice'] = exit_price_list
    df_res['TradePnL'] = pnl_list
    df_res['StopPrice'] = stop_price_list

    return df_res

def analyze_price_behavior_after_entry(df,
                                       holding_bars=15):
    """
    Для каждого входа (EntrySignal == 1) считаем:
      1) "До тайм-аута": i+1 ... i+holding_bars (если не вышли за границы дня)
         - Доля баров выше EntryPrice
         - Среднее (Close - EntryPrice)
         - Среднее ((Close - EntryPrice) / DayRangePrev)
      2) "До конца дня": i+1 ... последний бар текущего дня
         - То же самое
         
    Возвращает DataFrame со статистикой по каждому входу.
    """
    
    # Убедимся, что у нас есть нужные столбцы
    required_cols = ['EntrySignal', 'Close', 'DayRangePrev']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"В DataFrame нет колонки {c}")
    
    # Преобразуем индекс в сортированный список (на случай, если не отсортирован)
    df_sorted = df.sort_index().copy()
    df_sorted['date'] = df_sorted.index.date  # удобная колонка с датой
    
    # Найдём все индексы, где EntrySignal == 1
    entry_indices = df_sorted.index[df_sorted['EntrySignal'] == 1]
    
    results = []
    
    for idx in entry_indices:
        row_entry = df_sorted.loc[idx]
        entry_price = row_entry['Close']
        day_range = row_entry['DayRangePrev']
        entry_day = row_entry['date']  # дата входа
        
        # -----------------------------------------------------------
        # 1) Формируем срез "до тайм-аута"
        # -----------------------------------------------------------
        # 1.1) Индекс входа (числовой) в df_sorted
        i_loc = df_sorted.index.get_loc(idx)  
        
        # 1.2) Ищем максимально i_loc_end = i_loc + holding_bars
        i_loc_end = i_loc + holding_bars
        
        # Но нужно, чтобы i_loc_end не вышел за пределы:
        if i_loc_end >= len(df_sorted):
            i_loc_end = len(df_sorted) - 1
        
        # 1.3) Смотрим, чтобы не менять день (если день закончился раньше)
        #     Т. е. пока date совпадает, двигаемся. Как только day != entry_day — это уже следующий день
        #     Найдём индекс последнего бара того же дня:
        #     Можно отфильтровать df_sorted по той же дате и взять index[-1].
        same_day_mask = (df_sorted['date'] == entry_day)
        day_indices = df_sorted.loc[same_day_mask].index
        day_last_idx = day_indices[-1]
        
        # Числовая позиция последнего бара дня
        i_loc_day_last = df_sorted.index.get_loc(day_last_idx)
        
        # Фактическая граница по тайм-аута-срезу: минимум из (i_loc_end, i_loc_day_last)
        i_loc_end_timeout = min(i_loc_end, i_loc_day_last)
        
        if i_loc_end_timeout <= i_loc:  
            # Значит вообще нет "следующих баров" в этот день
            # (например, вошли в последний бар дня)
            subset_timeout = pd.DataFrame()
        else:
            # slice
            subset_timeout = df_sorted.iloc[i_loc+1 : i_loc_end_timeout+1]  
        
        # -----------------------------------------------------------
        # 2) Формируем срез "до конца дня": i+1 ... day_last_idx
        # -----------------------------------------------------------
        if i_loc_day_last <= i_loc:
            # Аналогично: если входим в последний бар, у нас нет "следующих"
            subset_dayend = pd.DataFrame()
        else:
            subset_dayend = df_sorted.iloc[i_loc+1 : i_loc_day_last+1]
        
        # -----------------------------------------------------------
        # Считаем нужные метрики
        # -----------------------------------------------------------
        
        # функция-помощник
        def calc_metrics(subset, entry_px, day_range):
            """
            Возвращает dict с:
              - fraction_above
              - mean_delta
              - mean_delta_scaled
            Если subset пуст, вернём None
            """
            if len(subset) == 0:
                return None
            closes = subset['Close'].values
            
            # Доля баров выше цены входа
            fraction_above = np.mean(closes > entry_px)
            
            # Среднее отклонение (Close - entry_px) по всем барам
            mean_delta = np.mean(closes - entry_px)
            
            # Среднее отклонение в долях DayRangePrev
            if day_range != 0:
                mean_delta_scaled = np.mean((closes - entry_px) / day_range)
            else:
                mean_delta_scaled = np.nan
            
            return {
                'fraction_above': fraction_above,
                'mean_delta': mean_delta,
                'mean_delta_scaled': mean_delta_scaled
            }
        
        # Метрики по тайм-ауту
        metrics_timeout = calc_metrics(subset_timeout, entry_price, day_range)
        # Метрики до конца дня
        metrics_dayend = calc_metrics(subset_dayend, entry_price, day_range)
        
        # Сохраним в общий список
        results.append({
            'entry_idx': idx,
            'entry_date': entry_day,
            'entry_price': entry_price,
            'day_range': day_range,
            
            'fraction_above_timeout': (
                metrics_timeout['fraction_above'] if metrics_timeout else np.nan
            ),
            'mean_delta_timeout': (
                metrics_timeout['mean_delta'] if metrics_timeout else np.nan
            ),
            'mean_delta_scaled_timeout': (
                metrics_timeout['mean_delta_scaled'] if metrics_timeout else np.nan
            ),
            
            'fraction_above_dayend': (
                metrics_dayend['fraction_above'] if metrics_dayend else np.nan
            ),
            'mean_delta_dayend': (
                metrics_dayend['mean_delta'] if metrics_dayend else np.nan
            ),
            'mean_delta_scaled_dayend': (
                metrics_dayend['mean_delta_scaled'] if metrics_dayend else np.nan
            )
        })
        
    # Собираем всё в DataFrame
    res_df = pd.DataFrame(results)
    return res_df

def apply_strategy_long_eod_takeprofit(
    df, 
    MinRange=0.022, 
    MaxRange=0.045,
    K=0.045,
    commission=0.00017,
    takeK=0.12,
    stopK=0.48  # или 0.02, если хотим всегда использовать
):
    """
    Лонговая стратегия:
      - Вход:
         1) До 12:00 (TimeFilter).
         2) Отн. диапазон вчерашнего дня в [MinRange, MaxRange].
         3) Текущая цена < вчерашнее закрытие (Trigger2).
         4) Цена не слишком низко (Trigger3: Close > Close_prev - K * DayRangePrev).
      - Выход:
         1) Тейк-профит: Close >= EntryPrice + takeK * DayRangePrev
         2) (Если stopK не None) — стоп-лосс: Close <= EntryPrice - stopK * DayRangePrev
         3) Конец дня (последний бар).

    Параметры:
      - MinRange, MaxRange: фильтр волатильности (DayRangePrev / Close).
      - K: фильтр «не глубокое падение» (Trigger3).
      - commission: комиссия (доля).
      - takeK: тейк-профит (доля от DayRangePrev).
      - stopK: стоп-лосс (доля от DayRangePrev). Если None — не используется.
    """
    df_res = df.copy()
    df_res.dropna(subset=['Close_prev', 'DayRangePrev'], inplace=True)

    df_res['RelDayRange'] = df_res['DayRangePrev'] / df_res['Close']
    df_res['TimeFilter'] = (df_res.index.hour < 12)
    df_res['Trigger1'] = (df_res['RelDayRange'] > MinRange) & (df_res['RelDayRange'] < MaxRange)
    df_res['Trigger2'] = df_res['Close'] < df_res['Close_prev']
    df_res['Trigger3'] = df_res['Close'] > (df_res['Close_prev'] - K * df_res['DayRangePrev'])

    df_res['EntrySignal'] = (
        df_res['TimeFilter'] &
        df_res['Trigger1'] &
        df_res['Trigger2'] &
        df_res['Trigger3']
    ).astype(int)

    df_res['EntryPrice'] = np.nan
    df_res.loc[df_res['EntrySignal'] == 1, 'EntryPrice'] = df_res['Close']

    df_res['date'] = df_res.index.date

    n = len(df_res)
    position = [0] * n
    entry_price_list = [np.nan] * n
    exit_price_list = [np.nan] * n
    pnl_list = [0] * n

    in_pos = False
    current_entry_price = np.nan
    take_profit_price = np.nan
    stop_loss_price = np.nan  # если нужно
    current_day = None

    close_values = df_res['Close'].values
    dayrange_values = df_res['DayRangePrev'].values
    date_values = df_res['date'].values

    for i, row in enumerate(df_res.itertuples()):
        if not in_pos:
            # Проверяем вход
            if row.EntrySignal == 1:
                in_pos = True
                current_entry_price = close_values[i]
                current_day = date_values[i]

                # Тейк-профит: Entry + takeK * DayRangePrev
                take_profit_price = current_entry_price + takeK * dayrange_values[i]

                # (Опционально) стоп-лосс
                if stopK is not None and stopK > 0:
                    stop_loss_price = current_entry_price - stopK * dayrange_values[i]
                else:
                    stop_loss_price = None

                position[i] = 1
                entry_price_list[i] = current_entry_price

        else:
            # Уже в лонге
            position[i] = 1
            current_close = close_values[i]

            # 1) Проверка тейк-профита
            if current_close >= take_profit_price:
                exit_price_list[i] = current_close
                commission_cost = commission * (current_entry_price + current_close)
                pnl_list[i] = (current_close - current_entry_price) - commission_cost

                in_pos = False
                current_entry_price = np.nan
                current_day = None
                take_profit_price = np.nan
                stop_loss_price = np.nan
            else:
                # 2) Если стоп-лосс включён
                if stop_loss_price is not None and current_close <= stop_loss_price:
                    exit_price_list[i] = current_close
                    commission_cost = commission * (current_entry_price + current_close)
                    pnl_list[i] = (current_close - current_entry_price) - commission_cost

                    in_pos = False
                    current_entry_price = np.nan
                    current_day = None
                    take_profit_price = np.nan
                    stop_loss_price = np.nan
                else:
                    # 3) Проверка конца дня
                    is_last_bar = False
                    if i == n - 1:
                        # последний бар вообще
                        is_last_bar = True
                    else:
                        if date_values[i+1] != current_day:
                            # следующий бар другой день => этот бар - последний в дне
                            is_last_bar = True

                    if is_last_bar:
                        exit_price_list[i] = current_close
                        commission_cost = commission * (current_entry_price + current_close)
                        pnl_list[i] = (current_close - current_entry_price) - commission_cost

                        in_pos = False
                        current_entry_price = np.nan
                        current_day = None
                        take_profit_price = np.nan
                        stop_loss_price = np.nan

    df_res['Position'] = position
    df_res['TradeEntryPrice'] = entry_price_list
    df_res['TradeExitPrice'] = exit_price_list
    df_res['TradePnL'] = pnl_list

    return df_res

import numpy as np
import pandas as pd

def analyze_downward_movement_after_signal(df, 
                                           signal_col='EntrySignal', 
                                           price_col='Close',
                                           bars_forward=6,
                                           use_low=True):
    """
    Смотрит, насколько цена уходит ниже цены сигнала в ближайшие bars_forward баров.
    
    Параметры:
      - signal_col: имя колонки, где 1 = сигнал на вход.
      - price_col: имя колонки с ценами (обычно 'Close').
      - bars_forward: на сколько баров вперёд смотреть.
      - use_low: если True, учитываем 'Low' (если есть в df), чтобы взять 
                 реальный минимум внутри бара, а не только закрытие.
    
    Возвращает DataFrame со строками для каждого сигнала, где есть:
      - 'signal_idx': индекс сигнала
      - 'signal_price': цена сигнала
      - 'min_price_nextN': минимальная цена (Low или Close) в ближайших bars_forward барах
      - 'delta_min': (min_price_nextN - signal_price) (часто будет отрицательное)
      - 'bars_forward': сколько реально баров смотрели (если в конце массива меньше)
    """
    if signal_col not in df.columns:
        raise ValueError(f"Колонка {signal_col} не найдена в DataFrame")
    if price_col not in df.columns:
        raise ValueError(f"Колонка {price_col} не найдена в DataFrame")
    
    df_sorted = df.sort_index()
    signals_idx = df_sorted.index[df_sorted[signal_col] == 1]

    # Если хотим смотреть Low — проверим, что 'Low' есть в df
    if use_low and ('Low' not in df_sorted.columns):
        raise ValueError("В DataFrame нет колонки 'Low', а use_low=True")

    results = []
    n = len(df_sorted)
    all_index = df_sorted.index
    close_vals = df_sorted[price_col].values
    low_vals = df_sorted['Low'].values if use_low else close_vals

    for idx in signals_idx:
        # Числовая позиция в DataFrame
        i_loc = df_sorted.index.get_loc(idx)
        signal_price = close_vals[i_loc]

        # Срез на следующие bars_forward баров
        i_loc_end = i_loc + bars_forward
        if i_loc_end >= n:
            i_loc_end = n - 1  # если мало баров до конца

        if i_loc_end <= i_loc:
            # значит сигнальный бар - последний вообще
            min_price = np.nan
            actual_forward = 0
        else:
            subset_low = low_vals[i_loc+1 : i_loc_end+1]  # i_loc+1, ... +bars_forward
            if len(subset_low) == 0:
                min_price = np.nan
                actual_forward = 0
            else:
                min_price = np.min(subset_low)
                actual_forward = len(subset_low)

        delta_min = min_price - signal_price if not np.isnan(min_price) else np.nan

        results.append({
            'signal_idx': idx,
            'signal_price': signal_price,
            'min_price_nextN': min_price,
            'delta_min': delta_min,
            'bars_forward': actual_forward
        })

    return pd.DataFrame(results)



   

if __name__ == "__main__":
    # Загрузка и подготовка данных
    df2 = moex_candles('IMOEXF','10','2020-11-30','2025-10-18')
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
    df_signals = apply_strategy_long_eod_takeprofit(
        df_prep,
        MinRange=0.022,
        MaxRange=0.045,
        K=0.045,
        commission=0.00017,
        takeK=0.12
    )
    
    plot_equity_curve(df_signals, title="Strategy EoD + TakeProfit=0.15 Range")
    stats_result = strategy_statistics(df_signals)
    print(stats_result)

    df_opt_res = optimize_parameters_eod(
        df_prep,
        min_range_values=[ 0.018,0.022],
        max_range_values=[ 0.045],
        k_values=[ 0.045],#,0.08,0.09, 0.1],
        commission_values=[0.00017],
        stopk_values=[0.48],  # пример — либо без стопа, либо 0.01
        takek_values=[0.12],
        processes=4
    )

    print("Результаты оптимизации (первые 10):")
    print(df_opt_res.sort_values(by='sharpe', ascending=False).head(10))

    best_by_pnl = df_opt_res.loc[df_opt_res['total_pnl'].idxmax()]
    print("\nЛучший по total_pnl:")
    print(best_by_pnl)

    # Применяем стратегию с лучшими параметрами
    best_params = {
        'MinRange':  best_by_pnl['MinRange'],
        'MaxRange':  best_by_pnl['MaxRange'],
        'K':         best_by_pnl['K'],
        'commission':best_by_pnl['commission'],
        'stopK':     best_by_pnl['stopK'],
        'takeK':     best_by_pnl['takeK']
    }

    df_best = apply_strategy_long_eod_takeprofit(
        df_prep,
        MinRange=best_params['MinRange'],
        MaxRange=best_params['MaxRange'],
        K=best_params['K'],
        commission=best_params['commission'],
        takeK=best_params['takeK'],
        stopK=best_params['stopK']
    )

    plot_equity_curve(df_best, title=f"Best EoD + Take {best_params['takeK']}, Stop {best_params['stopK']}")
    stats_best = strategy_statistics(df_best)
    print("\nСтатистика лучшего результата:\n", stats_best)
