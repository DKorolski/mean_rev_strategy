import backtrader as bt
import datetime
from pathlib import Path
import sys
import pandas as pd
import quantstats

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from moex_parser2 import moex_candles

class CombinedLongShortStrategy(bt.Strategy):
    params = (
        # Параметры для лонга
        ('min_range_long', 0.022),
        ('max_range_long', 0.045),
        ('k_long', 0.045),
        ('take_k_long', 0.12),
        ('stop_k_long', 0.48),
        ('comm_long', 0.00017),
        # Параметры для шорта
        ('min_range_short', 0.018),
        ('max_range_short', 0.05),
        ('k_short', 0.06),
        ('take_k_short', 0.17),
        ('stop_k_short', 0.5),
        ('comm_short', 0.00017),
        # Флаг отладки
        ('debug', True),
        # Механизм ордеров: если True – использовать bracket-ордера, если False – цепочку связанных ордеров
        ('usebracket', False),
        # Параметры для выхода до конца дня (без overnight)
        ('session_end_time', datetime.time(23, 40)),   # время последнего бара торговой сессии
        ('exit_offset', datetime.timedelta(minutes=20)),  # выход за 20 минут до конца сессии (примерно 2 бара при 10-минутном таймфрейме)
    )

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.datetime(0)
        print(f"{dt.isoformat()} {txt}")

    def __init__(self):
        self.orefs = []           # Для хранения ссылок на связанные ордера
        self.current_day = None   # Для контроля смены дня
        self.trade_journal = []   # Журнал сделок для логирования
        # Для фиксации параметров входящего ордера (размер и цену)
        self.entry_size = None
        self.entry_price = None

    def next(self):
        current_date = self.data.datetime.date(0)
        dt_now = self.data.datetime.datetime(0)
        if self.current_day is None:
            self.current_day = current_date

        # Рассчитываем дату-время конца сессии текущего дня с учетом отступа
        session_end_dt = datetime.datetime.combine(current_date, self.p.session_end_time)
        exit_threshold = session_end_dt - self.p.exit_offset

        # Если мы достигли времени выхода (до конца дня) и позиция открыта – закрываем позицию,
        # чтобы избежать overnight удержания.
        if dt_now >= exit_threshold and self.position:
            self.log("Время выхода (до конца дня): закрытие позиции для избежания overnight.")
            self.close()
            return

        # Если уже есть открытые заявки, не ищем новый сигнал
        if self.orefs:
            return

        # Если позиции нет – ищем сигнал для входа
        if not self.position:
            try:
                close_prev = self.data.close_prev[0]
                dayrangeprev = self.data.dayrangeprev[0]
            except AttributeError:
                self.log("Датасет не содержит необходимых полей (close_prev, dayrangeprev)")
                return

            current_close = self.data.close[0]
            rel_day_range = dayrangeprev / current_close if current_close != 0 else 0
            # Торгуем до 12:00 (как и раньше)
            time_filter = dt_now.hour < 12

            # --- Лонг условия ---
            trigger1_long = (rel_day_range > self.p.min_range_long) and (rel_day_range < self.p.max_range_long)
            trigger2_long = current_close < close_prev
            trigger3_long = current_close > (close_prev - self.p.k_long * dayrangeprev)
            long_signal = time_filter and trigger1_long and trigger2_long and trigger3_long

            # --- Шорт условия ---
            trigger1_short = (rel_day_range > self.p.min_range_short) and (rel_day_range < self.p.max_range_short)
            trigger2_short = current_close > close_prev
            trigger3_short = current_close < (close_prev + self.p.k_short * dayrangeprev)
            short_signal = time_filter and trigger1_short and trigger2_short and trigger3_short

            if self.p.debug:
                self.log(f"Bar: {dt_now}, Close: {current_close:.2f}, Close_prev: {close_prev}, RelRange: {rel_day_range:.4f}")
                self.log(f"Сигнал Лонг: {long_signal}, Сигнал Шорт: {short_signal}")

            if not self.p.usebracket:
                valid = dt_now + datetime.timedelta(days=1)
                if long_signal:
                    p1 = current_close  # вход по цене текущего бара
                    p2 = p1 - self.p.stop_k_long * dayrangeprev  # стоп-лосс
                    p3 = p1 + self.p.take_k_long * dayrangeprev  # тейк-профит

                    o1 = self.buy(exectype=None,
                                  price=p1,
                                  valid=valid,
                                  transmit=False)
                    self.log(f"{self.data.datetime.date(0)}: Oref {o1.ref} / Buy at {p1:.2f}")

                    o2 = self.sell(exectype=bt.Order.Stop,
                                   price=p2,
                                   valid=valid,
                                   parent=o1,
                                   transmit=False)
                    self.log(f"{self.data.datetime.date(0)}: Oref {o2.ref} / Sell Stop at {p2:.2f}")

                    o3 = self.sell(exectype=bt.Order.Limit,
                                   price=p3,
                                   valid=valid,
                                   parent=o1,
                                   transmit=True)
                    self.log(f"{self.data.datetime.date(0)}: Oref {o3.ref} / Sell Limit at {p3:.2f}")

                    self.orefs = [o1.ref, o2.ref, o3.ref]
                elif short_signal:
                    p1 = current_close  # вход по цене текущего бара
                    p2 = p1 + self.p.stop_k_short * dayrangeprev  # стоп-лосс для шорта
                    p3 = p1 - self.p.take_k_short * dayrangeprev  # тейк-профит для шорта

                    o1 = self.sell(exectype=None,
                                   price=p1,
                                   valid=valid,
                                   transmit=False)
                    self.log(f"{self.data.datetime.date(0)}: Oref {o1.ref} / Sell at {p1:.2f}")

                    o2 = self.buy(exectype=bt.Order.Stop,
                                  price=p2,
                                  valid=valid,
                                  parent=o1,
                                  transmit=False)
                    self.log(f"{self.data.datetime.date(0)}: Oref {o2.ref} / Buy Stop at {p2:.2f}")

                    o3 = self.buy(exectype=bt.Order.Limit,
                                  price=p3,
                                  valid=valid,
                                  parent=o1,
                                  transmit=True)
                    self.log(f"{self.data.datetime.date(0)}: Oref {o3.ref} / Buy Limit at {p3:.2f}")

                    self.orefs = [o1.ref, o2.ref, o3.ref]
            else:
                # Если usebracket True – использовать встроенные методы bracket
                if long_signal:
                    entry_price = current_close
                    take_price = entry_price + self.p.take_k_long * dayrangeprev
                    stop_price = entry_price - self.p.stop_k_long * dayrangeprev
                    self.log(f"Вход в Лонг: цена {entry_price:.2f} | TP: {take_price:.2f} | SL: {stop_price:.2f}")
                    self.bracket_order = self.buy_bracket(
                        size=1,
                        price=entry_price,
                        stopprice=stop_price,
                        limitprice=take_price,
                    )
                elif short_signal:
                    entry_price = current_close
                    take_price = entry_price - self.p.take_k_short * dayrangeprev
                    stop_price = entry_price + self.p.stop_k_short * dayrangeprev
                    self.log(f"Вход в Шорт: цена {entry_price:.2f} | TP: {take_price:.2f} | SL: {stop_price:.2f}")
                    self.bracket_order = self.sell_bracket(
                        size=1,
                        price=entry_price,
                        stopprice=stop_price,
                        limitprice=take_price,
                    )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.parent is None:
                self.entry_size = order.executed.size
                self.entry_price = order.executed.price

            if order.isbuy():
                self.log(f"BUY исполнен: Цена {order.executed.price:.2f}, Размер {order.executed.size}, Комиссия {order.executed.comm:.5f}")
            elif order.issell():
                self.log(f"SELL исполнен: Цена {order.executed.price:.2f}, Размер {order.executed.size}, Комиссия {order.executed.comm:.5f}")

            if not self.p.usebracket and self.orefs:
                self.orefs = []
            elif self.p.usebracket:
                if self.bracket_order and all(o.status in [o.Completed, o.Canceled, o.Margin, o.Rejected] for o in self.bracket_order):
                    self.bracket_order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Ордер отменён/отклонён/по марже")
            if not self.p.usebracket and self.orefs:
                self.orefs = []
            elif self.p.usebracket:
                if self.bracket_order and all(o.status in [o.Canceled, o.Margin, o.Rejected] for o in self.bracket_order):
                    self.bracket_order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            trade_size = self.entry_size if self.entry_size is not None else 1
            entry_price = self.entry_price if self.entry_price is not None else trade.price

            if abs(trade_size) < 1e-8:
                exit_price = trade.price
            else:
                if trade_size > 0:
                    exit_price = entry_price + (trade.pnl / trade_size)
                else:
                    exit_price = entry_price - (trade.pnl / abs(trade_size))
            
            entry_dt = bt.num2date(trade.dtopen)
            exit_dt = bt.num2date(trade.dtclose)
            self.trade_journal.append({
                'EntryTime': entry_dt,
                'ExitTime': exit_dt,
                'EntryPrice': entry_price,
                'ExitPrice': exit_price,
                'PnL': trade.pnl,
                'PnLComm': trade.pnlcomm,
                'Size': trade_size,
            })
            self.log(f"Сделка закрыта: Entry {entry_price:.2f}, Exit {exit_price:.2f}, PnL {trade.pnl:.2f}")
            self.entry_size = None
            self.entry_price = None

    def stop(self):
        self.log("=== Журнал сделок ===")
        for idx, t in enumerate(self.trade_journal, start=1):
            self.log(f"Сделка {idx}: {t}")


#2. Пользовательский класс данных с дополнительными полями.
class CustomData(bt.feeds.PandasData):
    """
    Ожидается, что входной DataFrame имеет следующие столбцы (в указанном порядке):
      0: open
      1: high
      2: low
      3: close
      4: volume
      5: close_prev
      6: dayrangeprev
    """
    lines = ('close_prev', 'dayrangeprev',)
    params = (
        ('datetime', None),  # если индекс уже является датой
        ('open', 0),
        ('high', 1),
        ('low', 2),
        ('close', 3),
        ('volume', 4),
        ('openinterest', -1),
        ('close_prev', 5),
        ('dayrangeprev', 6),
    )

#подготовка дневных данных
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
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        .rename(columns={
            'high': 'high_day',
            'low': 'low_day',
            'close': 'close_day'
        })
    )
    daily_info['high_prev'] = daily_info['high_day'].shift(1)
    daily_info['low_prev'] = daily_info['low_day'].shift(1)
    daily_info['close_prev'] = daily_info['close_day'].shift(1)
    daily_info['dayrangeprev'] = daily_info['high_prev'] - daily_info['low_prev']

    df_merged = pd.merge(
        df_intra.reset_index(),  # каждая свеча с отдельным временем
        daily_info[['date', 'close_prev', 'dayrangeprev']],
        on='date',
        how='left'
    )
    # Протягиваем значения до конца дня (при необходимости)
    df_merged[['close_prev', 'dayrangeprev']] = df_merged[['close_prev', 'dayrangeprev']].ffill()
    df_merged.set_index('datetime', inplace=True)
    df_merged.drop('date', axis=1, inplace=True)
    return df_merged

# 3. Часть запуска стратегии
if __name__ == '__main__':
    df2 = moex_candles('IMOEXF','10','2020-11-30','2025-10-18')
    df = df2[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index.names = ['datetime']
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    print(df)

    # Подготовка данных (расчёт Close_prev и DayRangePrev)
    df_prep = prepare_daily_info(df)
    cerebro = bt.Cerebro()

    # Если необходимо, убедитесь, что порядок колонок соответствует параметрам:
    df = df_prep[['open', 'high', 'low', 'close', 'volume', 'close_prev', 'dayrangeprev']]

    # Создаём экземпляр пользовательского фида
    data = CustomData(dataname=df)
    cerebro.adddata(data)

    # Добавляем стратегию
    cerebro.addstrategy(CombinedLongShortStrategy)

    # Настраиваем стартовый капитал и комиссию брокера
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.00017)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    print("Стартовый капитал:", cerebro.broker.getvalue())
    result = cerebro.run()
    print("Итоговый капитал:", cerebro.broker.getvalue())

    # Отобразить график (при наличии графической оболочки)
    cerebro.plot(style='candlestick')
    # Получаем результаты PyFolio
    strategy_instance = result[0]
    portfolio_stats = strategy_instance.analyzers.getbyname('PyFolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)

    # Выводим HTML-отчёт QuantStats
    quantstats.reports.html(returns, output='stats_mean_rev.html', title='Mean_rev_test')
