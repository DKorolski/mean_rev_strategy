import datetime
import json
import logging

import backtrader as bt

from custom_data import CustomData  # класс данных, настроенный на нужный порядок столбцов
from data_prep import prepare_daily_info  # добавляем дневные данные
from moex_parser2 import moex_candles  # берем свечи

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)


def load_config(config_file="config.json"):
    with open(config_file) as f:
        return json.load(f)


class CombinedLongShortStrategy(bt.Strategy):
    params = (
        # Параметры для лонга
        ("min_range_long", 0.022),
        ("max_range_long", 0.045),
        ("k_long", 0.045),
        ("take_k_long", 0.12),
        ("stop_k_long", 0.48),
        ("comm_long", 0.00017),
        # Параметры для шорта
        ("min_range_short", 0.018),
        ("max_range_short", 0.05),
        ("k_short", 0.06),
        ("take_k_short", 0.17),
        ("stop_k_short", 0.5),
        ("comm_short", 0.00017),
        # Флаг отладки
        ("debug", False),
        # Параметры для выхода (без overnight)
        ("session_end_time", datetime.time(23, 40)),  # время последнего бара сессии
        ("exit_offset", datetime.timedelta(minutes=20)),  # отступ до конца сессии
        # Управление объёмом позиции:
        # Если test=True, то размер позиции фиксирован = 1,
        # иначе размер рассчитывается как половина доступных средств / текущая цена
        ("test", False),
    )

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.datetime(0)
        logging.info(f"{dt.isoformat()} {txt}")

    def __init__(self):
        self.orefs = []  # Список референсов на связанные ордера
        self.current_day = None  # Контроль смены дня
        self.trade_journal = []  # Журнал сделок для логирования
        self.entry_size = None  # Сохранённый размер входящего ордера
        self.entry_price = None  # Сохранённая цена входа

    def next(self):
        current_date = self.data.datetime.date(0)
        dt_now = self.data.datetime.datetime(0)
        if self.current_day is None:
            self.current_day = current_date

        # Расчет времени выхода без overnight
        session_end_dt = datetime.datetime.combine(current_date, self.p.session_end_time)
        exit_threshold = session_end_dt - self.p.exit_offset

        if dt_now >= exit_threshold and self.position:
            self.log("Время выхода: закрытие позиции до окончания сессии (без overnight).")
            self.close()
            return

        if self.orefs:
            return

        # Если позиции нет, ищем сигнал для входа
        if not self.position:
            try:
                close_prev = self.data.close_prev[0]
                dayrangeprev = self.data.dayrangeprev[0]
            except AttributeError:
                self.log("Отсутствуют необходимые поля: close_prev или dayrangeprev.")
                return

            current_close = self.data.close[0]
            rel_day_range = dayrangeprev / current_close if current_close != 0 else 0
            # Торгуем до 12:00
            time_filter = dt_now.hour < 12

            # Лонг условия
            trigger1_long = (rel_day_range > self.p.min_range_long) and (
                rel_day_range < self.p.max_range_long
            )
            trigger2_long = current_close < close_prev
            trigger3_long = current_close > (close_prev - self.p.k_long * dayrangeprev)
            long_signal = time_filter and trigger1_long and trigger2_long and trigger3_long

            # Шорт условия
            trigger1_short = (rel_day_range > self.p.min_range_short) and (
                rel_day_range < self.p.max_range_short
            )
            trigger2_short = current_close > close_prev
            trigger3_short = current_close < (close_prev + self.p.k_short * dayrangeprev)
            short_signal = time_filter and trigger1_short and trigger2_short and trigger3_short

            if self.p.debug:
                self.log(
                    f"Bar: {dt_now}, Close: {current_close:.2f}, Close_prev: {close_prev}, RelRange: {rel_day_range:.4f}"
                )
                self.log(f"Сигнал Лонг: {long_signal}, Сигнал Шорт: {short_signal}")

            # Определяем размер позиции:
            if self.p.test:
                pos_size = 1
            else:
                # Рассчитываем размер как половина доступных средств / текущая цена
                pos_size = int(self.broker.getcash() * 0.9 / current_close)
                if pos_size < 1:
                    pos_size = 1

            valid = dt_now + datetime.timedelta(days=1)
            if long_signal:
                p1 = current_close
                p2 = p1 - self.p.stop_k_long * dayrangeprev
                p3 = p1 + self.p.take_k_long * dayrangeprev
                o1 = self.buy(exectype=bt.Order.Market, size=pos_size, valid=valid, transmit=False)
                self.log(
                    f"{self.data.datetime.date(0)}: Oref {o1.ref} / Buy Market at {p1:.2f} (Size {pos_size})"
                )
                o2 = self.sell(
                    exectype=bt.Order.Stop,
                    price=p2,
                    size=pos_size,
                    valid=valid,
                    parent=o1,
                    transmit=False,
                )
                self.log(f"{self.data.datetime.date(0)}: Oref {o2.ref} / Sell Stop at {p2:.2f}")
                o3 = self.sell(
                    exectype=bt.Order.Limit,
                    price=p3,
                    size=pos_size,
                    valid=valid,
                    parent=o1,
                    transmit=True,
                )
                self.log(f"{self.data.datetime.date(0)}: Oref {o3.ref} / Sell Limit at {p3:.2f}")
                self.orefs = [o1.ref, o2.ref, o3.ref]
            elif short_signal:
                p1 = current_close
                p2 = p1 + self.p.stop_k_short * dayrangeprev
                p3 = p1 - self.p.take_k_short * dayrangeprev
                o1 = self.sell(exectype=bt.Order.Market, size=pos_size, valid=valid, transmit=False)
                self.log(
                    f"{self.data.datetime.date(0)}: Oref {o1.ref} / Sell Market at {p1:.2f} (Size {pos_size})"
                )
                o2 = self.buy(
                    exectype=bt.Order.Stop,
                    price=p2,
                    size=pos_size,
                    valid=valid,
                    parent=o1,
                    transmit=False,
                )
                self.log(f"{self.data.datetime.date(0)}: Oref {o2.ref} / Buy Stop at {p2:.2f}")
                o3 = self.buy(
                    exectype=bt.Order.Limit,
                    price=p3,
                    size=pos_size,
                    valid=valid,
                    parent=o1,
                    transmit=True,
                )
                self.log(f"{self.data.datetime.date(0)}: Oref {o3.ref} / Buy Limit at {p3:.2f}")
                self.orefs = [o1.ref, o2.ref, o3.ref]

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.parent is None:
                self.entry_size = order.executed.size
                self.entry_price = order.executed.price
            if order.isbuy():
                self.log(
                    f"BUY исполнен: Цена {order.executed.price:.2f}, Размер {order.executed.size}, Комиссия {order.executed.comm:.5f}"
                )
            elif order.issell():
                self.log(
                    f"SELL исполнен: Цена {order.executed.price:.2f}, Размер {order.executed.size}, Комиссия {order.executed.comm:.5f}"
                )
            if self.orefs:
                self.orefs = []
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Ордер отменён/отклонён/по марже")
            if self.orefs:
                self.orefs = []

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
            self.trade_journal.append(
                {
                    "EntryTime": entry_dt,
                    "ExitTime": exit_dt,
                    "EntryPrice": entry_price,
                    "ExitPrice": exit_price,
                    "PnL": trade.pnl,
                    "PnLComm": trade.pnlcomm,
                    "Size": trade_size,
                }
            )
            self.log(
                f"Сделка закрыта: Entry {entry_price:.2f}, Exit {exit_price:.2f}, PnL {trade.pnl:.2f}"
            )
            self.entry_size = None
            self.entry_price = None

    def stop(self):
        self.log("=== Журнал сделок ===")
        for idx, t in enumerate(self.trade_journal, start=1):
            self.log(f"Сделка {idx}: {t}")


if __name__ == "__main__":
    # Загрузка конфигурации (если требуется)
    config = load_config()  # например, параметры можно загрузить из файла
    # Преобразуем session_end_time (например, "23:40:00") в объект datetime.time:
    time_parts = list(map(int, config["strategy"]["session_end_time"].split(":")))
    config["strategy"]["session_end_time"] = datetime.time(*time_parts)

    # Преобразуем exit_offset (число минут) в timedelta:
    config["strategy"]["exit_offset"] = datetime.timedelta(
        minutes=config["strategy"]["exit_offset"]
    )
    cerebro = bt.Cerebro()  # стандартное создание Cerebro без cheat_on_open

    # Пример: загрузка данных поставщика – замените на реальный источник для live/paper trading
    df = moex_candles("IMOEXF", "10", "2025-01-30", "2025-10-18")

    # Подготовка данных (расчёт Close_prev и DayRangePrev)
    df_prep = prepare_daily_info(df)
    # Предполагается, что df содержит столбцы: open, high, low, close, volume, close_prev, dayrangeprev

    data = CustomData(dataname=df_prep)
    cerebro.adddata(data)
    strat_params = config["strategy"]
    cerebro.addstrategy(CombinedLongShortStrategy, **strat_params)
    cerebro.broker.setcash(config["broker"]["cash"])
    cerebro.broker.setcommission(commission=config["broker"]["commission"])
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="PyFolio")
    logging.info("Стартовый капитал: %s", cerebro.broker.getvalue())
    cerebro.run()
    logging.info("Итоговый капитал: %s", cerebro.broker.getvalue())
