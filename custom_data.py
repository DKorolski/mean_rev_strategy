import backtrader as bt

class CustomData(bt.feeds.PandasData):
    """
    Custom Data Feed.
    
    Ожидается, что входной DataFrame имеет следующие столбцы (в указанном порядке):
      0: open
      1: high
      2: low
      3: close
      4: volume
      5: close_prev
      6: dayrangeprev
    
    Поле openinterest не используется, поэтому устанавливаем значение -1.
    """
    # Добавляем новые линии (close_prev, dayrangeprev)
    lines = ('close_prev', 'dayrangeprev',)
    
    params = (
        ('datetime', None),      # Если индекс DataFrame уже datetime
        ('open', 0),             # open - первый столбец
        ('high', 1),             # high - второй столбец
        ('low', 2),              # low - третий столбец
        ('close', 3),            # close - четвёртый столбец
        ('volume', 4),           # volume - пятый столбец
        ('openinterest', -1),    # openinterest не используется
        ('close_prev', 5),       # close_prev - шестой столбец
        ('dayrangeprev', 6),     # dayrangeprev - седьмой столбец
    )