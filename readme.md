# Mean Reversion Trading Strategy

## Strategy Description

For each instrument, whether it is a future or a stock, there is a specific window of current dynamic parameters that indicates the instrument has entered a phase of relative non-trending behavior. Provided that this phase persists, any movement in one direction will fade and eventually reverse. This forms the basis for building a counter-trend system.

### Steps:
- Identify a relatively strong dynamic in the specific instrument.
- Wait until the dynamics weaken.
- Identify the movement in either direction.
- Verify that this movement is relatively weak.
- Open a position opposite to the movement.
- Add a set of conditions for holding the position and exiting it.

System entry is executed during the first half of the day, before 12:00, as the current conditions are compared with the dynamics from the previous day. The influence of the previous day is only relevant in the early hours of trading. The check to confirm that the market has moved upward is performed using the condition:

```python
YestDayClose = Close[1]
Trigger2 = Close > YestDayClose
```

The third condition requires that the movement is not too strong. This means that, at the time of verification, the market should be close to yesterday’s closing level. For this, a coefficient K multiplied by the daily range DayRange is used.

## Exiting the Position

Exits are threefold – via take-profit, stop-loss, and timeout:
- **Take-Profit:** If, at the close of a 15-minute bar, the price is below yesterday's closing level, the position is closed.
- **Stop-Loss:** The position is closed when a predetermined loss level is reached.
- **Timeout:** Exit after a fixed number of bars if the price has not reached either the take-profit or the stop-loss level.

## Test Results

**Testing period:** November 14, 2023 - February 21, 2025

### Key Metrics:
- **Total PnL:** Positive, with steady capital growth.
- **Maximum Drawdown:** Within acceptable limits.
- **Sharpe Ratio:** Moderately high, indicating a good risk-to-reward ratio.
- **Number of Trades:** Over 100, with a high proportion of profitable trades.

### Charts:
- **Price Chart with Trade Marks (model_equity.png):** Shows entries (green triangles) and exits (red triangles) superimposed on the price series.
- **Equity Curve Chart (Equity Curve):** Demonstrates smooth capital growth with periodic pullbacks.
- **Extended Analysis (pre_prod_test.png):** Displays detailed trade statistics, including volumes, PnL, and return distribution.
- **Test Report (stats_mean_rev.html):** Contains a complete analytical review of the strategy with key metrics and graphical analysis.

## Conclusions and Possible Improvements

- The strategy demonstrates stable profitability, although there are periods of high volatility.
- The exit algorithm could be improved by adding dynamic stop-loss management.
- The strategy should be tested on lower timeframes (for example, 10-minute candles) to identify potential improvements.
- Additional analysis of factors affecting mean reversion, such as trading volumes or macroeconomic events, might be beneficial.

## Files

- **model_equity.png** – Price chart with trade entries and exits.
- **pre_prod_test.png** – Detailed trade report.
- **stats_mean_rev.html** – HTML test report.
- **mean_rev_app.py** – Strategy code.
- **mat_model_test.py** – Test models and calculations.

## Contacts

For additional information or suggestions for improving the strategy, please contact me.
