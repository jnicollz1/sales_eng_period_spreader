# Period Spreader

A Python library for spreading aggregated time series data (e.g., weekly totals) to daily granularity using intelligent distribution methods.


When dealing with time series data, you may potentially have weekly totals but need daily estimates. Simply dividing by 7 ignores real-world patterns, for example a sales spike on weekends, or if activity dips on holidays, etc. This tool learns those patterns and distributes aggregated data accordingly. The tool offers multiple methods. 

## Features

- **Three spreading methods:**
  - Percentage-based spreading using historical daily patterns
  - Prophet forecast-based spreading
  - LightGBM gradient boosting with engineered time features (RECURSIVE)

- **Smart feature engineering:**
  - Lag features (1, 7, 14, 28 days)
  - Rolling statistics (mean, std)
  - Calendar features (day of week, month, quarter)
  - US holiday flags

- **Guaranteed accuracy:** Weekly totals are preserved exactly — daily values sum to the original weekly total

- **Built for performance:** Uses Polars for fast DataFrame operations

## Installation

```bash
pip install polars pandas altair prophet lightgbm
```

## Quick Start

```python
from period_spreader import PeriodSpreader

spreader = PeriodSpreader()

# Spread weekly data to daily using LightGBM (recommended)
daily_result = spreader.spread_weekly_with_lgb(
    daily_df=historical_daily_data,
    weekly_df=weekly_totals,
    metric='spend'
)
```

## Methods

### `spread_by_percentage`
Uses a reference daily time series to determine contribution patterns, then applies those patterns to distribute weekly totals.

```python
daily_result = spreader.spread_by_percentage(
    daily_reference_df=reference_data,
    weekly_df=weekly_data,
    reference_metric='sales',
    weekly_metric='sales'
)
```

### `spread_weekly_with_lgb`
End-to-end spreading using LightGBM. Trains on daily share patterns (what % of weekly total each day represents), forecasts shares, normalizes per-week, and multiplies by weekly totals.

```python
daily_result = spreader.spread_weekly_with_lgb(
    daily_df=historical_daily,
    weekly_df=weekly_totals,
    metric='spend'
)
```

### `create_forecast` / `spread_by_forecast`
Uses Facebook Prophet for time series forecasting and spreading.

```python
# Prepare data for Prophet
df_prophet = spreader.prepare_for_forecast(daily_data, 'sales')

# Create forecast
forecast = spreader.create_forecast(df_prophet, periods=30)

# Spread using forecast
daily_result = spreader.spread_by_forecast(
    forecast_df=forecast,
    weekly_df=weekly_data,
    forecast_metric='yhat',
    weekly_metric='sales'
)
```

## Visualization

Built-in Altair charts for comparing time series and visualizing forecasts:

```python
# Compare two time series
chart = spreader.compare_time_series(df1, df2, 'metric1', 'metric2')

# Visualize forecast with confidence intervals
chart = spreader.visualize_forecast(forecast_df)

# Compare original vs forecasted
chart = spreader.compare_forecast_with_original(original_df, forecast_df, 'sales')
```

## Dependencies

- `polars` — DataFrame operations
- `pandas` — Prophet compatibility
- `prophet` — Time series forecasting
- `lightgbm` — Gradient boosting
- `altair` — Visualization

## License

MIT
