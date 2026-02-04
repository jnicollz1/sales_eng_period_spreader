# sales_eng_period_spreader
A modular Python library for distributing aggregated time series data to finer granularity. Given weekly totals and a reference pattern, it estimates daily values using three approaches:  Percentage-based spreading using historical daily patterns Prophet forecast-based spreading LightGBM gradient boosting with engineered features.
