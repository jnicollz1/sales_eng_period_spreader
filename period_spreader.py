"""
Period Spreader Tool v2

A modular class for spreading weekly/aggregated time series data to daily granularity
using percentage-based or forecast-based methods.

Author: Jake Nicoll
"""

import polars as pl
import altair as alt
from datetime import timedelta
import pandas as pd
from prophet import Prophet
from typing import Optional, Tuple


class PeriodSpreader:
    """
    A class for spreading aggregated time series data (e.g., weekly) to daily granularity.
    
    Supports three models:
    1. Percentage-based spreading: Uses daily contribution patterns from a reference dataset
    2. Prophet forecast-based spreading: Uses Prophet forecasts to distribute weekly totals
    3. LightGBM forecast-based spreading: Uses LightGBM gradient boosting to distribute weekly totals
    
    Attributes:
        None (stateless class - all methods are static or instance methods)
    """
    
    def __init__(self):
        """Initialize the PeriodSpreader instance."""
        pass
    
    @staticmethod
    def spread_by_percentage(
        daily_reference_df: pl.DataFrame,
        weekly_df: pl.DataFrame,
        reference_metric: str,
        weekly_metric: str,
        date_column: str = 'date'
    ) -> pl.DataFrame:
        """
        Spreads weekly data to daily data by multiplying by daily contribution.
        
        This method uses a reference daily time series to determine the daily contribution
        pattern, then applies that pattern to distribute weekly totals across days.
        
        Args:
            - daily_reference_df: Daily time series DataFrame (Polars) with date column
            - weekly_df: Weekly time series DataFrame (Polars) with date column
            - reference_metric: Metric column name in daily_reference_df to model off
            - weekly_metric: Metric column name in weekly_df to spread
            - date_column: Name of the date column (default: 'date')
            
        Returns:
            - Polars DataFrame with daily spread of weekly_metric
            
        Example:
            >>> spreader = PeriodSpreader()
            >>> daily_result = spreader.spread_by_percentage(
            ...     uk_daily_logs, 
            ...     germany_weekly_logs, 
            ...     'total_sales', 
            ...     'total_sales'
            ... )
        """
        # Validate date column exists
        if date_column not in daily_reference_df.columns:
            raise ValueError(f"Date column '{date_column}' not found in daily_reference_df")
        if date_column not in weekly_df.columns:
            raise ValueError(f"Date column '{date_column}' not found in weekly_df")
        
        # Find daily contribution to metric
        daily_contribution = (
            daily_reference_df
            .with_columns(
                (pl.col(reference_metric) / pl.col(reference_metric).sum()).alias('daily_contribution')
            )
        )
        
        # Find common dates between the two datasets
        # Date ranges to compare: daily vs. expanded weekly
        weekly_date_range = (
            pl.date_range(
                start = weekly_df[date_column].min(),
                end = weekly_df[date_column].max() + timedelta(days = 7),
                eager = True
            )
            .rename(date_column)
        ).to_frame()
        
        overlap_dates = (
            daily_contribution
            .join(
                weekly_date_range,
                on = date_column,
                how = 'inner'
            )
            .select(date_column, 'daily_contribution')
        )
        
        # Add in total weekly sales, multiply by daily contribution
        weekly_spread = (
            overlap_dates
            .with_columns(
                pl.lit((weekly_df[weekly_metric].sum())).alias(weekly_metric)
            )
            .with_columns(
                (pl.col('daily_contribution') * pl.col(weekly_metric)).alias(weekly_metric)
            )
            .select(date_column, weekly_metric)
        )
        
        return weekly_spread
    
    @staticmethod
    def prepare_for_forecast(
        df_polars: pl.DataFrame,
        metric: str,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Convert Polars DataFrame to Pandas format for Prophet forecasting.
        
        Args:
            - df_polars: Polars DataFrame with date and metric columns
            - metric: Name of the metric column to forecast
            - date_column: Name of the date column (default: 'date')
            
        Returns:
            - Pandas DataFrame with 'ds' (date) and 'y' (metric) columns
            
        Example:
            >>> spreader = PeriodSpreader()
            >>> df_pd = spreader.prepare_for_forecast(uk_daily_logs, 'total_sales')
        """
        if date_column not in df_polars.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        if metric not in df_polars.columns:
            raise ValueError(f"Metric column '{metric}' not found in DataFrame")
        
        df_pd = (
            df_polars.select(date_column, metric)
            .rename({
                date_column: 'ds',
                metric: 'y'
            })
            .to_pandas()
        )
        
        return df_pd
    
    @staticmethod
    def create_forecast(
        df_pd: pd.DataFrame,
        periods: int = 1,
        prophet_params: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Create a forecast using Prophet time series model.
        
        Args:
            - df_pd: Pandas DataFrame with 'ds' (date) and 'y' (metric) columns
            - periods: Number of periods to forecast ahead (default: 1)
            - prophet_params: Optional dictionary of Prophet parameters
            
        Returns:
            - Pandas DataFrame with forecast results including 'yhat', 'yhat_lower', 'yhat_upper'
            
        Example:
            >>> spreader = PeriodSpreader()
            >>> forecast = spreader.create_forecast(df_pd, periods = 7)
        """
        if 'ds' not in df_pd.columns or 'y' not in df_pd.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")
        
        # Instantiate Prophet model with optional parameters
        if prophet_params:
            m = Prophet(**prophet_params)
        else:
            m = Prophet()
        
        m.fit(df_pd)
        
        # Make future dataframe
        future = m.make_future_dataframe(periods = periods)
        forecast = m.predict(future)
        
        return forecast
    
    # US Federal Holidays (fixed dates and approximate floating dates)
    # This is a simple list - for production, consider using the 'holidays' package
    US_HOLIDAYS = {
        # Fixed holidays (month, day)
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas
        (12, 31), # New Year's Eve
        # Common retail holidays
        (2, 14),  # Valentine's Day
        (10, 31), # Halloween
        (11, 11), # Veterans Day
    }
    
    # Floating holidays approximated by week-of-year (week, weekday, name)
    # weekday: 0=Monday, 6=Sunday
    FLOATING_HOLIDAYS_APPROX = {
        # (month, week_of_month, weekday)
        (1, 3, 0),   # MLK Day - 3rd Monday of January
        (2, 3, 0),   # Presidents Day - 3rd Monday of February
        (5, 5, 0),   # Memorial Day - Last Monday of May (approx week 5)
        (9, 1, 0),   # Labor Day - 1st Monday of September
        (11, 4, 3),  # Thanksgiving - 4th Thursday of November
    }

    @staticmethod
    def _is_us_holiday(date_col: pl.Expr) -> pl.Expr:
        """Check if date is a US holiday (simplified check)."""
        month = date_col.dt.month()
        day = date_col.dt.day()
        
        # Check fixed holidays
        is_fixed_holiday = (
            ((month == 1) & (day == 1)) |   # New Year's
            ((month == 7) & (day == 4)) |   # Independence Day
            ((month == 12) & (day == 25)) | # Christmas
            ((month == 12) & (day == 31)) | # New Year's Eve
            ((month == 2) & (day == 14)) |  # Valentine's
            ((month == 10) & (day == 31)) | # Halloween
            ((month == 11) & (day == 11))   # Veterans Day
        )
        
        return is_fixed_holiday

    @staticmethod
    def _is_near_holiday(date_col: pl.Expr) -> pl.Expr:
        """Check if date is within 1 day of a major holiday."""
        month = date_col.dt.month()
        day = date_col.dt.day()
        
        # Days adjacent to major holidays
        is_near = (
            # Around New Year's
            ((month == 1) & (day <= 2)) |
            ((month == 12) & (day >= 30)) |
            # Around July 4th
            ((month == 7) & (day >= 3) & (day <= 5)) |
            # Around Christmas
            ((month == 12) & (day >= 23) & (day <= 26)) |
            # Around Thanksgiving (late November)
            ((month == 11) & (day >= 22) & (day <= 28))
        )
        
        return is_near

    @staticmethod
    def create_lgb_features(
        df: pl.DataFrame,
        metric: str,
        date_column: str = 'date',
        lags: list = None,
        rolling_windows: list = None,
        include_holiday_features: bool = True
    ) -> pl.DataFrame:
        """
        Create time-based features for LightGBM forecasting.

        Args:
            - df: Polars DataFrame with date and metric columns
            - metric: Name of the metric column to create features for
            - date_column: Name of the date column (default: 'date')
            - lags: List of lag periods (default: [1, 7, 14, 28])
            - rolling_windows: List of rolling window sizes (default: [7, 14, 28])
            - include_holiday_features: Whether to include US holiday flags (default: True)

        Returns:
            - Polars DataFrame with additional feature columns
        """
        if lags is None:
            lags = [1, 7, 14, 28]
        if rolling_windows is None:
            rolling_windows = [7, 14, 28]

        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")
        if metric not in df.columns:
            raise ValueError(f"Metric column '{metric}' not found in DataFrame")

        df = df.sort(date_column)

        # ===========================================
        # Basic Calendar Features
        # ===========================================
        df = df.with_columns([
            pl.col(date_column).dt.weekday().alias('day_of_week'),
            pl.col(date_column).dt.month().alias('month'),
            pl.col(date_column).dt.day().alias('day_of_month'),
        ])
        
        # ===========================================
        # Enhanced Calendar Features (High ROI)
        # ===========================================
        df = df.with_columns([
            # Weekend flag
            (pl.col(date_column).dt.weekday() >= 5).cast(pl.Int8).alias('is_weekend'),
            
            # Week of year (1-52)
            pl.col(date_column).dt.week().alias('week_of_year'),
            
            # Month boundary flags
            (pl.col(date_column).dt.day() <= 3).cast(pl.Int8).alias('is_month_start'),
            (pl.col(date_column).dt.day() >= 28).cast(pl.Int8).alias('is_month_end'),
            
            # Quarter
            ((pl.col(date_column).dt.month() - 1) // 3 + 1).alias('quarter'),
        ])
        
        # ===========================================
        # Holiday Features
        # ===========================================
        if include_holiday_features:
            df = df.with_columns([
                PeriodSpreader._is_us_holiday(pl.col(date_column)).cast(pl.Int8).alias('is_holiday'),
                PeriodSpreader._is_near_holiday(pl.col(date_column)).cast(pl.Int8).alias('is_near_holiday'),
            ])

        # ===========================================
        # Lag Features
        # ===========================================
        for lag in lags:
            df = df.with_columns(
                pl.col(metric).shift(lag).alias(f'lag_{lag}')
            )

        # ===========================================
        # Rolling Statistics
        # ===========================================
        for window in rolling_windows:
            df = df.with_columns([
                pl.col(metric).rolling_mean(window_size=window).alias(f'rolling_mean_{window}'),
                pl.col(metric).rolling_std(window_size=window).alias(f'rolling_std_{window}'),
            ])
        
        # ===========================================
        # Rolling Mean by Day-of-Week
        # Average metric for this weekday over past N occurrences
        # ===========================================
        # For each day, compute rolling mean of same-weekday values
        # Using lag 7, 14, 21, 28 (same weekday 1-4 weeks ago)
        df = df.with_columns(
            ((pl.col(f'lag_7') + 
              pl.col(f'lag_14') + 
              pl.col(f'lag_28')) / 3.0).alias('rolling_mean_same_weekday')
        )

        return df

    @staticmethod
    def create_lgb_forecast(
        df: pl.DataFrame,
        metric: str,
        periods: int = 7,
        date_column: str = 'date',
        lgb_params: Optional[dict] = None
    ) -> pl.DataFrame:
        """
        Create a forecast using LightGBM gradient boosting model.

        Args:
            - df: Polars DataFrame with date and metric columns
            - metric: Name of the metric column to forecast
            - periods: Number of days to forecast ahead (default: 7)
            - date_column: Name of the date column (default: 'date')
            - lgb_params: Optional dictionary of LightGBM parameters

        Returns:
            - Polars DataFrame with 'date' and 'yhat' columns for forecast

        Example:
            >>> spreader = PeriodSpreader()
            >>> forecast = spreader.create_lgb_forecast(daily_data, 'total_sales', periods=14)
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is required for this method. "
                "Install with: pip install lightgbm"
            )
        #select only the columns we need 
        # Select only the columns we need
        df = df.select([date_column, metric])
        # Create features
        df_features = PeriodSpreader.create_lgb_features(df, metric, date_column)

        # Drop rows with NaN from lag/rolling features
        df_clean = df_features.drop_nulls()

        # Define feature columns
        feature_cols = [c for c in df_clean.columns if c not in [date_column, metric]]

        X = df_clean.select(feature_cols).to_pandas()
        y = df_clean.select(metric).to_pandas().values.ravel()

        # Default parameters
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100
        }
        params = {**default_params, **(lgb_params or {})}

        # Train model
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)

        # Recursive forecasting
        predictions = []
        last_date = df[date_column].max()
        current_df = df.clone()

        for i in range(periods):
            next_date = last_date + timedelta(days=i + 1)

            # Rebuild features with current history
            current_features = PeriodSpreader.create_lgb_features(
                current_df, metric, date_column
            )

            # Get the last row's features for prediction
            last_row = current_features.tail(1).select(feature_cols)

            # Predict
            pred = model.predict(last_row.to_pandas())[0]
            pred = max(0, pred)  # Ensure non-negative

            predictions.append({date_column: next_date, 'yhat': pred})

            # Append prediction to history for next iteration
            new_row = pl.DataFrame({date_column: [next_date], metric: [pred]})
            current_df = pl.concat([current_df, new_row])

        return pl.DataFrame(predictions)

    @staticmethod
    def spread_by_lgb_forecast(
        forecast_df: pl.DataFrame,
        weekly_df: pl.DataFrame,
        weekly_metric: str,
        date_column: str = 'date'
    ) -> pl.DataFrame:
        """
        Spread weekly data to daily using LightGBM forecasted patterns.
        
        Uses a LightGBM forecast to determine daily contribution patterns,
        then applies those patterns to distribute weekly totals.
        
        Args:
            - forecast_df: Polars DataFrame with LightGBM forecast (must have 'yhat' column)
            - weekly_df: Weekly time series DataFrame (Polars) with date column
            - weekly_metric: Metric column name in weekly_df to spread
            - date_column: Name of the date column (default: 'date')
            
        Returns:
            - Polars DataFrame with daily spread of weekly_metric
            
        Example:
            >>> spreader = PeriodSpreader()
            >>> forecast = spreader.create_lgb_forecast(historical_data, 'sales', periods=30)
            >>> daily_result = spreader.spread_by_lgb_forecast(
            ...     forecast,
            ...     weekly_data,
            ...     'total_sales'
            ... )
        """
        if 'yhat' not in forecast_df.columns:
            raise ValueError("forecast_df must have 'yhat' column from LightGBM forecast")
        
        return PeriodSpreader.spread_by_percentage(
            forecast_df,
            weekly_df,
            'yhat',
            weekly_metric,
            date_column
        )

    @staticmethod
    def compute_daily_share(
        df: pl.DataFrame,
        metric: str,
        date_column: str = 'date'
    ) -> pl.DataFrame:
        """
        Compute daily share (proportion) of metric within each week.
        
        Args:
            - df: Polars DataFrame with date and metric columns
            - metric: Name of the metric column
            - date_column: Name of the date column (default: 'date')
            
        Returns:
            - Polars DataFrame with 'week_start' and 'share' columns added
        """
        # Compute Monday-aligned week start
        df = df.with_columns(
            (pl.col(date_column) - pl.duration(days=pl.col(date_column).dt.weekday())).alias('week_start')
        )
        
        # Compute share within each week
        df = df.with_columns(
            (pl.col(metric) / pl.col(metric).sum().over('week_start')).alias('share')
        )
        
        return df

    @staticmethod
    def create_lgb_share_forecast(
        df: pl.DataFrame,
        metric: str,
        weekly_df: pl.DataFrame,
        date_column: str = 'date',
        lgb_params: Optional[dict] = None
    ) -> pl.DataFrame:
        """
        Create a LightGBM forecast trained on daily SHARE (not raw values).
        
        This method learns the intra-week pattern (what % of weekly total each day gets)
        rather than absolute values, making spreading more accurate.
        
        Args:
            - df: Polars DataFrame with historical daily data
            - metric: Name of the metric column to model
            - weekly_df: Weekly data to spread (used to determine forecast period)
            - date_column: Name of the date column (default: 'date')
            - lgb_params: Optional dictionary of LightGBM parameters
            
        Returns:
            - Polars DataFrame with 'date', 'week_start', and 'share' columns
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is required for this method. "
                "Install with: pip install lightgbm"
            )
        
        # Select only needed columns
        df = df.select([date_column, metric])
        
        # Compute daily share within each week
        df_share = PeriodSpreader.compute_daily_share(df, metric, date_column)
        
        # Create features on SHARE instead of raw metric
        df_features = PeriodSpreader.create_lgb_features(
            df_share.select([date_column, 'share']).rename({'share': metric}),
            metric,
            date_column
        )
        
        # Drop nulls from lag features
        df_clean = df_features.drop_nulls()
        
        # Define feature columns
        feature_cols = [c for c in df_clean.columns if c not in [date_column, metric]]
        
        X = df_clean.select(feature_cols).to_pandas()
        y = df_clean.select(metric).to_pandas().values.ravel()
        
        # Default parameters
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100
        }
        params = {**default_params, **(lgb_params or {})}
        
        # Train model
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        
        # Determine forecast period from weekly_df
        # Need to cover all weeks in weekly_df
        first_week_start = weekly_df[date_column].min()
        last_week_start = weekly_df[date_column].max()
        # Generate dates for all days in all weeks
        forecast_start = first_week_start
        forecast_end = last_week_start + timedelta(days=6)  # Include full last week
        
        # Generate all dates to forecast
        all_forecast_dates = []
        current_date = forecast_start
        while current_date <= forecast_end:
            all_forecast_dates.append(current_date)
            current_date = current_date + timedelta(days=1)
        
        # Recursive forecasting
        predictions = []
        current_df = df_share.select([date_column, 'share']).rename({'share': metric})
        
        for next_date in all_forecast_dates:
            # Rebuild features with current history
            current_features = PeriodSpreader.create_lgb_features(
                current_df, metric, date_column
            )
            
            # Get the last row's features for prediction
            last_row = current_features.tail(1).select(feature_cols)
            
            # Predict share
            pred = model.predict(last_row.to_pandas())[0]
            pred = max(0.001, pred)  # Ensure positive share
            
            # Compute week_start for this date (Monday-aligned)
            week_start = next_date - timedelta(days=next_date.weekday())
            
            predictions.append({
                date_column: next_date,
                'week_start': week_start,
                'share': pred
            })
            
            # Append prediction to history for next iteration
            new_row = pl.DataFrame({date_column: [next_date], metric: [pred]})
            current_df = pl.concat([current_df, new_row])
        
        return pl.DataFrame(predictions)

    @staticmethod
    def spread_by_lgb_share(
        share_forecast_df: pl.DataFrame,
        weekly_df: pl.DataFrame,
        weekly_metric: str,
        date_column: str = 'date'
    ) -> pl.DataFrame:
        """
        Spread weekly data to daily using share-based forecast with per-week normalization.
        
        GUARANTEES:
        - Exactly 7 days per week
        - Weekly totals match EXACTLY (sum of daily = weekly total)
        - No missing dates (uses fallback if forecast missing a date)
        
        Args:
            - share_forecast_df: DataFrame with 'date', 'week_start', 'share' columns
            - weekly_df: Weekly totals to spread (date column should be Monday week starts)
            - weekly_metric: Metric column name in weekly_df
            - date_column: Name of the date column (default: 'date')
            
        Returns:
            - Polars DataFrame with daily spread values (exactly 7 * len(weekly_df) rows)
        """
        if 'share' not in share_forecast_df.columns:
            raise ValueError("share_forecast_df must have 'share' column")
        
        results = []
        
        # Process each week explicitly
        for row in weekly_df.iter_rows(named=True):
            week_start = row[date_column]
            weekly_total = row[weekly_metric]
            
            # Generate exactly 7 days for this week (Monday through Sunday)
            week_dates = [week_start + timedelta(days=i) for i in range(7)]
            
            # Get shares for these 7 dates from forecast
            shares = []
            for d in week_dates:
                # Look up this date in forecast
                match = share_forecast_df.filter(pl.col(date_column) == d)
                if len(match) > 0:
                    shares.append(float(match['share'][0]))
                else:
                    # Fallback: uniform distribution if date missing from forecast
                    shares.append(1.0 / 7.0)
            
            # Normalize shares within this week to sum to exactly 1.0
            total_share = sum(shares)
            if total_share > 0:
                norm_shares = [s / total_share for s in shares]
            else:
                norm_shares = [1.0 / 7.0] * 7  # Fallback to uniform
            
            # Apply normalized shares to weekly total
            for i, d in enumerate(week_dates):
                daily_value = norm_shares[i] * weekly_total
                results.append({
                    date_column: d,
                    weekly_metric: daily_value
                })
        
        return pl.DataFrame(results).sort(date_column)

    @staticmethod
    def spread_weekly_with_lgb(
        daily_df: pl.DataFrame,
        weekly_df: pl.DataFrame,
        metric: str,
        date_column: str = 'date',
        lgb_params: Optional[dict] = None
    ) -> pl.DataFrame:
        """
        End-to-end: spread weekly data to daily using share-based LightGBM.
        
        This is the recommended method for accurate weekly-to-daily spreading.
        It trains on daily share patterns, forecasts shares for the weekly period,
        normalizes per-week, and multiplies by weekly totals.
        
        Args:
            - daily_df: Historical daily data for training
            - weekly_df: Weekly totals to spread (date column = Monday week starts)
            - metric: Metric column name (must exist in both DataFrames)
            - date_column: Name of the date column (default: 'date')
            - lgb_params: Optional dictionary of LightGBM parameters
            
        Returns:
            - Polars DataFrame with daily spread values
            
        Example:
            >>> spreader = PeriodSpreader()
            >>> daily_result = spreader.spread_weekly_with_lgb(
            ...     historical_daily,
            ...     weekly_totals,
            ...     'spend'
            ... )
        """
        # Step 1: Train on share and forecast
        share_forecast = PeriodSpreader.create_lgb_share_forecast(
            daily_df,
            metric,
            weekly_df,
            date_column,
            lgb_params
        )
        
        # Step 2: Spread with per-week normalization
        result = PeriodSpreader.spread_by_lgb_share(
            share_forecast,
            weekly_df,
            metric,
            date_column
        )
        
        return result

    @staticmethod
    def spread_by_forecast(
        forecast_df: pd.DataFrame,
        weekly_df: pl.DataFrame,
        forecast_metric: str,
        weekly_metric: str,
        date_column: str = 'date'
    ) -> pl.DataFrame:
        """
        Spread weekly data to daily using forecasted patterns.
        
        Uses a Prophet forecast to determine daily contribution patterns,
        then applies those patterns to distribute weekly totals.
        
        Args:
            - forecast_df: Pandas DataFrame with Prophet forecast results
            - weekly_df: Weekly time series DataFrame (Polars) with date column
            - forecast_metric: Metric column name in forecast_df to model off (e.g., 'yhat')
            - weekly_metric: Metric column name in weekly_df to spread
            - date_column: Name of the date column (default: 'date')
            
        Returns:
            - Polars DataFrame with daily spread of weekly_metric
            
        Example:
            >>> spreader = PeriodSpreader()
            >>> daily_result = spreader.spread_by_forecast(
            ...     forecast_df, 
            ...     germany_weekly_logs, 
            ...     'yhat', 
            ...     'total_sales'
            ... )
        """
        # Convert forecast to Polars
        forecast_df['date'] = forecast_df['ds'].dt.date
        forecast_df_polars = pl.from_pandas(forecast_df)
        
        # Rename forecast metric to match expected format
        if forecast_metric != 'yhat':
            forecast_df_polars = forecast_df_polars.rename({forecast_metric: 'yhat'})
        
        # Use percentage-based spreading with forecast as reference
        split_df = PeriodSpreader.spread_by_percentage(
            forecast_df_polars,
            weekly_df,
            'yhat',
            weekly_metric,
            date_column
        )
        
        return split_df
    
    @staticmethod
    def compare_time_series(
        df1: pl.DataFrame,
        df2: pl.DataFrame,
        metric1: str,
        metric2: str,
        label1: str = 'daily_time_series',
        label2: str = 'weekly_time_series',
        date_column: str = 'date'
    ) -> alt.Chart:
        """
        Create an Altair chart comparing two time series with correlation.
        
        Args:
            - df1: First time series DataFrame (Polars)
            - df2: Second time series DataFrame (Polars)
            - metric1: Metric column name in df1
            - metric2: Metric column name in df2
            - label1: Label for df1 in the chart (default: 'daily_time_series')
            - label2: Label for df2 in the chart (default: 'weekly_time_series')
            - date_column: Name of the date column (default: 'date')
            
        Returns:
            - Altair Chart object comparing the two time series
            
        Example:
            >>> spreader = PeriodSpreader()
            >>> chart = spreader.compare_time_series(
            ...     uk_daily_logs, 
            ...     germany_daily_logs, 
            ...     'total_sales', 
            ...     'total_sales'
            ... )
        """
        # Wide format to calculate correlation
        df_wide = (
            df1
            .select(date_column, metric1)
            .rename({metric1: 'metric_left'})
            .join(
                df2
                .select(date_column, metric2)
                .rename({metric2: 'metric_right'}),
                on = date_column,
                how = 'inner'
            )
        )
        
        # Long format to plot
        df_long = (
            pl.concat([
                df1
                .select(date_column, metric1)
                .with_columns(pl.lit(label1).alias('time_series_type'))
                .rename({metric1: 'value'}),
                df2
                .select(date_column, metric2)
                .with_columns(pl.lit(label2).alias('time_series_type'))
                .rename({metric2: 'value'})
            ])
        )
        
        # Calculate correlation
        correlation = df_wide.select(pl.corr('metric_left', 'metric_right')).to_series().item()
        
        # Create the chart
        chart = alt.Chart(df_long.to_pandas()).mark_line().encode(
            x = alt.X(date_column, title = 'Date'),
            y = alt.Y('value:Q', title = 'Metric Value'),
            color = alt.Color('time_series_type:N', title = 'Time Series Type')
        ).properties(
            title = f'Time Series Comparison - Correlation: {correlation:.3f}'
        )
        
        return chart
    
    @staticmethod
    def visualize_forecast(forecast_df: pd.DataFrame) -> alt.Chart:
        """
        Create an Altair chart visualizing Prophet forecast with confidence intervals.
        
        Args:
            - forecast_df: Pandas DataFrame with Prophet forecast results
            
        Returns:
            - Altair Chart object showing forecast with confidence bands
            
        Example:
            >>> spreader = PeriodSpreader()
            >>> chart = spreader.visualize_forecast(forecast_df)
        """
        forecast_df['date'] = pd.to_datetime(forecast_df['ds'])
        line = alt.Chart(forecast_df).mark_line().encode(
            x = alt.X('date', title = 'Date'),
            y = alt.Y('yhat', title = 'Forecasted Metric')
        )
        band = alt.Chart(forecast_df).mark_area(opacity = 0.2).encode(
            x = 'date',
            y = 'yhat_upper',
            y2 = 'yhat_lower'
        )
        
        chart = (band + line).properties(
            title = 'Forecasted Metric with Confidence Intervals'
        )
        return chart
    
    @staticmethod
    def compare_forecast_with_original(
        original_df: pl.DataFrame,
        forecast_df: pd.DataFrame,
        metric: str,
        date_column: str = 'date'
    ) -> alt.Chart:
        """
        Create an Altair chart comparing original data with forecast.
        
        Args:
            - original_df: Original Polars DataFrame that was forecasted
            - forecast_df: Forecasted Pandas DataFrame from Prophet
            - metric: Metric column name in original_df to compare
            - date_column: Name of the date column (default: 'date')
            
        Returns:
            - Altair Chart object plotting both time series
            
        Example:
            >>> spreader = PeriodSpreader()
            >>> chart = spreader.compare_forecast_with_original(
            ...     uk_daily_logs, 
            ...     forecast_df, 
            ...     'total_sales'
            ... )
        """
        # Prepare original data
        df1 = (
            original_df
            .with_columns(pl.lit('original').alias('source'))
            .rename({metric: 'metric'})
        )
        
        # Prepare forecast data
        forecast_df['source'] = 'forecast'
        forecast_df['date'] = forecast_df['ds'].dt.date
        forecast_df_polars = pl.from_pandas(forecast_df)
        forecast_df_polars = forecast_df_polars.rename({
            'yhat': 'metric'
        })
        
        # Combine
        df_concat = pl.concat([
            df1.select(date_column, 'metric', 'source'),
            forecast_df_polars.select('date', 'metric', 'source')
        ])
        
        # Create visualization
        chart = alt.Chart(df_concat.to_pandas()).mark_line().encode(
            x = alt.X('date', title = 'Date'),
            y = alt.Y('metric', title = 'Metric Value'),
            color = alt.Color('source:N', title = 'Source')
        ).properties(
            title = 'Original vs Forecasted Time Series'
        )
        
        return chart

