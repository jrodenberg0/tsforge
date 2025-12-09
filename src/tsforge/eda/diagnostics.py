import numpy as np
import pandas as pd


def pct_missing_dates(x):
    freq = pd.infer_freq(x)
    if not freq:
        return np.nan
    expected_idx = pd.date_range(start=x.min(), end=x.max(), freq=freq)
    return (len(expected_idx) - len(x)) / len(expected_idx) * 100


def datetime_diagnostics(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str = None,  # Optional target for seasonal analysis
) -> pd.DataFrame:
    """Comprehensive time series diagnostics per unique_id for forecasting.

    Returns key information about temporal structure, gaps, and frequency patterns
    that inform forecasting approach and data quality.

    If target_col is provided, also returns seasonal pattern information.
    """

    def infer_frequency(dates):
        """Infer frequency and return as string"""
        freq = pd.infer_freq(dates.sort_values())
        return freq if freq else "irregular"

    def count_gaps(dates):
        """Count number of gaps in expected regular sequence"""
        dates_sorted = dates.sort_values()
        if len(dates_sorted) < 2:
            return 0
        freq = pd.infer_freq(dates_sorted)
        if freq is None:
            return np.nan
        expected_range = pd.date_range(dates_sorted.min(), dates_sorted.max(), freq=freq)
        return len(expected_range) - len(dates_sorted)

    def span_days(dates):
        """Total time span in days"""
        return (dates.max() - dates.min()).total_seconds() / 86400

    def obs_per_year(dates):
        """Approximate observations per year"""
        span = (dates.max() - dates.min()).total_seconds() / 86400
        if span == 0:
            return np.nan
        return (len(dates) / span) * 365.25

    def has_duplicates(dates):
        """Check if there are duplicate timestamps"""
        return dates.duplicated().any()

    # Base aggregations that always run
    base_agg = {
        # Basic temporal boundaries
        "start_date": (date_col, "min"),
        "end_date": (date_col, "max"),
        "n_obs": (date_col, "count"),
        "span_days": (date_col, span_days),
        # Frequency detection
        "inferred_freq": (date_col, infer_frequency),
        "obs_per_year": (date_col, obs_per_year),
        # Gap analysis
        "n_gaps": (date_col, count_gaps),
        "pct_missing": (date_col, pct_missing_dates),
        # Data quality flags
        "has_duplicates": (date_col, has_duplicates),
    }
    result = df.groupby(id_col).agg(**base_agg)

    if target_col is not None:

        def _seasonal_summary(group):
            if group.empty or group[target_col].dropna().empty:
                return pd.Series({"peak_month": np.nan, "peak_quarter": np.nan})

            month_means = group.groupby(group[date_col].dt.month)[target_col].mean()
            quarter_means = group.groupby(group[date_col].dt.quarter)[target_col].mean()

            return pd.Series(
                {
                    "peak_month": month_means.idxmax() if not month_means.empty else np.nan,
                    "peak_quarter": quarter_means.idxmax() if not quarter_means.empty else np.nan,
                }
            )

        seasonal_stats = df.groupby(id_col).apply(_seasonal_summary)
        result = result.merge(seasonal_stats, left_index=True, right_index=True, how="left")
    return result
