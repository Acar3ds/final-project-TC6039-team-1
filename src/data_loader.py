import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class MissingColumnError(Exception):
    """Raised when a required column is missing from the DataFrame."""
    pass

def load_data(raw_data_dir: str = "data/raw/") -> pd.DataFrame:
    """
    Read JSON files from raw_data_dir, convert frame_timestamp to local Mexico City
    datetime and return a pandas DataFrame with columns:
      ['counter', 'date', 'hour', 'minute', 'day_type', 'weekday_number']
    """
    monterrey_tz = ZoneInfo("America/Mexico_City")

    # Collect all data
    all_data = []

    for json_file in Path(raw_data_dir).glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

                # Handle both single dict and list of dicts
                items = data if isinstance(data, list) else [data]

                for item in items:
                    if "frame_timestamp" in item:
                        timestamp_value = item["frame_timestamp"]

                        # Accept Unix epoch in either seconds or milliseconds.
                        epoch = float(timestamp_value)
                        if epoch > 1e12:
                            epoch /= 1000.0

                        dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
                        dt_local = dt_utc.astimezone(monterrey_tz)
                        day_type = "weekend" if dt_local.weekday() >= 5 else "weekday"
                        weekday_number = dt_local.weekday()  # Monday=0, Sunday=6

                        all_data.append({
                            'counter': len(all_data) + 1,
                            'date': dt_local.date().isoformat(),
                            'hour': dt_local.hour,
                            'minute': dt_local.minute,
                            'day_type': day_type,
                            'weekday_number': weekday_number
                          })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    # Return DataFrame (may be empty)
    df = pd.DataFrame(all_data)
    return df




def clean_data(
    df: pd.DataFrame,
    output_file: Optional[str] = "data/processed/df.csv",
    iqr_multiplier: float = 1.5,
    drop_weekends: bool = True,
    fill_gaps: bool = False,
    gap_freq: str = "15min",
) -> pd.DataFrame:
    """
    Clean a DataFrame with explicit NaN handling and outlier removal.

    - Assert the DataFrame is not empty.
    - Handle missing values:
        * For numeric columns: fill NaN with the column median.
        * For non-numeric (categorical/object): fill NaN with the mode if available.
        * For date-like columns named 'date': drop rows with missing dates.
    - Outlier handling (IQR method): for numeric columns, compute Q1/Q3 and IQR
      and remove rows where the value is outside [Q1 - iqr_multiplier*IQR, Q3 + iqr_multiplier*IQR].
      This is a standard, robust method to remove extreme outliers while keeping
      the majority of the data. The multiplier defaults to 1.5 (Tukey's rule).
    - If present, 'hour' is within 0-23, 'minute' within 0-59, 'weekday_number' within 0-6,
      and 'counter' is positive. If values fall outside these ranges prior to cleaning,
      they'll be clipped after NaN imputation.

    Returns a cleaned copy of the DataFrame. If `output_file` is not None,
    the cleaned DataFrame will be written to that CSV path.
    """
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    if df.empty:
        raise AssertionError("Input DataFrame is empty")

    # Required columns for time-based operations
    required = ['date', 'hour', 'minute']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise MissingColumnError(f"Missing required columns: {missing}")

    df_clean = df.copy()

    # Identify columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

    # Basic physical range assertions (if those columns exist)
    if 'hour' in df_clean.columns:
        if not df_clean['hour'].dropna().between(0, 23).all():
            # we'll clip after imputation
            pass
    if 'minute' in df_clean.columns:
        if not df_clean['minute'].dropna().between(0, 59).all():
            pass
    if 'weekday_number' in df_clean.columns:
        if not df_clean['weekday_number'].dropna().between(0, 6).all():
            pass
    if 'counter' in df_clean.columns:
        if not (df_clean['counter'].dropna() > 0).all():
            pass

    # Handle missing values
    # Dates: drop rows where 'date' is missing (we need a date for temporal analysis)
    df_clean = df_clean[~df_clean['date'].isna()]

    for col in numeric_cols:
        median = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median)

    for col in object_cols:
        if df_clean[col].isna().any():
            try:
                mode = df_clean[col].mode().iloc[0]
                df_clean[col] = df_clean[col].fillna(mode)
            except Exception:
                df_clean[col] = df_clean[col].fillna('')

    # Clip columns with known physical ranges
    if 'hour' in df_clean.columns:
        df_clean['hour'] = df_clean['hour'].clip(0, 23).astype(int)
    if 'minute' in df_clean.columns:
        df_clean['minute'] = df_clean['minute'].clip(0, 59).astype(int)
    if 'weekday_number' in df_clean.columns:
        df_clean['weekday_number'] = df_clean['weekday_number'].clip(0, 6).astype(int)
    if 'counter' in df_clean.columns:
        df_clean['counter'] = df_clean['counter'].clip(lower=1).astype(int)

    # Outlier removal using IQR method for numeric columns
    # Keep track of rows to drop (any column flagged as outlier will remove the row)
    if numeric_cols:
        mask = pd.Series(True, index=df_clean.index)
        for col in numeric_cols:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            col_mask = df_clean[col].between(lower, upper)
            mask &= col_mask
        df_clean = df_clean[mask]

    df_clean = df_clean.reset_index(drop=True)

    # Optionally drop weekends
    if drop_weekends and 'day_type' in df_clean.columns:
        df_clean = df_clean[df_clean['day_type'] == 'weekday']

    # If requested, fill gaps by resampling to a regular time grid (e.g., 15 minutes)
    if fill_gaps:
        # build a datetime column
        df_clean['datetime'] = pd.to_datetime(df_clean['date']) + pd.to_timedelta(df_clean['hour'], unit='h') + pd.to_timedelta(df_clean['minute'], unit='m')
        # set index and count occurrences in each period
        s = df_clean.set_index('datetime').groupby(pd.Grouper(freq=gap_freq)).size()
        # create continuous index and fill missing with 0
        if not s.empty:
            full_idx = pd.date_range(start=s.index.min(), end=s.index.max(), freq=gap_freq)
            s = s.reindex(full_idx, fill_value=0)

        # Build filled DataFrame
        filled = pd.DataFrame({'datetime': s.index, 'volume': s.values})
        filled['date'] = filled['datetime'].dt.date.astype(str)
        filled['hour'] = filled['datetime'].dt.hour
        filled['minute'] = filled['datetime'].dt.minute
        filled['weekday_number'] = filled['datetime'].dt.weekday
        filled['day_type'] = np.where(filled['weekday_number'] >= 5, 'weekend', 'weekday')
        filled['counter'] = 0
        df_clean = filled[['counter', 'date', 'hour', 'minute', 'day_type', 'weekday_number', 'volume', 'datetime']]

    # Optionally write cleaned DataFrame to CSV
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df_clean.to_csv(output_path, index=False)
            print(f"Cleaned data written to {output_path}")
        except Exception as e:
            print(f"Failed to write cleaned data to {output_path}: {e}")

    return df_clean


def eda_summary(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> dict:
    """
    Produce an EDA summary dict containing descriptive statistics and six plots: 
    histograms, correlation matrix (heatmap), and scatter pairplot.

    The function returns a dictionary with keys:
    - 'describe': descriptive statistics (from pandas `.describe()` as nested dict)
    - 'n_rows', 'n_cols'
    - 'plots': a dict mapping plot names to PNG byte contents (binary PNG data)

    Notes:
    - The function will call `clean_data` internally to ensure missing values and
      obvious outliers are handled before computing statistics and plots.
    - Pairplot generation can be slow on very large DataFrames; this function
      will sample up to 1000 rows for pairplots to keep it responsive.
    """
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    if df.empty:
        raise AssertionError("Input DataFrame is empty")

    # Use the cleaned dataset (clean_data will also write the CSV by default)
    df_clean = clean_data(df, iqr_multiplier=iqr_multiplier)

    summary = {}
    summary['n_rows'], summary['n_cols'] = df_clean.shape
    # descriptive statistics
    try:
        summary['describe'] = df_clean.describe(include='all').to_dict()
    except Exception:
        summary['describe'] = {}

    # Prepare plots (return matplotlib Figure objects)
    plots = {}

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    # 1) Histograms for numeric columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df_clean[col].dropna(), kde=False, ax=ax)
        ax.set_title(f'Histogram: {col}')
        fig.tight_layout()
        plots[f'histogram_{col}'] = fig

    # 3) Correlation matrix heatmap (numeric cols only)
    if len(numeric_cols) >= 2:
        corr = df_clean[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Correlation matrix')
        fig.tight_layout()
        plots['correlation_matrix'] = fig

    # 4) Scatter pairplot (sample if large)
    if len(numeric_cols) >= 2:
        sample = df_clean[numeric_cols].sample(n=min(len(df_clean), 1000), random_state=0)
        # seaborn pairplot returns a PairGrid; use its figure
        g = sns.pairplot(sample)
        plots['pairplot'] = g.fig

    summary['plots'] = plots
    return summary


def load_data():
    """
    Carga los datos procesados desde data/processed/df.csv y devuelve un DataFrame.
    """
    csv_path = Path("data/processed/df.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo {csv_path}")
    df = pd.read_csv(csv_path)
    return df

if __name__ == "__main__":
    # When executed directly: load data, clean it (which writes the CSV),
    # and print a short summary.
    df = load_data()
    if df.empty:
        print("No raw data files found or no valid records parsed.")
    else:
        cleaned = clean_data(df)
        print(f"Loaded {len(df)} rows; cleaned {len(cleaned)} rows and wrote to disk.")
        summary = eda_summary(cleaned)
        # Save figures and statistics to the analysis directory
        analysis_dir = Path("data/analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save descriptive statistics to JSON
        stats_file = analysis_dir / "descriptive_statistics.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(summary['describe'], f, indent=2, default=str)
            print(f"Saved descriptive statistics to {stats_file}")
        except Exception as e:
            print(f"Failed to save descriptive statistics: {e}")

        for name, fig in summary['plots'].items():
            try:
                fig.savefig(analysis_dir / f"{name}.png")
                print(f"Saved plot to {analysis_dir / f'{name}.png'}")
            except Exception:
                # If the object is not a Figure, attempt to convert or skip
                try:
                    # some seaborn objects may expose .fig
                    fig.fig.savefig(analysis_dir / f"{name}.png")
                    print(f"Saved plot to {analysis_dir / f'{name}.png'}")
                except Exception:
                    print(f"Could not save plot {name}")
