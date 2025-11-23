import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None          # Raw dataframe
        self.df_clean = None    # Cleaned dataframe

    # -------------------------------
    # LOAD
    # -------------------------------
    def load_csv(self):
        """Load CSV into self.df."""
        self.df = pd.read_csv(self.filepath)
        self.df_clean = self.df.copy()
        return self

    # -------------------------------
    # COLUMN OPERATIONS
    # -------------------------------
    def drop_columns(self, columns):
        """Drop columns from df_clean."""
        if self.df_clean is not None:
            self.df_clean = self.df_clean.drop(columns=columns, errors="ignore")
        return self

    def add_headline_length(self, column='headline', new_column='headline_len_words'):
        """Add a word count column for any text column."""
        if self.df_clean is not None and column in self.df_clean:
            self.df_clean[new_column] = self.df_clean[column].astype(str).str.split().str.len()
        return self

    # -------------------------------
    # OUTLIER REMOVAL
    # -------------------------------
    def remove_outliers(self, column, method="iqr", z_thresh=3.0):
        """
        Remove outliers using IQR or Z-score.
        Stores the cleaned dataframe in df_clean.
        """

        if self.df_clean is None or column not in self.df_clean:
            return self

        series = self.df_clean[column]

        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (series >= lower) & (series <= upper)

        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            z = (series - mean) / std
            mask = z.abs() <= z_thresh

        else:
            raise ValueError("method must be 'iqr' or 'zscore'")

        self.df_clean = self.df_clean[mask]

        return self

    # -------------------------------
    # DATE PARSING
    # -------------------------------
    def parse_dates(self, columns=["date"], timezone="America/New_York"):
        """
        Parse date columns and convert to timezone.
        """
        if self.df_clean is None:
            return self

        for col in columns:
            if col in self.df_clean:
                # Convert to UTC
                self.df_clean[col] = pd.to_datetime(
                    self.df_clean[col], errors="coerce", utc=True
                )
                # Convert timezone
                self.df_clean[col] = self.df_clean[col].dt.tz_convert(timezone)

        return self

    # -------------------------------
    # GETTERS
    # -------------------------------
    def get_raw(self):
        """Return raw dataframe."""
        return self.df

    def get_clean(self):
        """Return cleaned dataframe."""
        return self.df_clean
