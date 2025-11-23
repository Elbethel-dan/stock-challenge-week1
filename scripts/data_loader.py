import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_clean = None

    def load_csv(self):
        self.df = pd.read_csv(self.filepath)
        return self.df

    def drop_columns(self, columns):
        self.df = self.df.drop(columns=columns)

    def add_headline_length(self, column='headline'):
        self.df['headline_len_words'] = self.df[column].str.split().str.len()

    def remove_outliers(self, column='headline_len_words'):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        self.df_clean = self.df[(self.df[column] >= lower) & (self.df[column] <= upper)]
        return self.df_clean

    def parse_dates(self, columns=["date"], timezone="America/New_York"):
  
        for column in columns:
            # Convert to datetime (coerce invalid values)
            self.df_clean[column] = pd.to_datetime(
                self.df_clean[column],
                errors="coerce",
                utc=True
        )

        # Convert to desired timezone
            self.df_clean[column] = self.df_clean[column].dt.tz_convert(timezone)