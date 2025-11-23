import matplotlib.pyplot as plt
import pandas as pd

class Plotting:
    def __init__(self, df=None):
        """
        If df is provided, instance methods can use it.
        Static methods allow external df usage.
        """
        self.df = df

    # -----------------------------
    # Instance Methods (use self.df)
    # -----------------------------
    def headline_length_hist(self, column='headline_len_words'):
        """Histogram of headline length."""
        plt.hist(self.df[column], bins=30)
        plt.xlabel("Headline Length (words)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Headline Lengths")
        plt.show()

    def weekday_trends(self, date_column='date', headline_column='headline'):
        """Plot article count per weekday."""
        weekday_counts = (
            self.df.groupby(self.df[date_column].dt.day_name())[headline_column]
            .count()
            .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        )

        plt.figure(figsize=(8,5))
        weekday_counts.plot(kind='bar')
        plt.title("Articles Published by Weekday")
        plt.xlabel("Weekday")
        plt.ylabel("Number of Articles")
        plt.show()

    def hourly_trends(self, date_column='date', headline_column='headline'):
        """Plot article count per hour of the day."""
        hour_counts = self.df.groupby(self.df[date_column].dt.hour)[headline_column].count()

        plt.figure(figsize=(8,5))
        hour_counts.plot(kind='bar')
        plt.title("Articles Published by Hour of Day")
        plt.xlabel("Hour")
        plt.ylabel("Number of Articles")
        plt.show()

    # -----------------------------
    # Static Methods (take df arg)
    # -----------------------------
    @staticmethod
    def boxplot_column(df, column, title="Box Plot"):
        """Boxplot for a numerical column."""
        plt.figure(figsize=(8, 5))
        plt.boxplot(df[column].dropna())
        plt.title(title)
        plt.ylabel(column)
        plt.show()

    @staticmethod
    def plot_yearly_publication_counts(df, date_column="date"):
        """Plot number of articles per year (does NOT modify df)."""

        # Convert to datetime without modifying df
        dates = pd.to_datetime(df[date_column], errors="coerce")

        # Extract years
        years = dates.dt.year
        yearly_counts = years.value_counts().sort_index()

        # Print counts
        for year, count in yearly_counts.items():
            print(f"{year}: {count} articles")

        # Plot
        plt.figure(figsize=(10, 5))
        yearly_counts.plot(kind="bar")
        plt.title("Yearly Publication Counts")
        plt.xlabel("Year")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_publisher_domains(df, top_n=10):
        """Plot top publisher domains."""
        df = df.copy()

        df['publisher_domain'] = df['publisher'].str.extract(r'@([\w\.-]+)$')
        domain_counts = df['publisher_domain'].value_counts().head(top_n)

        plt.figure(figsize=(10, 5))
        domain_counts.plot(kind='bar')
        plt.title(f"Top {top_n} Publisher Domains")
        plt.xlabel("Domain")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_top_publishers(df, top_n=10):
        """Plot top publishers by count."""
        publisher_counts = df['publisher'].value_counts().head(top_n)

        plt.figure(figsize=(10,5))
        publisher_counts.plot(kind='bar')
        plt.title(f"Top {top_n} Publishers")
        plt.xlabel("Publisher")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.show()
