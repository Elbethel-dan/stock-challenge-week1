import matplotlib.pyplot as plt
import pandas as pd
class Plotting:
    def init(self, df):
        self.df = df

    def headline_length_hist(self, column='headline_len_words'):
        plt.hist(self.df[column], bins=30)
        plt.xlabel("Headline Length (words)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Headline Lengths")
        plt.show()

    def boxplot_column(df, column, title="Box Plot"):
        plt.figure(figsize=(8,5))
        plt.boxplot(df[column].dropna())
        plt.title(title)
        plt.ylabel(column)
        plt.show()

    def plot_yearly_publication_counts(df, date_column="date"):
        """
        Plots the number of articles published per year.
        """

        # Ensure the date column is datetime
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        # Extract year
        df["year"] = df[date_column].dt.year

        # Count articles per year
        yearly_counts = df["year"].value_counts().sort_index()

        # Print values
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

    def weekday_trends(self, date_column='date', headline_column='headline'):
        weekday_counts = self.df.groupby(self.df[date_column].dt.day_name())[headline_column].count()
        weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        weekday_counts = weekday_counts.reindex(weekday_order)
        plt.figure(figsize=(8,5))
        weekday_counts.plot(kind='bar', color='skyblue')
        plt.title("Articles Published by Weekday")
        plt.xlabel("Weekday")
        plt.ylabel("Number of Articles")
        plt.show()

    def hourly_trends(self, date_column='date', headline_column='headline'):
        hour_counts = self.df.groupby(self.df[date_column].dt.hour)[headline_column].count()
        plt.figure(figsize=(8,5))
        hour_counts.plot(kind='bar', color='orange')
        plt.title("Articles Published by Hour of Day")
        plt.xlabel("Hour")
        plt.ylabel("Number of Articles")
        plt.show()


    def plot_publisher_domains(df, top_n=10):

        df['publisher_domain'] = df['publisher'].str.extract(r'@([\w\.-]+)$')

         # Count by domain
        domain_counts = df['publisher_domain'].value_counts().head(top_n)

    # Plot
        plt.figure(figsize=(10, 5))
        domain_counts.plot(kind='bar', color='green')
        plt.title(f"Top {top_n} Publisher Domains by Number of Articles")
        plt.xlabel("Domain")
        plt.ylabel("Number of Articles")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


    def plot_top_publishers(df, top_n=10):

        publisher_counts = df['publisher'].value_counts().head(top_n)

        plt.figure(figsize=(10,5))
        publisher_counts.plot(kind='bar')
        plt.title(f"Top {top_n} Publishers")
        plt.xlabel("Publisher")
        plt.ylabel("Number of Articles")
        plt.tight_layout()
        plt.show()