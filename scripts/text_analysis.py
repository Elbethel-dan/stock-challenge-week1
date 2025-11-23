from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class TextAnalysis:
    def __init__(self, df, text_column='processed_headline', publisher_column='publisher', stock_column='stock'):
        """
        Text and publisher analysis class.
        Handles token lists or raw text.
        """
        self.df = df
        self.text_column = text_column
        self.publisher_column = publisher_column
        self.stock_column = stock_column

    # ------------------------------------------------------
    # Utility: normalize text column to string for vectorizer
    # ------------------------------------------------------
    def _to_text(self, tokens):
        if isinstance(tokens, list):
            return " ".join(tokens)
        if isinstance(tokens, str):
            return tokens
        return ""

    # ------------------------------------------------------
    # Top keywords
    # ------------------------------------------------------
    def top_keywords(self, n=20):
        """Return top N most frequent words."""

        all_words = []

        for entry in self.df[self.text_column]:
            if isinstance(entry, list):
                all_words.extend(entry)
            elif isinstance(entry, str):
                all_words.extend(entry.split())

        word_freq = Counter(all_words)
        return word_freq.most_common(n)

    # ------------------------------------------------------
    # N-grams
    # ------------------------------------------------------
    def top_ngrams(self, n=20, ngram_range=(2, 3), language='english'):
        """Return top N n-grams."""

        text_series = self.df[self.text_column].apply(self._to_text)

        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=language
        )
        X = vectorizer.fit_transform(text_series)

        sum_words = X.sum(axis=0)

        ngram_freq = [
            (word, int(sum_words[0, idx]))
            for word, idx in vectorizer.vocabulary_.items()
        ]

        return sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:n]

    # ------------------------------------------------------
    # Publisher counts
    # ------------------------------------------------------
    def publisher_counts(self):
        """Return counts of articles per publisher."""
        return self.df[self.publisher_column].value_counts()

    # ------------------------------------------------------
    # Top publishers and their top stocks
    # ------------------------------------------------------
    def top_publishers_stocks(self, top_n=5, stock_top_n=10):
        """Return a dict of publishers mapped to their most mentioned stocks."""

        top_publishers = self.publisher_counts().head(top_n).index

        output = {}

        for pub in top_publishers:
            subset = self.df[self.df[self.publisher_column] == pub]
            output[pub] = subset[self.stock_column].value_counts().head(stock_top_n)

        return output

    # ------------------------------------------------------
    # Publisher domain extraction (safe)
    # ------------------------------------------------------
    def extract_publisher_domain(self, inplace=False):
        """
        Extract domains from publisher field.
        Example: john@nytimes.com → nytimes.com
        """

        df = self.df if inplace else self.df.copy()

        df["publisher_domain"] = df[self.publisher_column].str.extract(r'@([\w\.-]+)$')

        return df["publisher_domain"].value_counts().head(10)
