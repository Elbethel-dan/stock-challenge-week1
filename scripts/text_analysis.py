from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


class TextAnalyzer:

    def __init__(self, df: pd.DataFrame):
        """
        :param df: The Pandas DataFrame containing the text data.
        """
        self.df = df

    def _validate_column(self, text_column: str):
        """Ensure the provided text column exists in the DataFrame."""
        if text_column not in self.df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. Available columns: {list(self.df.columns)}"
            )

    def _to_text(self, tokens):
        """Utility: normalize text column entry to plain string for vectorizer."""
        if isinstance(tokens, list):
            return " ".join(tokens)
        if isinstance(tokens, str):
            return tokens
        return ""

    def top_keywords(self, n: int = 20, text_column: str = None):
        """
        Return top N most frequent single words (unigrams).

        :param n: Number of keywords to return.
        :param text_column: Column containing text (string or list of tokens).
        """
        self._validate_column(text_column)

        all_words = []
        for entry in self.df[text_column]:
            if isinstance(entry, list):
                all_words.extend(entry)
            elif isinstance(entry, str):
                all_words.extend(entry.split())

        word_freq = Counter(all_words)
        return word_freq.most_common(n)

    def top_ngrams(
        self,
        n: int = 20,
        text_column: str = None,
        ngram_range: tuple = (2, 3),
        language: str = "english"
    ):
        """
        Return top N n-grams (multi-word expressions).

        :param n: Number of n-grams to return.
        :param text_column: Column containing text.
        :param ngram_range: (min_n, max_n), e.g., (2, 3).
        :param language: Stopword language.
        """
        self._validate_column(text_column)

        text_series = self.df[text_column].apply(self._to_text)

        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=language
        )
        X = vectorizer.fit_transform(text_series)

        sum_words = X.sum(axis=0)

        ngram_freq = [
            (ngram, int(sum_words[0, idx]))
            for ngram, idx in vectorizer.vocabulary_.items()
        ]

        return sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:n]
