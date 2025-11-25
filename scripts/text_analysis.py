from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class TextAnalyzer:
    """
    Core reusable class for analyzing text content (keywords and n-grams).
    Handles token lists or raw text strings.
    """
    def __init__(self, df: pd.DataFrame, text_column: str = 'text'):
        """
        :param df: The Pandas DataFrame containing the text data.
        :param text_column: The name of the column containing the text (str or list of tokens).
        """
        self.df = df
        self.text_column = text_column

    def _to_text(self, tokens):
        """Utility: normalize text column to a single string for vectorizer."""
        if isinstance(tokens, list):
            return " ".join(tokens)
        if isinstance(tokens, str):
            return tokens
        return ""

    def top_keywords(self, n: int = 20):
        """Return top N most frequent single words (unigrams)."""

        all_words = []
        for entry in self.df[self.text_column]:
            if isinstance(entry, list):
                all_words.extend(entry)
            elif isinstance(entry, str):
                all_words.extend(entry.split())

        word_freq = Counter(all_words)
        return word_freq.most_common(n)

    def top_ngrams(self, n: int = 20, ngram_range: tuple = (2, 3), language: str = 'english'):
        """Return top N n-grams (phrases) for the specified range."""

        text_series = self.df[self.text_column].apply(self._to_text)

        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=language
        )
        X = vectorizer.fit_transform(text_series)

        sum_words = X.sum(axis=0)

        # Map words back to their counts
        ngram_freq = [
            (word, int(sum_words[0, idx]))
            for word, idx in vectorizer.vocabulary_.items()
        ]

        return sorted(ngram_freq, key=lambda x: x[1], reverse=True)[:n]