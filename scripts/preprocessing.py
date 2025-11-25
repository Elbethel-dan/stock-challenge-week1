import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

class Preprocessor:
    def __init__(self, language="english", remove_numbers=True):
        """
        Text Preprocessor with tokenization, stopword removal,
        punctuation removal, and optional number removal.
        """
        self.language = language
        self.remove_numbers = remove_numbers
        self.punctuation_table = str.maketrans("", "", string.punctuation)

        # Ensure stopwords are downloaded
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words(language))
        except OSError:
            # Handle unsupported language
            print(f"Warning: Stopwords not available for '{language}'. No stopwords will be removed.")
            self.stop_words = set()

    def preprocess_text(self, text):
        """Clean and tokenize a single text string."""

        if pd.isna(text):
            return []

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(self.punctuation_table)

        # Tokenize
        words = word_tokenize(text)

        # Remove stopwords
        if self.stop_words:
            words = [w for w in words if w not in self.stop_words]

        # Optional: remove numbers
        if self.remove_numbers:
            words = [w for w in words if not w.isnumeric()]

        return words

    def apply_to_dataframe(self, df, column='headline', inplace=True):
        """
        Apply preprocessing to a DataFrame column.
        Adds 2 new columns:
            - processed_headline: list of tokens
            - processed_headline_str: joined string
        """

        if not inplace:
            df = df.copy()

        df['processed_headline'] = df[column].apply(self.preprocess_text)
        df['processed_headline_str'] = df['processed_headline'].apply(lambda x: " ".join(x))

        return df
