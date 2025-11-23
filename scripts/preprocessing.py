import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Preprocessor:
    def init(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        if pd.isna(text):
            return []
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text)
        words = [word for word in words if word not in self.stop_words]
        return words

    def apply_to_dataframe(self, df, column='headline'):
        df['processed_headline'] = df[column].apply(self.preprocess_text)
        df['processed_headline_str'] = df['processed_headline'].apply(lambda x: ' '.join(x))
        return df