from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

class TextAnalysis:
    def init(self, df, text_column='processed_headline', publisher_column='publisher'):
        self.df = df
        self.text_column = text_column
        self.publisher_column = publisher_column

    def top_keywords(self, n=20):
        all_words = [word for tokens in self.df[self.text_column] for word in tokens]
        word_freq = Counter(all_words)
        return word_freq.most_common(n)

    def top_ngrams(self, n=20, ngram_range=(2,3)):
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
        X = vectorizer.fit_transform(self.df[self.text_column].apply(lambda x: ' '.join(x)))
        sum_words = X.sum(axis=0)
        ngrams_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        return sorted(ngrams_freq, key=lambda x: x[1], reverse=True)[:n]

    def publisher_counts(self):
        return self.df[self.publisher_column].value_counts()

    def top_publishers_stocks(self, top_n=5):
        top_publishers = self.publisher_counts().head(top_n).index
        result = {}
        for pub in top_publishers:
            result[pub] = self.df[self.df[self.publisher_column] == pub]['stock'].value_counts().head(10)
        return result

    def extract_publisher_domain(self):
        self.df['publisher_domain'] = self.df[self.publisher_column].str.extract(r'@([\w\.-]+)$')
        return self.df['publisher_domain'].value_counts().head(10)