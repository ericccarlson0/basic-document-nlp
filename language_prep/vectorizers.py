from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

class SnowballStemCountVectorizer(CountVectorizer):
    def __init__(self, **kwargs):
        super(SnowballStemCountVectorizer, self).__init__(**kwargs)
        self.stemmer = SnowballStemmer('english', ignore_stopwords=True)

    def build_analyzer(self):
        analyzer = super(SnowballStemCountVectorizer, self).build_analyzer()
        return lambda doc: ([self.stemmer.stem(word) for word in analyzer(doc)])

class LemmatizerCountVectorizer(CountVectorizer):
    def __init__(self, **kwargs):
        super(LemmatizerCountVectorizer, self).__init__(**kwargs)
        self.lemmatizer = WordNetLemmatizer()

    def build_analyzer(self):
        analyzer = super(LemmatizerCountVectorizer, self).build_analyzer()
        return lambda doc: ([self.lemmatizer.lemmatize(word) for word in analyzer(doc)])