import nltk

from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

class SnowballStemCountVectorizer(CountVectorizer):
    def __init__(self):
        super(CountVectorizer, self).__init__()
        self.stemmer = SnowballStemmer('english', ignore_stopwords=True)

    def build_analyzer(self):
        analyzer = super(SnowballStemCountVectorizer, self).build_analyzer()
        return lambda doc: ([self.stemmer.stem(w) for w in analyzer(doc)])