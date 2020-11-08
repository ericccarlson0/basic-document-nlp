import re
import nltk

from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet

# see https://arxiv.org/pdf/1204.0191.pdf for inspiration on correcting OCR errors.

alphanumeric_re = re.compile('[\W_]+')
redundant_pos = ['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'RP', 'TO']
# excluded 'PRP (personal pronouns), PRP$ (possessive pronouns), WDT, WP, WRB (wh-determiner, pronoun, adverb)
np_grammar = r"NP: {<DT>?<JJ>*<NN>}"
vp_grammar = r"VP: {<NP><V><RB|RBR|RBS>}"


# It is generally a good practice to use third-party modules -- but the 'basic' mode here
# performs 10-100 times faster than the 'nltk' mode.
def tokenize_str(sentences: str, mode: str = 'nltk'):
    if mode == 'basic':
        tokens = [to_lowercase(word) for word in sentences.split()]
    elif mode == 'nltk':
        tokens = nltk.word_tokenize(sentences)
    else:
        raise ValueError(f"""mode must be 'nltk' or 'basic' but was {mode}""")

    return tokens

def to_alphanumeric(word: str):
    return alphanumeric_re.sub('', word)

def to_lowercase(word: str):
    return alphanumeric_re.sub('', word).lower()

def should_filter_pos(word: str):
    tag = nltk.pos_tag([word])
    tag = tag[0][1]
    if tag in redundant_pos:
        return True

    return False

# NP, VP, ADJP, ADVP, PP
# S <- NP, VP
# NP <- DT, NN, PP? (pronoun), PN? (proper name)
# VP <- V (NP) (PP) (ADV)
# PP <- PP (NP)
# AP <- ADJ (PP)
def parse_n_phrase(tree):
    noun_parser = nltk.RegexpParser(grammar=np_grammar)
    parsed = noun_parser.parse(tree)

    # TODO: more processing?
    return parsed

def parse_v_phrase(tree):
    verb_parser = nltk.RegexpParser(grammar=vp_grammar)
    parsed = verb_parser.parse(tree)

    # TODO: more processing?
    return parsed

# TODO: rename.
# We can generate hypernymns, hyponyms, synonyms, and antonyms from the synset.
# We can use #wup_similarity.
def generate_wordnet(word: str):
    pass

# TODO: would we want to bass this CountVectorizer in?
def generate_bow(sentences: List):
    cv = CountVectorizer()
    bag = cv.fit_transform(sentences).toarray()

    return bag

# stemming -> differentiating between versions of the same word
# chunking -> phrases <- words (with PoS)
