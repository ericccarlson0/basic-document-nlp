import re
import nltk

from typing import List

# This doc is probably one of the most disorganized here. The decision must be made as to whether or not
# I want to do pre-processing myself, in order to use tools such as sklearn, or to leave it to tools such
# as spaCy.

# see https://arxiv.org/pdf/1204.0191.pdf for ideas on correcting OCR errors.

alpha_regex = re.compile('[\W_]+')
redundant_pos = ['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'RP', 'TO']
# excluded 'PRP (personal pronouns), PRP$ (possessive pronouns), WDT, WP, WRB (wh-determiner, pronoun, adverb)
np_grammar = r"NP: {<DT>?<JJ>*<NN>}"
vp_grammar = r"VP: {<NP><V><RB|RBR|RBS>}"

# It is generally a good practice to use third-party modules, but the 'basic' mode here
# performs 10-100 times faster than the 'nltk' mode...
def tokenize_str(sentences: str, mode: str = 'nltk'):
    if mode == 'basic':
        tokens = [to_lowercase(word) for word in sentences.split()]
    elif mode == 'nltk':
        tokens = nltk.word_tokenize(sentences)
    else:
        raise ValueError(f"""mode must be 'nltk' or 'basic' but was {mode}""")

    return tokens

def to_alphanumeric(word: str):
    return alpha_regex.sub('', word)

def to_lowercase(word: str):
    return alpha_regex.sub('', word).lower()

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

# TODO: implement
def generate_wordnet(word: str):
    pass

# TODO: implement
def generate_bow(text: str):
    pass

