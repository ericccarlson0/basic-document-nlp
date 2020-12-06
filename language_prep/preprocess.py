import re
import nltk
import spacy

# This doc is probably one of the most disorganized here. The decision must be made as to whether or not
# I want to do pre-processing myself, in order to use tools such as sklearn, or to leave it to tools such
# as spaCy.

# see https://arxiv.org/pdf/1204.0191.pdf for ideas on correcting OCR errors.

alpha_regex = re.compile('[\W_]+')
non_ws_regex = re.compile('[\S]+')

redundant_pos = ['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'RP', 'TO']
# excluded 'PRP (personal pronouns), PRP$ (possessive pronouns), WDT, WP, WRB (wh-determiner, pronoun, adverb)
np_grammar = r"NP: {<DT>?<JJ>*<NN>}"
vp_grammar = r"VP: {<NP><V><RB|RBR|RBS>}"

# Use words that are in the GloVe dataset.
words = set()
def load_glove_words():
    glove_dir = "/Users/ericcarlson/Desktop/Personal Projects/basic-document-nlp/resources/glove.42B.300d.txt"

    print("In the process of loading all words used in GloVe embeddings...")

    with open(glove_dir) as glove_file:
        for line in glove_file:
            words.add(line.split()[0])

    print("Done.")

def tokenize_str(sentences: str, mode: str = 'nltk'):
    if mode == 'basic':
        tokens = [to_alpha_lower(word) for word in sentences.split()]
    elif mode == 'nltk':
        tokens = nltk.word_tokenize(sentences)
    else:
        raise ValueError(f"""Mode must be 'nltk' or 'basic' but was {mode}...""")

    return tokens

def to_alpha(word: str):
    return alpha_regex.sub("", word)

def to_alpha_lower(word: str):
    return alpha_regex.sub("", word).lower()


def is_word(word: str):
    if len(words) == 0:
        load_glove_words()

    return word in words

# TODO: How to deal with cases such as proper nouns?
def to_cleaned_str(doc):
    result = []
    for token in doc:
        if token.is_space:
            result.append(" ")
        else:
            word = token.text_with_ws.lower()

            if not is_word(word.strip()):
                result.append(
                    non_ws_regex.sub("_", word)
                )
            else:
                result.append(word)

    return "".join(result)

# Uses NLTK.
# TODO: unfinished
def is_stopword(word: str):
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

