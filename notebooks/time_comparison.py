import time

from nlp import preprocess

# I prefer this to Lorem Ipsum.
# Could use timeit instead of this, too...
sentences = """
Whereof one cannot speak, thereof one must be silent. \
The limits of my language mean the limits of my world. \
The world is everything that is the case. \
I don't know why we are, but I am sure that it is not in order to enjoy ourselves. \
Hell isn't other people. Hell is yourself. \
The real question of life following death isn't whether or not it exists, but what 
problem it will solve if it actually does. \
Only describe, don't explain. \
A man will be imprisoned in a room with a door that's unlocked and opens inwards, so 
long as it never occurs to him to pull rather than to push. \
Do not, for heaven's sake be scared of talking nonsense! But you must pay attention to \
your nonsense.
"""

start = time.time()
basic_tokens = preprocess.tokenize_str(sentences, mode='basic')
print(f"""Time for 'basic' to tokenize:\t{time.time() - start}""")
print(type(basic_tokens))

start = time.time()
nltk_tokens = preprocess.tokenize_str(sentences, mode='nltk')
print(f"""Time for 'nltk' to tokenize:\t{time.time() - start}""")
print(type(nltk_tokens))
