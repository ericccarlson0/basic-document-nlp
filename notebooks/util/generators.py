import sys

def generator_with_max(source, max_count: int = sys.maxsize):
    count = 0
    while count < max_count:
        yield next(source)
        count += 1