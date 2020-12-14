from typing import List

def print_list(list_: List):
    print(list_to_string(list_))

def list_to_string(list_: List, num_elements: int = 3, start: int = 0) \
        -> str:
    if num_elements + start > len(list_):
        raise Exception(f"Cannot print {num_elements} from {start} with list of length {len(list_)}")

    ret = "["

    for i in range(num_elements):
        element = list_[start + i]
        ret += str(element)
        ret += ", "

    ret += "...]"

    return ret

def numerical_list_to_string(list_: List, num_elements: int = 3, start: int = 0, dec_places: int = 4) \
        -> str:
    if num_elements + start > len(list_):
        raise Exception(f"Cannot print {num_elements} from {start} with list of length {len(list_)}")

    ret = "["

    for i in range(num_elements):
        element = list_[start + i]
        element = int(element * (10 ** dec_places)) / (10 ** dec_places)

        ret += str(element)
        ret += ", "

    ret += "...]"

    return ret