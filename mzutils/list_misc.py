def remove_elements_from_list(lst: list, elements: list):
    """
    remove elements from lst.
    """
    return [e for e in lst if e not in elements]


def pop_indices(lst, indices):
    """
    pop the lst given a list or tuple of indices.
    this function modifies lst directly inplace.
    >>> pop_indices([1,2,3,4,5,6], [0,4,5])
    >>> [2, 3, 4]
    """
    for n in sorted(indices, reverse=True):
        lst.pop(n)
    return lst


def pad_list(lst: list, length: int, element=''):
    """
    pad a list to length with elements.
    returned list will have length equal to length.
    """
    if len(lst) >= length:
        return lst[:length]

    return lst + [element for _ in range(length - len(lst))]


def split_list_with_len(lst: list, length: int):
    """
    return a list of sublists with len == length (except the last one)
    """
    return [lst[i:i + length] for i in range(0, len(lst), length)]


def flatten(lst):
    """
    convert nested list to a list of elements.
    """
    if type(lst) == list:
        result = []
        for i in lst:
            result += flatten(i)
        return result
    else:
        return [lst]
