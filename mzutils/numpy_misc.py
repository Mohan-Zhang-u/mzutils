import numpy as np


# source: https://stackoverflow.com/questions/5807047/efficient-way-to-take-the-minimum-maximum-n-values-and-indices-from-a-matrix-usi

def argmin_indices(one_d_array, n=None):
    """
    for a 1-d numpy array, returns the indices of elements from min to max.
    If n is provided, it returns only the indices of the smallest n elements.
    :param one_d_array:
    :param n:
    :return:
    """
    l = len(one_d_array)
    flat_indices = np.argpartition(one_d_array, l - 1)
    if n is not None:
        return flat_indices[:n]
    return flat_indices


def argmax_indices(one_d_array, n=None):
    """
    for a 1-d numpy array, returns the indices of elements from max to min.
    If n is provided, it returns only the indices of the largest n elements.
    :param one_d_array:
    :param n:
    :return:
    """
    l = len(one_d_array)
    flat_indices = np.argpartition(one_d_array, l - 1)
    flat_indices = np.flip(flat_indices, axis=0)
    if n is not None:
        return flat_indices[:n]
    return flat_indices


