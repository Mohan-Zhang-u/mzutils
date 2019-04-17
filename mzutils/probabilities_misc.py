import math
import numpy as np
import mzutils.numpy_misc


def permutation(a, b):
    """
    Calculate the number of permutations from a choose b
    """
    if a > b:
        return int(math.factorial(a) / math.factorial(a - b))
    raise Exception("a should be less than b")


def binomial_coefficient(a, b):
    if a > b:
        return int(math.factorial(a) / math.factorial(a - b) / math.factorial(b))
    raise Exception("a should be less than b")


def vote_for_cluster(elements, vote_metric, top_n=1, default_element=None):
    """
    provided with a metric that compares two elements, this function retrieves the cluster centre
    (the top n elements in the cluster centre with highest sum of scores after cross compared)
    $$\Theta(n^2)$$
    :param elements: a list of elements.
    :param vote_metric: a metric function that compares two elements at a time. The higher means the similar.
    The range of this function should be from zero to one.
    :param top_n: number of elements to be retrieved.
    :param default_element: if the computed matrix is a null matrix, what to return.
    :return: list of (element, score) tuples.

    >>> #e.g. compare the similarity of two numbers.
    >>> def vote_metric(a, b):
    >>>     return pow(math.e, -abs(a-b))

    """
    length = len(elements)
    matrix = np.zeros((length, length), dtype=float)

    if top_n > length:
        top_n = length

    for i in range(length):
        matrix[i, i] = vote_metric(elements[i], elements[i])
    for i in range(length):
        for j in range(i + 1, length):
            matrix[i, j] = vote_metric(elements[i], elements[j])
            matrix[j, i] = vote_metric(elements[j], elements[i])
    voted_weights = matrix.sum(axis=0)
    max_weights_indices = mzutils.numpy_misc.argmax_indices(voted_weights, top_n)
    max_weights_values = np.take(voted_weights, max_weights_indices)
    re_list = []
    for i in range(top_n):
        re_list.append((elements[max_weights_indices[i]], max_weights_values[i]))
    return re_list
