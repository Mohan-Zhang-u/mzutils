import random
import sys

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

    >>> import numpy as np
    >>> import mzutils
    >>> a=np.array([4,3,1,2])
    >>> mzutils.argmax_indices(a)
    array([0, 1, 3, 2])
    """
    l = len(one_d_array)
    flat_indices = np.argpartition(one_d_array, l - 1)
    flat_indices = np.flip(flat_indices, axis=0)
    if n is not None:
        return flat_indices[:n]
    return flat_indices


def argmax(l, default_input=0, default_output=0, if_equal='random', decorator_length=1):
    """

    :param l:
    :param default_input:
    :param default_output:
    :param if_equal: can be 'random', 'latter', 'former', 'list', this one decides if there are multiple options,
     which one to return. 'list' means return a list of all possible options.
    :param decorator_length: apply a decorator distribution to pick the peak of this list. For details please consult
    argmax_decorator.
    :return:

    >>> a=[1,2,2,3,2,1,4,4,4,4,5,2,5,5,5,5,5,4,3]
    >>> print(argmax(a, if_equal='latter'))
    16
    >>> print(argmax(a, if_equal='former'))
    10
    >>> print(argmax(a, if_equal='list'))
    [10, 12, 13, 14, 15, 16]
    >>> print(argmax(a, if_equal='list', decorator_length=2))
    [13, 14, 15]
    >>> print(argmax(a, if_equal='list', decorator_length=3))
    [14]
    """
    if decorator_length > 1:
        if decorator_length >= len(l):
            raise Exception("decorator_length should be less than the length of input list.")
        l = argmax_decorator(l, decorator_length)

    max_val = -sys.maxsize
    re_list = [l[0]]
    for i in range(len(l)):
        if l[i] == max_val:
            re_list.append(i)
        if l[i] > max_val:
            re_list = [i]
            max_val = l[i]
    if max_val == default_input:
        return default_output
    if if_equal == 'latter':
        return re_list[-1]
    elif if_equal == 'former':
        return re_list[0]
    elif if_equal == 'random':
        return re_list[random.randint(0, len(re_list) - 1)]
    else:
        return re_list


def argmax_decorator(l, decorator_length):
    fm = l[1:decorator_length]
    fm.reverse()
    lt = l[-decorator_length:-1]
    lt.reverse()
    lpad = fm + l + lt
    ltrans = []
    for i in range(decorator_length - 1, decorator_length - 1 + len(l)):
        sum = lpad[i]
        for j in range(1, decorator_length):
            sum += (lpad[i + j] + lpad[i - j]) / (j + 1)
        ltrans.append(sum)
    return ltrans
