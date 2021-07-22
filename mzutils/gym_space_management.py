import numpy as np


def normalize_spaces(space, max_space=None, min_space=None, skip_columns=None):
    """
    normalize each column of observation/action space to be in [-1,1] such that it looks like a Box
    space can be the whole original space (X by D) or just one row in the original space (D,)
    :param space: numpy array
    :param max_space: numpy array, the maximum value of each column of the space, normally
        we would get this from reading the dataset or prior knowledge
    :param min_space: numpy array, the minimum value of each column of the space, normally
        we would get this from reading the dataset or prior knowledge
    :param skip_columns: numpy array or list, columns to skip from normalization
    """
    assert not isinstance(space, list)
    if max_space is None:
        max_space = space.max(axis=0)
    if min_space is None:
        min_space = space.min(axis=0)
    gap = max_space - min_space
    gap += 1e-8 # to avoid div by 0
    full_sum = max_space + min_space
    re_space = (2 * space - full_sum) / gap
    if skip_columns is not None:
        if len(space.shape) == 1:
            re_space[skip_columns] = space[skip_columns]
        else:
            re_space[:, skip_columns] = space[:, skip_columns]
    return re_space, max_space, min_space


def denormalize_spaces(space_normalized, max_space=None, min_space=None, skip_columns=None):
    """
    same as above, and space_normalized can be the whole normalized original space or just one row in the normalized space
    """
    assert not isinstance(space_normalized, list)
    if max_space is None:
        max_space = space_normalized.max(axis=0)
    if min_space is None:
        min_space = space_normalized.min(axis=0)
    gap = max_space - min_space
    gap += 1e-8 # to avoid div by 0
    full_sum = max_space + min_space
    re_space = (space_normalized * gap + full_sum) / 2
    if skip_columns is not None:
        if len(space_normalized.shape) == 1:
            re_space[skip_columns] = space_normalized[skip_columns]
        else:
            re_space[:, skip_columns] = space_normalized[:, skip_columns]
    return re_space, max_space, min_space


def list_of_str_to_numpy_onehot_dict(lst):
    """
    create a onehot lookup dictionary according to the list of strings passed in
    """
    re_dict = {}
    for i, key in enumerate(lst):
        onehot_vec = np.zeros(len(lst), dtype=np.float64)
        onehot_vec[i] = 1.0
        re_dict[key] = onehot_vec
    return re_dict
