def remove_elements_from_list(lst: list, elements: list):
    """
    remove elements from lst.
    """
    return [e for e in lst if e not in elements]
