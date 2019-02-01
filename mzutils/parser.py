def argparse_bool(s):
    """
    parse the string s to boolean for argparse
    :param s:
    :return: bool
    """
    if s.lower() in ('yes', 'y', 't', 'true', 1):
        return True
    elif s.lower() in ('no', 'n', 'f', 'false', 0):
        return False
    else:
        raise Exception("input should be a boolean!")
