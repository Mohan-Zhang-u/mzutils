def argparse_bool(s):
    """
    parse the string s to boolean for argparse
    :param s:
    :return: bool or None by default
    example usage: parser.add_argument("--train", help="train (finetune) the model or not", type=mzutils.argparse_bool, default=True)
    """
    if not isinstance(s, str):
        return s
    if s.lower() in ('yes', 'y', 't', 'true', 1):
        return True
    elif s.lower() in ('no', 'n', 'f', 'false', 0):
        return False
    else:
        return None
