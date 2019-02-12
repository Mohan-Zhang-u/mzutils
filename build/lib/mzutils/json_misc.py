import codecs
import json


def dump_config(file_path, dict):
    """

    :param file_path:
    :param dict:
    :return:
    """
    with codecs.open(file_path, 'w+', encoding='utf-8') as fp:
        json.dump(dict, fp)


def load_config(file_path):
    """

    :param file_path:
    :return: dict object
    """
    with codecs.open(file_path, 'r', encoding='utf-8') as fp:
        return json.load(fp)
