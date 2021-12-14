import codecs
import json


def dump_json(dictionary, file_path):
    """
    
    :param dict:
    :param file_path:
    :return:
    """
    with codecs.open(file_path, 'w+', encoding='utf-8') as fp:
        json.dump(dictionary, fp)


def load_json(file_path):
    """

    :param file_path:
    :return: dict object
    """
    with codecs.open(file_path, 'r', encoding='utf-8') as fp:
        return json.load(fp)
