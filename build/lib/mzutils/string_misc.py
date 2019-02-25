import nltk
import pickle
import codecs

def add_spaces_between_special_characters(InputText):
    """
    :param InputText:
    :return:
    """
    regular = "1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM "
    b = set(list(InputText))
    spec = []
    for i in b:
        if i not in regular:
            spec.append(i)
    for i in spec:
        InputText = InputText.replace(i, ' ' + i + ' ')
    return InputText


def select_first_sentence(InputText, language='english'):
    return nltk.tokenize.sent_tokenize(InputText, language)[0]


def py_serialize(filepath, itemlist):
    """
    serialize python objects using pickle
    :param filepath:
    :param itemlist:
    :return:
    """
    with codecs.open(filepath, "wb+") as fp:
        pickle.dump(itemlist, fp)


def py_deserialize(filepath):
    """

    :param filepath:
    :return:
    """
    itemlist = None
    with codecs.open('filepath', 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist
