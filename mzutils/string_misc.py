import ast
import codecs
import pickle

import nltk


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
    with codecs.open(filepath, 'rb') as fp:
        itemlist = pickle.load(fp)
    return itemlist


def str_rep_to_list(s):
    """
    convert a string representation of list to a python list object.
    :param s:
    :return:
    """
    return ast.literal_eval(s)


# def string_segementor_on_word_length(content, max_length, language='english'):
#     """
#     segment a long document to several small documents based on the nltk tokenized word length.
#     sentence structure will be kept.
#     :param content: content to be segmented by world length, with complete sentences.
#     :param max_length: document segments' max length.
#     :param language: for the use of nltk, default english.
#     :return: a list of segmented contents
#     """
#     contents = []
#     sentences = nltk.sent_tokenize(content, language)
#     i = 0
#     word_count = 0
#     document = ""
#     while i < len(sentences):
#         sentence = sentences[i]
#         current_count = len(nltk.word_tokenize(sentence, language))
#         if word_count + current_count < max_length:
#             document = document + sentence + " "
#             word_count = word_count + current_count
#             i = i + 1
#         else:
#             contents.append(document)
#             word_count = 0
#             document = ""
#     contents.append(document)
#     return contents


def chinese_document_segementor_on_word_length(content, max_length):
    """
    segment a long document to several small documents based on the nltk tokenized word length.
    sentence structure will be kept.
    :param content: content to be segmented by world length, with complete sentences.
    :param max_length: document segments' max length.
    :return: a list of segmented contents
    """
    contents = []
    sentences = chinese_sent_tokenize(content, max_length)
    i = 0
    word_count = 0
    document = ""
    while i < len(sentences):
        sentence = sentences[i]
        current_count = len(sentence)
        if word_count + current_count < max_length:
            document = document + sentence + " "
            word_count = word_count + current_count
            i = i + 1
        else:
            contents.append(document)
            word_count = 0
            document = ""
    contents.append(document)
    return contents


def chinese_sent_tokenize(content, max_length):
    """
    a Chinese sentence tokenizer to solve nltk.sent_tokenize bugs mentioned here: https://github.com/nltk/nltk/issues/1824
    :param content:
    :return:
    """
    security_number = int(max_length*2/3)
    sentences = []
    length = len(content)
    idx = 0
    for i in range(length):
        if content[i] in '？！。；？！；。?!.;\r\n':
            sentence = content[idx:i + 1].lstrip().rstrip()
            if len(sentence)>=max_length:
                sentence_list = [sentence[i:i + security_number] for i in range(0, len(sentence), security_number)]
                sentences = sentences + sentence_list
            else:
                sentences.append(sentence)
            idx = i + 1
    if content[idx:] != "":
        sentences.append(content[idx:].lstrip().rstrip())
    return sentences
