import ast
import codecs
import pickle
import re

import nltk


def replace_nth_occur(InputText, old, new, n=0, option='only nth'):
    """
    https://stackoverflow.com/questions/35091557/replace-nth-occurrence-of-substring-in-string
    n count from 0.
    This function replaces occurrences of string 'old' with string 'new'.
    There are three types of replacement of string 'old':
    1) 'only nth' replaces only nth occurrence (default).
    2) 'all left' replaces nth occurrence and all occurrences to the left.
    3) 'all right' replaces nth occurrence and all occurrences to the right.
    if n out of bound, do nothing.
    """
    n = n + 1
    if option == 'only nth':
        left_join = old
        right_join = old
    elif option == 'all left':
        left_join = new
        right_join = old
    elif option == 'all right':
        left_join = old
        right_join = new
    else:
        print("Invalid option. Please choose from: 'only nth' (default), 'all left' or 'all right'")
        return None
    groups = InputText.split(old)
    if groups[:n] != [] and groups[n:] != []:
        nth_split = [left_join.join(groups[:n]), right_join.join(groups[n:])]
        return new.join(nth_split)
    return InputText


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


def string_segementor_on_word_length(content, max_length, language='english'):
    """
    segment a long document to several small documents based on the nltk tokenized word length.
    sentence structure will be kept.
    :param content: content to be segmented by world length, with complete sentences.
    :param max_length: document segments' max length.
    :param language: for the use of nltk, default english.
    :return: a list of segmented contents
    """
    contents = []
    sentences = nltk.sent_tokenize(content, language)
    i = 0
    word_count = 0
    document = ""
    while i < len(sentences):
        sentence = sentences[i]
        current_count = len(nltk.word_tokenize(sentence, language))
        if current_count >= max_length:
            document = document + sentence + " "
            contents.append(document)
            word_count = 0
            document = ""
            i = i + 1
            print("Warning: " + "there is a sentence with word length " + str(
                current_count) + " , but the maximum document length is " + str(max_length))
            continue
            # raise Exception("there is a sentence with word length " + str(current_count) + " , but the maximum document length is " + str(max_length))
        if word_count + current_count >= max_length:
            contents.append(document)
            word_count = 0
            document = ""
        else:
            document = document + sentence + " "
            word_count = word_count + current_count
            i = i + 1
    contents.append(document)
    return contents


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
            if len(contents) > 16000:
                raise Exception(str(contents[-20:]))
    contents.append(document)
    return contents


def chinese_sent_tokenize(content, max_length):
    """
    a Chinese sentence tokenizer to solve nltk.sent_tokenize bugs mentioned here: https://github.com/nltk/nltk/issues/1824
    :param content:
    :return:
    """
    content = content.lstrip().rstrip() + "."
    security_number = int(max_length * 2 / 3)
    sentences = []
    length = len(content)
    idx = 0
    for i in range(length):
        if content[i] in '？！。；？！；。?!.;\r\n':
            sentence = content[idx:i + 1].lstrip().rstrip()
            if len(sentence) >= max_length:
                sentence_list = [sentence[i:i + security_number] for i in range(0, len(sentence), security_number)]
                sentences = sentences + sentence_list
            elif len(sentence) < 5:
                continue
            else:
                sentences.append(sentence)
            idx = i + 1
    return sentences


def detag(sentences, tags):
    """
    remove all tag strings out of all sentences
    :param sentences: a list of string sentences
    :param sentences: a list of string tags
    :return: 
    """
    re_sentences = []
    for sentence in sentences:
        re_sent = sentence
        for tag in tags:
            re_sent = re_sent.replace(tag, "")
        re_sentences.append(re_sent)
    return re_sentences


def file_detag(file_path, tags):
    """
    remove all tag strings from the file
    :param file_path: file path
    :param sentences: a list of string tags
    :return: 
    """
    rfp = codecs.open(file_path, 'r')
    sentences = rfp.readlines()
    rfp.close()
    sentences = detag(sentences, tags)
    with open(file_path, 'w') as wfp:
        wfp.writelines(sentences)


def char_in_language(in_char, expected=None):
    """
    :param in_char: single utf-8 input char
    :param expected: whether the char is in expected range or not.
    :return: 'num' (numbers), 'alphabet' (english alphabets), 'ascii' (ascii strings) the lanuge code. If expected range is provided, returns whether the char is in the expected range or not.
    https://www.andiamo.co.uk/resources/iso-language-codes/
    >>> print(char_in_language('1', 'alphabet'))
    False
    >>> print(char_in_language('a', 'ascii'))
    True
    >>> print(char_in_language('a'))
    alphabet
    >>> print(char_in_language('时', 'num'))
    False
    """
    assert (len(in_char) == 1)
    in_range = ''
    in_char = ord(in_char)
    if in_char >= ord('\u0000') and in_char <= ord('\u007f'):
        in_range = 'ascii'
        if expected == in_range:
            return True
    if in_char >= ord('\u0030') and in_char <= ord('\u0039'):
        in_range = 'num'
        if expected == in_range:
            return True
        if expected is not None and expected != in_range:
            return False
    if (in_char >= ord('\u0041') and in_char <= ord('\u005a')) or (
            in_char >= ord('\u0061') and in_char <= ord('\u007a')):
        in_range = 'alphabet'
        if expected == in_range:
            return True
        if expected is not None and expected != in_range:
            return False
    if expected in ['ascii', 'num', 'alphabet']:
        return False
    if in_range != '':
        return in_range
    # above check for ascii ranges.

    if in_char >= ord('\u4e00') and in_char <= ord('\u9fa5'):
        in_range = 'zh'
        if expected is not None:
            if expected == in_range:
                return True
            else:
                return False
        else:
            return in_range
    if in_char >= ord('\u4e00') and in_char <= ord('\u9fa5'):
        in_range = 'zh'
        if expected is not None:
            if expected == in_range:
                return True
            else:
                return False
        else:
            return in_range
    if in_char >= ord('\u0800') and in_char <= ord('\u4dff'):
        in_range = 'ja'
        if expected is not None:
            if expected == in_range:
                return True
            else:
                return False
        else:
            return in_range
    else:
        if expected is not None:
            return False
        else:
            return ''


def string_in_language(in_string, expected=None):
    """
    :param in_char: utf-8 input string
    :param expected: language
    :return: true or false (full ascii string is going to be false when expected is not ascii)
    """

    ascii_matches = re.findall(r'[\u0000-\u007f]+', in_string)
    if len(ascii_matches) > 0 and in_string == ascii_matches[0]:
        if expected == 'ascii':
            return True
        else:
            return False

    if expected == 'zh':
        matches = re.findall(r'[\u0000-\u007f\u4e00-\u9fff]+', in_string)
        if len(matches) > 0 and in_string == matches[0]:
            return True
        else:
            return False
    if expected == 'ja':
        matches = re.findall(r'[\u0000-\u007f\u0800-\u4e00]+', in_string)
        if len(matches) > 0 and in_string == matches[0]:
            return True
        else:
            return False
    else:
        return False


def nltk_english_pos_tags():
    """
    return a dictionary of general classes of pos tags used by nltk.pos_tag.
    https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
    """
    digits = ('CD',)
    foreign = ('FW',)
    adjective = ('JJ', 'JJR', 'JJS')
    noun = ('NN', 'NNS', 'NNP', 'NNPS')
    adverb = ('RB', 'RBR', 'RBS', 'RP')
    verb = ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    entities = digits + foreign + adjective + noun + adverb + verb

    pronoun = ('PDT', 'POS', 'PRP', 'PRP$')
    wh = ('WDT', 'WP', 'WP$', 'WRB')
    meaningless_along = ('CC', 'DT', 'EX', 'IN', 'LS', 'MD', 'TO', 'UH') + pronoun + wh

    return {'entities': entities, 'meaningless_along': meaningless_along, 'digits': digits, 'foreign': foreign,
            'adjective': adjective, 'noun': noun, 'adverb': adverb, 'verb': verb, 'pronoun': pronoun, 'wh': wh}
