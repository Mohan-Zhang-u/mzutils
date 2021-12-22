import numpy as np


def helper_flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item


def subword_tokenize_labels(tokens, labels, tokenizer, bert_special_tokens=True):
    """
    :param tokens: something like ['John', 'Johanson', 'lives', 'in', 'Ramat', 'Gan', 'Gang', '.']. can get from 
    tokens = tokenizer.basic_tokenizer.tokenize("John Johanson lives in Ramat Gan Gang.")
    tokens = nltk.word_tokenize("John Johanson lives in Ramat Gan Gang.")

    :param labels: [1, 2, 0, 0, 1, 2, 2, 0]. NER tokens. 
    labels:
    0 -> O out
    1 -> B beginning
    2 -> I continued
    3 -> X sub-words that are not tagged.
    
    :param tokenizer: e.g. transformers.BertTokenizer.from_pretrained('bert-base-cased')

    :param bert_special_tokens: add '[CLS]' and '[SEP]' or not.
    
    :return: (['[CLS]', 'john', 'johan', '##son', 'lives', 'in', 'rama', '##t', 'gan', 'gang', '.', '[SEP]'], [101, 2198, 13093, 3385, 3268, 1999, 14115, 2102, 25957, 6080, 1012, 102], array([ 1,  2,  4,  5,  6,  8,  9, 10]), [0, 1, 2, 3, 0, 0, 1, 3, 2, 2, 0, 0])
    """
    assert len(tokens) == len(labels)

    subwords = list(map(tokenizer.tokenize, tokens))  # subwords here is flattened.
    subword_lengths = list(map(len, subwords))
    subwords = list(helper_flatten(subwords))
    if bert_special_tokens:
        subwords = ['[CLS]'] + subwords + ['[SEP]']  # 
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
    bert_labels = [[label] + (sublen - 1) * [3] for sublen, label in zip(subword_lengths, labels)]
    bert_labels = [0] + list(helper_flatten(bert_labels)) + [0]
    encoded_subwords = tokenizer.encode(subwords, add_special_tokens=False)
    assert len(subwords) == len(bert_labels)

    return subwords, encoded_subwords, token_start_idxs, bert_labels


def labels_from_subword_labels(tokens, bert_labels, tokenizer, bert_special_tokens=True):
    """
    :param tokens: something like ['John', 'Johanson', 'lives', 'in', 'Ramat', 'Gan', '.']. can get from 
    tokens = tokenizer.basic_tokenizer.tokenize("John Johanson lives in Ramat Gan.")
    tokens = nltk.word_tokenize("John Johanson lives in Ramat Gan.")

    :param bert_labels: [0, 1, 2, 3, 0, 0, 1, 3, 2, 2, 0, 0]. NER tokens from subword_tokenize_labels. 
    labels:
    0 -> O out
    1 -> B beginning
    2 -> I continued
    3 -> X sub-words that are not tagged.
    
    :param tokenizer: e.g. transformers.BertTokenizer.from_pretrained('bert-base-cased')

    :param bert_special_tokens: add '[CLS]' and '[SEP]' or not.
    
    :return: (['John', 'Johanson', 'lives', 'in', 'Ramat', 'Gan', 'Gang', '.'], [1, 2, 0, 0, 1, 2, 2, 0])
    """
    if bert_special_tokens:
        bert_labels = bert_labels[1:-1]  # remove '[CLS]' and '[SEP]'
    subwords = list(map(tokenizer.tokenize, tokens))
    # now, select correct labels according to subword length. Always select the first given label for each word.
    curr_working_idx = 0
    labels = []
    for subword_list in subwords:
        labels.append(bert_labels[curr_working_idx])
        curr_working_idx += len(subword_list)
    assert len(tokens) == len(labels)
    return tokens, labels


def rejoin_bert_tokenized_sentence(sentence):
    """
    original sentence is "The Smiths' used their son's car."
    tokenizer.basic_tokenizer.tokenize("The Smiths' used their son's car.") gives ['the', 'smiths', "'", 'used', 'their', 'son', "'", 's', 'car', '.']
    fine_text returns "the smiths ' used their son ' s car ."
    tokenizer.basic_tokenizer.tokenize(fine_text) gives ['the', 'smiths', "'", 'used', 'their', 'son', "'", 's', 'car', '.'] again.
    """
    text = ' '.join([x for x in sentence])
    fine_text = text.replace(' ##', '')
    return fine_text
