import codecs
import csv


def write_tsv(file_path, rows):
    """
    :param file_path:
    :param rows: a list of rows to be written in the tsv file. The rows are lists of items.
    :return:
    """
    with codecs.open(file_path, "w+", encoding="utf-8") as fp:
        tsv_writer = csv.writer(fp, delimiter='\t')
        for row in rows:
            tsv_writer.writerow(row)
            # tsv_writer.writerow(["index", "question11", "question2"])
            # tsv_writer.writerow(["0", sentence1, sentence2])


def read_tsv(file_path):
    """
    read a tsv into a nested python list.
    :param file_path:
    :return:
    """
    cached_list = []
    with codecs.open(file_path, "r", encoding="utf-8") as fp:
        tsv_reader = csv.reader(fp, delimiter='\t')
        for row in tsv_reader:
            cached_list.append(row)
    return cached_list
