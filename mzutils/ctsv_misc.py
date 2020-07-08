import codecs
import csv
import os
import sys
import mzutils.list_misc


def write_tsv(file_path, rows):
    """
    :param file_path:
    :param rows: a list of rows to be written in the tsv file. The rows are lists of items.
    :return:
    """
    csv.field_size_limit(sys.maxsize)
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
    csv.field_size_limit(sys.maxsize)
    cached_list = []
    with codecs.open(file_path, "r", encoding="utf-8") as fp:
        tsv_reader = csv.reader(fp, delimiter='\t')
        for row in tsv_reader:
            cached_list.append(row)
    return cached_list


def append_tsv(file_path, rows):
    """
    :param file_path:
    :param rows: a list of rows to be written in the tsv file. The rows are lists of items.
    :return:
    """
    csv.field_size_limit(sys.maxsize)
    with codecs.open(file_path, "a+", encoding="utf-8") as fp:
        tsv_writer = csv.writer(fp, delimiter='\t')
        for row in rows:
            tsv_writer.writerow(row)


def segment_large_csv(file_path, destination_path, segmentation_length, duplicate_header=False):
    """
    segment a large file to several smaller files to a destination.
    If duplicate_header is True, the first line of  the original large file will be duplicated to every segmented files,
    results in the length of segmented file = segmentation_length + 1. which also means that
    :param file_path:
    :param destination_path:
    :param segmentation_length:
    :param duplicate_header:
    :return: how many files are segmented.
    """
    csv.field_size_limit(sys.maxsize)
    filename, file_extension = os.path.splitext(os.path.basename(file_path))
    header = None
    with codecs.open(file_path, "r", encoding="utf-8") as fp:
        csv_reader = csv.reader(fp)
        if duplicate_header:
            header = csv_reader.__next__()
            segmentation_length += 1
        j = 0
        while True:
            i = 0
            j += 1
            current_filepath = os.path.join(destination_path, filename + str(j) + file_extension)
            with codecs.open(current_filepath, "w+", encoding="utf-8") as fp:
                csv_writer = csv.writer(fp)
                if duplicate_header:
                    csv_writer.writerow(header)
                while i < segmentation_length:
                    try:
                        row = next(csv_reader)
                        csv_writer.writerow(row)
                        i += 1
                    except StopIteration:
                        return j


def segment_large_tsv(file_path, destination_path, segmentation_length, duplicate_header=False):
    """
    segment a large file to several smaller files to a destination.
    If duplicate_header is True, the first line of  the original large file will be duplicated to every segmented files,
    results in the length of segmented file = segmentation_length + 1. which also means that
    :param file_path:
    :param destination_path:
    :param segmentation_length:
    :param duplicate_header:
    :return: how many files are segmented.
    """
    csv.field_size_limit(sys.maxsize)
    filename, file_extension = os.path.splitext(os.path.basename(file_path))
    header = None
    with codecs.open(file_path, "r", encoding="utf-8") as fp:
        tsv_reader = csv.reader(fp, delimiter='\t')
        if duplicate_header:
            header = tsv_reader.__next__()
            segmentation_length += 1
        j = 0
        while True:
            i = 0
            j += 1
            current_filepath = os.path.join(destination_path, filename + str(j) + file_extension)
            with codecs.open(current_filepath, "w+", encoding="utf-8") as fp:
                tsv_writer = csv.writer(fp, delimiter='\t')
                if duplicate_header:
                    tsv_writer.writerow(header)
                while i < segmentation_length:
                    try:
                        row = next(tsv_reader)
                        tsv_writer.writerow(row)
                        i += 1
                    except StopIteration:
                        return j


def save_tsv_as_csv(tsv_file, csv_file=None):
    csv.field_size_limit(sys.maxsize)
    from mzutils.os_misc import parent_dir_and_name, basename_and_extension
    with codecs.open(tsv_file, "r", encoding="utf-8") as tfp:
        if csv_file is None:
            csv_file = os.path.join(parent_dir_and_name(tsv_file)[0], basename_and_extension(tsv_file)[0]) + '.csv'
        with codecs.open(csv_file, "w+", encoding="utf-8") as cfp:
            tsv_reader = csv.reader(tfp, delimiter='\t')
            csv_writer = csv.writer(cfp, delimiter=',')
            for row in tsv_reader:
                csv_writer.writerow(row)


def find_max_sub_list_length(lst: list):
    """
    pick the left longest sub_list
    """
    if len(lst) == 0:
        return 0, None
    max_len = 0
    max_sub_list = lst[0]
    for sub_list in lst:
        if len(sub_list) > max_len:
            max_len = len(sub_list)
            max_sub_list = sub_list
    return max_len, max_sub_list



def beautify_csv_lines_horizontal(lst: list):
    """
    the list contain sub_lists with different lengths. This function helps to write them with paddings.
    return back list of sub_lists with the same length.
    """
    lst = lst.copy()
    max_len, _ = find_max_sub_list_length(lst)
    for i in range(len(lst)):
        lst[i] = mzutils.list_misc.pad_list(lst[i], max_len)
    return lst


def beautify_csv_lines(lst: list):
    """
    the list contain sub_lists with different lengths. This function helps to write them with paddings.
    return back list of sub_lists of rows to write to csv.
    """
    curr_lst = []
    lst = beautify_csv_lines_horizontal(lst)
    for i in range(len(lst[0])):
        curr_lst.append([lst[j][i] for j in range(len(lst))])
    return curr_lst
