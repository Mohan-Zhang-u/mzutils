import codecs
import errno
import os
import shutil
import tarfile
import time
import zipfile
from inspect import getfullargspec

import nltk


def parent_dir_and_name(file_path):
    """
    >>> file_path="a/b.c"
    >>> parent_dir_and_name(file_path)
    ('/root/.../a', 'b.c')
    :param file_path:
    :return:
    """
    return os.path.split(os.path.abspath(file_path))


def basename_and_extension(file_path):
    """
    >>> file_path="a/b.c"
    >>> basename_and_extension(file_path)
    ('b', '.c')
    :param file_path:
    :return:
    """
    return os.path.splitext(os.path.basename(file_path))


def get_things_in_loc(in_path, just_files=True, endswith=None):
    """
    in_path can be file path or dir path.
    This function return a list of file paths
    in in_path if in_path is a dir, or within the 
    parent path of in_path if it is not a dir.
    just_files=False will let the function go recursively
    into the subdirs.
    :endswith: None or a list of file extensions (to end with).
    """
    # TODO: check for file
    if not os.path.exists(in_path):
        print(str(in_path) + " does not exists!")
        return
    re_list = []
    if not os.path.isdir(in_path):
        in_path = parent_dir_and_name(in_path)[0]

    for name in os.listdir(in_path):
        if endswith is None or (True in [name.endswith(ext) for ext in endswith]):
            name_path = os.path.abspath(os.path.join(in_path, name))
            if os.path.isfile(name_path):
                re_list.append(name_path)
            elif not just_files:
                if os.path.isdir(name_path):
                    re_list += get_things_in_loc(name_path, just_files)
    return re_list


def get_checkpoints_in_loc(in_path, keywords=['checkpoint-'], files_or_folders='folders'):
    """
    This function will loop through in_path to find all files/folders that includes all keywords 
    if files_or_folders='files'/'folders'. 
    again, in_path can be file path or dir path.
    The function is meant to grab all checkpoint-XXXX in a folder.
    """
    if not os.path.exists(in_path):
        print(str(in_path) + " does not exists!")
        return
    re_list = []
    if not os.path.isdir(in_path):
        in_path = parent_dir_and_name(in_path)[0]

    for name in os.listdir(in_path):
        name_path = os.path.abspath(os.path.join(in_path, name))
        pattern_truth = all([keyword in name_path for keyword in keywords])
        if pattern_truth:
            if os.path.isfile(name_path) and files_or_folders == 'files':
                re_list.append(name_path)
            elif os.path.isdir(name_path) and files_or_folders == 'folders':
                re_list.append(name_path)
    return re_list


def clean_dir(dir_path, just_files=True):
    """
    Clean up a directory.
    :param dir_path:
    :param just_files: If just_files=False, also remove all directory trees in that directory.
    :return:
    """
    if not os.path.isdir(dir_path):
        if not os.path.exists(dir_path):
            print(str(dir_path) + " does not exists!")
            return
        if not os.path.isdir(dir_path):
            print(str(dir_path) + " has to be a directory!")
            return
    for name in os.listdir(dir_path):
        name_path = os.path.join(dir_path, name)
        if os.path.isfile(name_path):
            os.remove(name_path)
        elif not just_files:
            if os.path.isdir(name_path):
                shutil.rmtree(name_path)


def mkdir_p(dir_path):
    """
    mkdir -p functionality in python
    :param dir_path:
    :return:
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def unzip_all(dir_path, target_path, endswith=".zip"):
    """

    :param dir_path:
    :param target_path:
    :param endswith: ".zip", ".tar.gz" or ".tar"
    :return:
    """
    for item in os.listdir(dir_path):
        if item.endswith(endswith):
            if endswith == ".zip":
                zip_ref = zipfile.ZipFile(os.path.join(dir_path, item), 'r')
            elif endswith == ".tar.gz":
                zip_ref = tarfile.open(os.path.join(dir_path, item), 'r:gz')
            elif endswith == ".tar":
                zip_ref = tarfile.open(os.path.join(dir_path, item), 'r:')
            else:
                continue
            zip_ref.extractall(target_path)
            zip_ref.close()


def documents_segementor_on_word_length(documents_dir, store_dir, max_length, language='english',
                                        clean_store_dir=False):
    """
    segment a long document to several small documents based on the nltk tokenized word length.
    sentence structure will be kept.
    :param documents_dir: where all documents located.
    :param store_dir: where to store segmented documents.
    :param max_length: document segments' max length.
    :param language: for the use of nltk, default english.
    :param clean_store_dir:
    :return: number of documents after segmented.
    """
    final_num_of_docs = 0
    if not os.path.isdir(documents_dir):
        raise Exception("documents_dir: where all documents located.")
    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)
    if clean_store_dir:
        clean_dir(store_dir, just_files=False)
    names = [name for name in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, name))]
    for name in names:
        final_num_of_docs += helper_document_segmentor(documents_dir, store_dir, name, max_length, language)
    return final_num_of_docs


# ------------------helper funcs-----------------------------


def helper_document_segmentor(documents_dir, store_dir, name, max_length, language):
    documents = []
    with codecs.open(os.path.join(documents_dir, name), "r", "utf-8") as fp:
        filecontent = fp.read()
        sentences = nltk.sent_tokenize(filecontent, language)
        i = 0
        word_count = 0
        document = ""
        while i < len(sentences):
            sentence = sentences[i]
            current_count = len(nltk.word_tokenize(sentence, language))
            if current_count >= max_length:
                document = document + sentence + " "
                documents.append(document)
                word_count = 0
                document = ""
                i = i + 1
                print("Warning: " + "there is a sentence with word length " + str(
                    current_count) + " , but the maximum document length is " + str(max_length))
                continue
                # raise Exception("there is a sentence with word length " + str(current_count) + " , but the maximum document length is " + str(max_length))
            if word_count + current_count >= max_length:
                documents.append(document)
                word_count = 0
                document = ""
            else:
                document = document + sentence + " "
                word_count = word_count + current_count
                i = i + 1
        documents.append(document)
    helper_save_documents(store_dir, name, documents)
    return len(documents)


def helper_save_documents(store_dir, name, documents):
    with codecs.open(os.path.join(store_dir, name), "w+", "utf-8") as fp:
        fp.write(documents[0])
    for document in documents[1:]:
        filepath = helper_check_existance_and_add_timestamp(store_dir, name)
        with codecs.open(filepath, "w+", "utf-8") as fp:
            fp.write(document)


def helper_check_existance_and_add_timestamp(store_dir, name):
    timestamp = str(int(round(time.time() * 1000)))
    filepath = os.path.join(store_dir, name)
    filename, extension = os.path.splitext(filepath)
    while os.path.exists(filename + extension):
        filename += timestamp
    return filename + extension


def loop_through_copy_files_to_one_dir(looped_dir, target_dir, include_link=False):
    """
    function to loop through nested directories and copy all the files to a target directory.
    :param looped_dir:
    :param target_dir: a directory string.
    :return:
    """
    if not os.path.isdir(looped_dir):
        raise Exception("looped_dir: a directory.")
    if not os.path.isdir(target_dir):
        raise Exception("target_dir: a directory.")
    for thing in os.listdir(looped_dir):
        thing = os.path.join(looped_dir, thing)
        if os.path.isdir(thing):
            loop_through_copy_files_to_one_dir(thing, target_dir)
        elif os.path.isfile(thing):
            shutil.move(thing, os.path.join(target_dir, parent_dir_and_name(thing)[1]))
        elif include_link:
            shutil.move(thing, os.path.join(target_dir, parent_dir_and_name(thing)[1]))
    return


def loop_through_return_abs_file_path(looped_dir):
    """
    function to loop through nested directories and return file absolute path in a list.
    :param looped_dir:
    :return: list
    """
    re_list = []
    if not os.path.isdir(looped_dir):
        raise Exception("looped_dir: a directory.")
    for thing in os.listdir(looped_dir):
        thing = os.path.join(looped_dir, thing)
        thing = os.path.abspath(thing)
        if os.path.isdir(thing):
            re_list = re_list + loop_through_return_abs_file_path(thing)
        elif os.path.isfile(thing):
            re_list.append(thing)
    return re_list


def loop_through_store_files_to_list(looped_dir, encoding="utf-8"):
    """
    function to loop through nested directories and store the content of all files into a list separately.
    This function does not care about symbolic link inside the nested directories.
    :param looped_dir:
    :param encoding:
    :return: list
    """
    re_list = []
    if not os.path.isdir(looped_dir):
        raise Exception("looped_dir: a directory.")
    for thing in os.listdir(looped_dir):
        thing = os.path.join(looped_dir, thing)
        if os.path.isdir(thing):
            re_list = re_list + loop_through_store_files_to_list(thing, encoding)
        elif os.path.isfile(thing):
            with codecs.open(thing, 'r', encoding) as fp:
                filecontent = fp.read()
                re_list.append(filecontent)
    return re_list


def loop_through_store_lines_to_list(looped_dir, encoding="utf-8"):
    """
    function to loop through nested directories and store the lines of all files into a list.
    This function does not care about symbolic link inside the nested directories.
    :param looped_dir:
    :param encoding:
    :return: list
    """
    re_list = []
    if not os.path.isdir(looped_dir):
        raise Exception("looped_dir: a directory.")
    for thing in os.listdir(looped_dir):
        thing = os.path.join(looped_dir, thing)
        if os.path.isdir(thing):
            re_list = re_list + loop_through_store_files_to_list(thing, encoding)
        elif os.path.isfile(thing):
            with codecs.open(thing, 'r', encoding) as fp:
                filecontent = fp.readlines()
                re_list += filecontent
    return re_list


def save__init__args(values, underscore=False, overwrite=False, subclass_only=False):
    """
    Use in `__init__()` only; assign all args/kwargs to instance attributes.
    To maintain precedence of args provided to subclasses, call this in the
    subclass before `super().__init__()` if `save__init__args()` also appears
    in base class, or use `overwrite=True`.  With `subclass_only==True`, only
    args/kwargs listed in current subclass apply.
    usage:
    >>> class AgentModel:
    ...     def __init__(
    ...             self,
    ...             meta_info_attr_size=7,
    ...             obs_shape=(3, 64, 64),
    ...             reward_shape=(1,),
    ...             n_agents=1,
    ...             obs_last_action=False,
    ...             obs_agent_id=True,
    ...             rnn_hidden_dim=64,
    ...             based_on='observation',
    ...             n_actions=11,
    ...             use_cuda=True,):
    ...         save__init__args(locals())
    >>> a=AgentModel()
    >>> a.rnn_hidden_dim
    >>> 64
    """
    prefix = "_" if underscore else ""
    self = values['self']
    args = list()
    Classes = type(self).mro()
    if subclass_only:
        Classes = Classes[:1]
    for Cls in Classes:  # class inheritances
        if '__init__' in vars(Cls):
            args += getfullargspec(Cls.__init__).args[1:]
    for arg in args:
        attr = prefix + arg
        if arg in values and (not hasattr(self, attr) or overwrite):
            setattr(self, attr, values[arg])


def set_local_vars_from_yaml(yaml_loc, name_space_dict):
    """
    set local variables from yaml file.
    :param yaml_loc: your yaml file location.
    :param name_space_dict: a dictionary that contains the local variables. e.g. locals()
    :return: None
    for example, if your yaml file contains a variable called num_workers and the value of 
    which is an integer 4, then
    >>> set_local_vars_from_yaml('path_to_file.yaml', locals())
    >>> num_workers
    4
    """
    import yaml
    with open(yaml_loc, 'r') as fp:
        config_dict = yaml.safe_load(fp)
    name_space_dict.update(config_dict)


class TimeRecorder:
    def __init__(self):
        """
        need to import time.
        """
        self.init_time = time.time()
        self.base_time = self.init_time
        self.times = []

    def record(self, name=""):
        current = time.time()
        self.times.append((name, current - self.base_time))
        self.base_time = current

    def get_times(self):
        return self.times

    def get_times_str(self):
        return str(self.get_times())
