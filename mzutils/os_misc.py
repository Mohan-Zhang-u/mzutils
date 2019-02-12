import os
import shutil
import codecs
import time
import errno
import zipfile
import tarfile

# dependencies
import nltk


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
    :param language: for the use of nltk, default english
    :param clean_store_dir:
    :return:
    """
    if not os.path.isdir(documents_dir):
        raise Exception("documents_dir: where all documents located.")
    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)
    if clean_store_dir:
        clean_dir(store_dir, just_files=False)
    names = [name for name in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, name))]
    for name in names:
        helper_document_segmentor(documents_dir, store_dir, name, max_length, language)


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
            if word_count + current_count < max_length:
                document = document + sentence + " "
                word_count = word_count + current_count
                i = i + 1
            else:
                documents.append(document)
                word_count = 0
                document = ""
        documents.append(document)
    helper_save_documents(store_dir, name, documents)


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
