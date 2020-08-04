import codecs
import json
import os
import mzutils.json_misc
import mzutils.os_misc


# ---------------------------------SQuAD 1.1 Functionss---------------------------------

# The structure looks like this:
# SQuAD:https://rajpurkar.github.io/SQuAD-explorer/
#
# file.json
# ├── "data"
# │   └── [i]
# │       ├── "paragraphs"
# │       │   └── [j]
# │       │       ├── "context": "paragraph text"
# │       │       └── "qas"
# │       │           └── [k]
# │       │               ├── "answers"
# │       │               │   └── [l]
# │       │               │       ├── "answer_start": N
# │       │               │       └── "text": "answer"
# │       │               ├── "id": "<uuid>"
# │       │               └── "question": "paragraph question?"
# │       └── "title": "document id"
# └── "version": 1.1


def generate_multi_test_cases(list_of_paragraphs, list_of_questions, json_store_path):
    """
    given pairs of paragraphs and questions, it creates a json file just like how training/dev/test data stored in
    SQuAD 1.1
    :param list_of_paragraphs:
    :param list_of_questions:
    :param json_store_path:
    :return:
    """
    assert len(list_of_paragraphs) == len(list_of_questions)
    length_of_them = len(list_of_paragraphs)
    data = []
    version = "1.1"
    jsondict = {}
    jsondict["data"] = data
    jsondict["version"] = version

    for j in range(length_of_them):
        new_paragraph = {}
        new_paragraph["context"] = list_of_paragraphs[j]
        new_paragraph["qas"] = [{"answers": [{"answer_start": -1, "text": ""}], "question": list_of_questions[j],
                                 "id": j}]
        data.append({"title": "", "paragraphs": [new_paragraph]})  # here we can have multiple paragraph in paragraphs

    with codecs.open(json_store_path, 'w+', encoding='utf-8') as fp:
        json.dump(jsondict, fp)


def simple_squad_segmentor(squad_file_path, store_location, num_of_paragraphs=500):
    mzutils.os_misc.mkdir_p(store_location)
    squad_file_name = mzutils.os_misc.basename_and_extension(squad_file_path)[0]
    squad_file_data = mzutils.json_misc.load_config(squad_file_path)["data"]
    epoch = len(squad_file_data) // num_of_paragraphs + 1
    for i in range(epoch):
        store_dict = {"data": squad_file_data[(i * num_of_paragraphs):((i + 1) * num_of_paragraphs)], "version": "1.1"}
        with codecs.open(os.path.join(store_location, squad_file_name) + str(i) + ".json", 'w+',
                         encoding='utf-8') as fp:
            json.dump(store_dict, fp)


# ---------------------------------TriviaQA Functions---------------------------------

# file.json
# ├── [{}] "Data"
# │       ├── {} "Answer"
# │       │   └── [] "Aliases"
# │       │   └── [] "NormalizedAliases"
# │       │   └── "NormalizedValue"
# │       ├── "Question"
# │       └── "QuestionId"
# other useless rows omitted.


def retrieve_questions_from_triviaQA(file_path, destination_path=None):
    """
    :param file_path:
    :return:[{"Question" : "", "QuestionId" : "", "AcceptableAnswers" : ""}]
    or
    None and write {"data": [{"Question" : "", "QuestionId" : "", "AcceptableAnswers" : ""}]}
    """
    return_list = []
    data_list = mzutils.json_misc.load_config(file_path)["Data"]
    for data in data_list:
        AcceptableAnswers = data["Answer"]["Aliases"] + data["Answer"]["NormalizedAliases"] + [
            data["Answer"]["NormalizedValue"]]
        return_list.append(
            {"question": data["Question"], "questionid": data["QuestionId"], "acceptableanswers": AcceptableAnswers})
    if not destination_path:
        return return_list
    else:
        mzutils.json_misc.dump_config(destination_path, {"data": return_list})


def generate_multi_test_cases_triviaQA(retrieved_json_path, json_store_path, documents_path, missing_file_path=None):
    """
    given pairs of paragraphs and questions, it creates a json file just like how training/dev/test data stored in
    SQuAD 1.1
    the format is :
    {
        "[question_order, answer_order, "qid", ["ground_truths"]] : "ans",
        "[0, 0, 'tc_1250', ['The Swiss Miss', 'Martina hingis', 'Martina Hingisov\u00e1', 'Martina Hingis', 'MartinaHingis', 'Martina Hingisova', 'Hingis', 'hingis', 'swiss miss', 'martina hingis', 'martina hingisova', 'martinahingis', 'martina hingisov\u00e1', 'martina hingis']]": "Li Na",
    }
    """
    retrieved_list = mzutils.json_misc.load_config(retrieved_json_path)['data']
    missing_files = []

    data = []
    version = "1.1"
    jsondict = {}
    jsondict["data"] = data
    jsondict["version"] = version
    j = -1

    for i, retrieved_data in enumerate(
            retrieved_list):  # i:question number from 0; j: number of question|answer pairs from 0

        if i % 500 == 0:
            print(str(i) + " questions formatted ... ")

        question = retrieved_data["question"]
        questionid = retrieved_data["questionid"]
        acceptableanswers = retrieved_data["acceptableanswers"]
        documents = retrieved_data["documents"]

        for document_name in documents:
            doc_path = os.path.join(documents_path, document_name)
            if not os.path.exists(doc_path):
                missing_files.append(document_name)
            else:
                j += 1
                with codecs.open(doc_path, 'r', encoding='utf8') as fp:
                    document_content = fp.read()
                new_paragraph = {}
                new_paragraph["context"] = document_content
                new_paragraph["qas"] = [{"answers": [{"answer_start": -1, "text": ""}], "question": question,
                                         "id": str([i, j, questionid, acceptableanswers])}]
                data.append(
                    {"title": "", "paragraphs": [new_paragraph]})  # here we can have multiple paragraph in paragraphs

    with codecs.open(json_store_path, 'w+', encoding='utf-8') as fp:
        json.dump(jsondict, fp)
    if missing_file_path:
        with codecs.open(missing_file_path, 'w+', encoding='utf-8') as fp:
            json.dump(missing_files, fp)


# ---------------------------------TriviaQA Evaluation Functions---------------------------------


def concatenate_predictions_dicts(squadjsons_files_dir, output_file=None):
    """

    :param squadjsons_files_dir: should be one of "wikipedia-train" "wikipedia-dev" "web-train" "web-dev" "verified-web-dev" "verified-wikipedia-dev"
    it is actually a directory with format:
    squadjsons_files_dir
    ├── squadjsons%d
    │   └── predictions.json
    :param outout_dir: directory to store concatenated predictions.json
    :return: None
    """
    if not output_file:
        output_file = os.path.join(squadjsons_files_dir, "predictions.json")

    output_dict = {}
    dir_num = 0
    while True:
        squadjsonsnum_dir = os.path.join(squadjsons_files_dir,
                                         "squadjsons" + str(dir_num))  # this is the path of directory squadjsons%d
        if not os.path.isdir(squadjsonsnum_dir):
            break
        output_dict.update(mzutils.json_misc.load_config(os.path.join(squadjsonsnum_dir, "predictions.json")))
        dir_num += 1
    mzutils.json_misc.dump_config(output_file, output_dict)
