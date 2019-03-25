import codecs
import json
import mzutils.json_misc


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


# ---------------------------------TriviaQA Functionss---------------------------------
# file.json
# ├── [{}] "data"
# │       ├── {} "Answer"
# │       │   └── [] "Aliases"
# │       │   └── [] "NormalizedAliases"
# │       │   └── "NormalizedValue"
# │       ├── "Question"
# │       └── "QuestionId"
# other useless rows omitted.
def retrive_questions_from_triviaQA(file_path, destination_path = None):
    """
    :param file_path:
    :return:[{"Question" : "", "QuestionId" : "", "AcceptableAnswers" : ""}]
    or
    None and write {"data": [{"Question" : "", "QuestionId" : "", "AcceptableAnswers" : ""}]}
    """
    return_list = []
    data_list = mzutils.json_misc.load_config(file_path)["data"]
    for data in data_list:
        AcceptableAnswers = data["answer"]["Aliases"] + data["answer"]["NormalizedAliases"] + [
            data["answer"]["NormalizedValue"]]
        return_list.append(
            {"Question": data["Question"], "QuestionId": data["QuestionId"], "AcceptableAnswers": AcceptableAnswers})
    if not destination_path:
        return return_list
    else:
        mzutils.json_misc.dump_config(destination_path, {"data": return_list})
