import pickle
import codecs

def serialize(obj, location):
    with open(location, 'wb') as fp:
        pickle.dump(obj, fp)


def deserialize(location):
    with open(location, 'rb') as fp:
        return pickle.load(fp)

def serialize_file_to_list(file_location, location, strip=False):
    """
    serialize a file from file_location to a list of sentences in location as a python pickle
    file_location: location of file to load
    location: location of pickle
    strip: whether rstrip the sentence to get rid of \n
    """
    with codecs.open(file_location, 'r', 'utf-8') as fpr:
        if strip:
            sentences = fpr.readlines()
        else:
            sentences = [line.rstrip() for line in fpr]
    with open(location, 'wb') as fp:
        pickle.dump(sentences, fp)
