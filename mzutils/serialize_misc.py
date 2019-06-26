import pickle


def serialize(obj, location):
    with open(location, 'wb') as fp:
        pickle.dump(obj, fp)


def deserialize(location):
    with open(location, 'rb') as fp:
        return pickle.load(fp)
