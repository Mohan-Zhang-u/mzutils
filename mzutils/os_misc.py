import os
import shutil
import time


def hw():
    return "hello world!"


# remove all files in a directory. If just_files=False, also remove all directory trees in that directory.
def clean_dir(dir_path, just_files=True):
    for name in os.listdir(dir_path):
        name_path = os.path.join(dir_path, name)
        if os.path.isfile(name_path):
            os.remove(name_path)
        elif not just_files:
            if os.path.isdir(name_path):
                shutil.rmtree(name_path)
