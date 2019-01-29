import os
import shutil
import time

def hw():
    return "hello world!"


# remove all files in a directory. If just_files=False, also remove all directory trees in that directory.
def clean_dir(dir_path, just_files=True):
    for name in os.listdir(dir_path):
        if os.path.isfile(dir_path + name):
            os.remove(dir_path + name)
        elif not just_files:
            if os.path.isdir(dir_path + name):
                shutil.rmtree(dir_path + name)