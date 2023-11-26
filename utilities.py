import os
import shutil

def create_directory(directory, delete_if_exists=False):
    if delete_if_exists and os.path.exists(directory):
        shutil.rmtree(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)
