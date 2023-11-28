import os
import shutil
import glob

def create_directory(directory, delete_if_exists=False):
    if delete_if_exists and os.path.exists(directory):
        shutil.rmtree(directory)

    if not os.path.exists(directory):
        os.makedirs(directory)

def get_files_from_dir(dir_pattern):
    files = glob.glob(dir_pattern)
    return files

def copy_file_to_dir(src, dst_dir):
    _, tail = os.path.split(src)
    shutil.copyfile(src, os.path.join(dst_dir, tail))