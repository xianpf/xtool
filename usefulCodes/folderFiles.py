import os, shutil


def mkdirs(path, rm_old=True):
    if os.path.exists(path):
        if rm_old:
            shutil.rmtree(path)
        else:
            return path
    os.makedirs(path)
    return path
   
