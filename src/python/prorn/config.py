import os
from ConfigParser import ConfigParser


def get_root_dir():
    fpath = os.path.abspath(__file__)
    (cwdir, fname) = os.path.split(fpath)
    rdir = os.path.abspath(os.path.join(cwdir, '..', '..', '..'))
    return rdir


ROOT_DIR = get_root_dir()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TEMP_DIR = '/tmp'
