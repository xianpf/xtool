from os.path import join, dirname, realpath
from setuptools import setup
import sys

# assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
#     "The Spinning Up repo is designed to work with Python 3.6 and greater." \
#     + "Please install it before proceeding."

with open(join("xtool", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='xtool',
    py_modules=['xtool'],
    version=__version__,#'0.1',
    install_requires=[
        'ipython',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy',
    ],
    description="Teaching tools for using python and machine learning tasks.",
    author="XIAN Pengfei",
)
