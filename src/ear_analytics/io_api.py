""" IO support for the tool set. """
import os
import configparser

import pandas as pd

from itertools import dropwhile
from json import load, dumps


def read_data(file_path, **kwargs):
    """
    This function reads data properly whether the input `file_path` is a list
    of concret files, is a directory and returns a pandas.DataFrame object.

    kwargs are passed to pandas.read_csv function.
    """

    def load_files(filenames, base_path=None):
        for filename in filenames:
            if base_path:
                path_file = os.path.join(base_path, filename)
            else:
                path_file = filename
            yield pd.read_csv(path_file, **kwargs)

    if isinstance(file_path, str):
        # `file_path` is only a string containing some file or a directory
        if os.path.isfile(file_path):
            print(f'reading file {file_path}')
            data_f = pd.concat(load_files([file_path]), ignore_index=True)
        elif os.path.isdir(file_path):
            print(f'reading files contained in directory {file_path}')
            print(f'{os.listdir(file_path)}')
            data_f = pd.concat(load_files(os.listdir(file_path),
                               base_path=file_path),
                               ignore_index=True)
        else:
            print(f'{file_path} does not exist!')
            raise FileNotFoundError
    elif isinstance(file_path, list):
        # `file_path` is a list containing files
        try:
            no_exists_file = next(dropwhile(os.path.exists, file_path))
            print(f'file {no_exists_file} does not exist!')
            raise FileNotFoundError
        except StopIteration:
            print(f'reading files {file_path}')
            data_f = pd.concat(load_files(file_path), ignore_index=True)
    return data_f


def read_ini(filename):
    """
    Load the configuration file `filename`
    """

    def parse_float_tuple(in_str):
        """
        Given a tuple of two ints in str type, returns a tuple of two ints.
        """
        return tuple(float(k.strip()) for k in in_str[1:-1].split(','))

    config = configparser.ConfigParser(converters={'tuple': parse_float_tuple})
    config.read(filename)
    return config


def read_configuration(filename):
    """
    Read configuration stored at `filename` in JSON format.
    Returns the loaded dict directly.
    """
    with open(filename, 'r') as fn:
        return load(fn)


def print_configuration(filename):
    print(dumps(read_configuration(filename), indent=2))
