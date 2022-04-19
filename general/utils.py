import json
import re
import os
import logging
from os.path import join
import yaml

import numpy as np
from sklearn.model_selection import ParameterGrid


def get_next_same_name(parent_dir, pattern='', sep='', crude=False):
    """
    Scans the folder parent dir for files/folders with name 'pattern{}' with {} being an integer. Returns
    the path to the file/folder with name 'pattern-{}' with the highest number +1.

    Args:
        parent_dir (str): path to the parent directory of the files / folders with the pattern
        pattern (str, optional): pattern of the file to look for. Defaults to ''.
        crude (bool, optional): if True, if file does not exist, will return parent_dir/pattern. Else,
                                will add 0.
    """
    if pattern == '':
        sep = ''

    if crude and not os.path.exists(join(parent_dir, pattern)):
        return join(parent_dir, pattern)

    if not os.path.exists(parent_dir):
        return join(parent_dir, '{}{}{}'.format(pattern, sep, 0))

    directories = [o for o in os.listdir(parent_dir) if re.search(r'^{}{}\d+$'.format(pattern, sep), o)]
    if len(directories) == 0:
        max_nb = 0
    else:
        nbrs = [int(re.findall(r'^{}{}(\d+)$'.format(pattern, sep), o)[0]) for o in directories]
        max_nb = max(nbrs) + 1
    return join(parent_dir, '{}{}{}'.format(pattern, sep, max_nb))


def save_json(dic, path, sort_keys=True, indent=4):
    with open(path, 'w') as fp:
        json.dump(dic, fp, sort_keys=sort_keys, indent=indent)


def load_json(path):
    with open(path, 'r') as fp:
        return json.load(fp)


def set_borders_to(ar: "np.ndarray", border: "Tuple", value: float = 0, ):
    res = ar + 0
    res[:border[0], :] = value
    res[-border[0]:, :] = value
    res[:, :border[1]] = value
    res[:, -border[1]:] = value
    return res


def log_console(to_print='', *args, level='info', logger=None, **kwargs):
    if logger is None:
        print(to_print, *args, **kwargs)
    else:
        to_print = '{}'.format(to_print)
        for st in args:
            to_print = to_print + ' {} '.format(st)
        getattr(logger, level.lower())(to_print)


def one_hot_array(ar: np.ndarray, nb_chans: int = "auto", axis: int = -1, background: float = 0) -> np.ndarray:
    """ Performs one hot encoding of an array. Adds a channel on the axis axis, such that all the channels are binary.

    Args:
        ar (np.ndarray): array to one hot encode.
        axis (np.ndarray): axis where the additional channel is created.
        background (None | float): value to ignore

    Returns:
        np.ndarray: one hot encoded array
    """
    unique_values = sorted(list(set(np.unique(ar)).difference([background])))
    if nb_chans == 'auto':
        nb_chans = len(unique_values)

    if axis == -1:
        res = np.zeros(ar.shape + (nb_chans,))

        for idx, value in enumerate(unique_values):
            res[..., idx] = ar == value
        return res

    elif axis == 0:
        res = np.zeros((nb_chans,) + ar.shape)

        for idx, value in enumerate(unique_values):
            res[idx] = ar == value
        return res

    else:
        raise ValueError("axis must be 0 or -1.")
    

def max_min_norm(ar: np.ndarray) -> np.ndarray:
    armin = ar.min()
    armax = ar.max()

    if armin == armax:
        return ar / armax

    ar = ar + 0
    ar[ar == np.infty] = ar[ar != np.infty].max()
    return (ar - armin) / (armax - armin)


def format_time(s):
    h = s // (3600)
    s %= 3600
    m = s // 60
    s %= 60
    return "%02i:%02i:%02i" % (h, m, s)



def create_logger(logger_name=None, all_logs_path=None, error_path=None, level="info"):

    level_dicts = {'debug': logging.DEBUG, 'info': logging.INFO}
    logging.basicConfig(
        level=level_dicts[level],
        format='%(message)s',
    )

    logger = logging.getLogger(__name__ if logger_name is None else logger_name)

    # Handling logs
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if error_path is not None:
        error_handler = logging.FileHandler(error_path)
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.WARNING)
        logger.addHandler(error_handler)

    if all_logs_path is not None:
        info_handler = logging.FileHandler(all_logs_path)
        info_handler.setFormatter(formatter)
        info_handler.setLevel(logging.DEBUG)
        logger.addHandler(info_handler)

    return logger


def save_yaml(dic, path):
    with open(path, 'w') as f:
        yaml.dump(dic, f)


def dict_cross(dic):
    """
    Does a cross product of all the values of the dict.

    Args:
        dic (dict): dict to unwrap

    Returns:
        list: list of the dict
    """

    return list(ParameterGrid(dic))