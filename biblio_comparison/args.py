""" File to launch the bibliography comparison. We describe the config for the LMorph and SMorph.
The studied morphological operation are defined in `biblio_comparison/args_morp_ops_diskorect.py` and
`biblio_comparison/args_morp_ops_mnist.py`.
You can change the values for each argument. You must also choose the dataset and model to train on.
"""

import torch.nn as nn
import torch.optim as optim

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from general.utils import dict_cross
from .args_morp_ops_mnist import morp_operations as morp_operations_mnist
from .args_morp_ops_diskorect import morp_operations as morp_operations_diskorect


all_args = {}

all_args['n_try'] = [0]
# all_args['n_try'] = range(1, 11)

all_args['experiment_name'] = [
    "biblio"
]

#########################

morp_operations = []


# DATA ARGS
all_args['preprocessing'] = [  # for axspa roi
    None,
]
all_args['dataset_path'] = [
    # 'data/deep_morpho/dataset_0',
    'generate',
]
all_args['in_ram'] = [
    # False,
    True,
]
all_args['random_gen_fn'] = [
    # get_random_rotated_diskorect,
    get_random_diskorect_channels
]
all_args['random_gen_args'] = [
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02}
    # {'size': (50, 50), 'n_shapes': 30, 'max_shape': (15, 15), 'p_invert': 0.5, 'n_holes': 15, 'max_shape_holes': (7, 7)}

]

all_args['n_inputs'] = [
    1_000_000,
]
all_args['train_test_split'] = [(1, 1, 0)]


# TRAINING ARGS

all_args['loss'] = [nn.MSELoss()]
all_args['num_workers'] = [
    20,
]
all_args['freq_imgs'] = [300]
all_args['n_epochs'] = [20]


# MODEL ARGS
all_args['kernel_size'] = [
    # 7,
    "adapt",
]

##################
# LMorph, SMorph #
##################

all_args['model'] = [
    "lmorph",
    "smorph",
]
all_args['optimizer'] = [optim.Adam]
all_args['batch_size'] = [256]
all_args['learning_rate'] = [1e-3*2]

all_args_lsmorph = []

# Uncomment to try on diskorect
all_args_lsmorph += dict_cross(dict(**all_args, **{'dataset_type': ["diskorect"], "morp_operation": morp_operations_diskorect}))

all_args['mnist_args'] = [
    {"threshold": 30, "size": (50, 50), "invert_input_proba": 0}
]
# Uncomment to try on mnist
all_args_lsmorph += dict_cross(dict(**all_args, **{'dataset_type': ["mnist"], "morp_operation": morp_operations_mnist}))


all_args['mnist_args'] = [
    {"threshold": 30, "size": (50, 50), "invert_input_proba": 1}
]
# Uncomment to try on inverted_mnist
all_args_lsmorph += dict_cross(dict(**all_args, **{'dataset_type': ["mnist"], "morp_operation": morp_operations_mnist}))


##############
# Adaptative #
##############

all_args['model'] = ["adaptative"]
all_args['optimizer'] = [optim.SGD]
all_args['learning_rate'] = [10]
all_args['batch_size'] = [64]



all_args_adaptative = (
    dict_cross(dict(**all_args, **{'dataset_type': ["mnist"], "morp_operation": morp_operations_mnist})) +
    dict_cross(dict(**all_args, **{'dataset_type': ["diskorect"], "morp_operation": morp_operations_diskorect})) +
    []
)


### End of editable zone ###


all_args = (
    all_args_lsmorph +
    # all_args_adaptative +  # We do not launch adaptative as we did not manage to make it converge.
    []
)

#
for idx, args in enumerate(all_args):


    # args['kernel_size'] = 'adapt'
    args['n_atoms'] = 'adapt'


    if args["kernel_size"] == "adapt":
        args["kernel_size"] = args["morp_operation"].selems[0][0][0].shape[0]

    args['experiment_subname'] = f"{args['dataset_type']}/{args['morp_operation'].name}/{args['model']}"

    if args["n_atoms"] == 'adapt':
        args['n_atoms'] = len(args['morp_operation'])

    if args['dataset_type'] == "diskorect":
        args['n_epochs'] = 1
        args["random_gen_args"] = args["random_gen_args"].copy()
        args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
        args['random_gen_args']['size'] = args['random_gen_args']['size'] + (args["morp_operation"].in_channels[0],)


    if args['dataset_type'] == "mnist":
        args['n_inputs'] = 70_000
        if args['mnist_args']['invert_input_proba'] == 1:
            args['experiment_subname'] = f"inverted_{args['experiment_subname']}"
