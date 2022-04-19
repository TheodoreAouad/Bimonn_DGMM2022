import numpy as np
import torch.optim as optim

from deep_morpho.datasets.generate_forms3 import get_random_diskorect_channels
from deep_morpho.loss import MaskedMSELoss, MaskedDiceLoss, MaskedBCELoss
from general.utils import dict_cross
from deep_morpho.morp_operations import ParallelMorpOperations


loss_dict = {
    "MaskedMSELoss": MaskedMSELoss,
    "MaskedDiceLoss": MaskedDiceLoss,
    "MaskedBCELoss": MaskedBCELoss,
}

all_args = {}

all_args['batch_seed'] = [2302492412]  # put to None for random seed

all_args['n_try'] = [0]
# all_args['n_try'] = range(1, 11)  # uncomment to launch 10 identical trials

all_args['experiment_name'] = [
    "diskorect_opening_hstick"
]


# DATA ARGS
all_args['morp_operation'] = [ParallelMorpOperations.opening(('hstick', 7))]
all_args['dataset_type'] = ['diskorect']
all_args['preprocessing'] = [None]
all_args['random_gen_fn'] = [get_random_diskorect_channels]
all_args['random_gen_args'] = [
    {'size': (50, 50), 'n_shapes': 20, 'max_shape': (20, 20), 'p_invert': 0.5, 'n_holes': 10, 'max_shape_holes': (10, 10), 'noise_proba': 0.02}
]
all_args['mnist_args'] = [
    {"threshold": 30, "size": (50, 50), "invert_input_proba": 0},
]
all_args['n_inputs'] = [200000]
all_args['train_test_split'] = [(0.8, 0.2, 0)]


# TRAINING ARGS
all_args['learning_rate'] = [0.01]

# if max_plus, then the loss is MSELoss
all_args['loss_data_str'] = ["MaskedMSELoss"]
all_args['optimizer'] = [optim.Adam]
all_args['batch_size'] = [64]
all_args['num_workers'] = [20]
all_args['freq_imgs'] = [300]
all_args['n_epochs'] = [1]


# MODEL ARGS
all_args['atomic_element'] = ["bisel"]
all_args['kernel_size'] = ["adapt"]
all_args['n_atoms'] = ['adapt']
all_args['channels'] = ['adapt']
all_args['init_weight_mode'] = ["conv_0.5"]
all_args['activation_P'] = [0]
all_args['force_lui_identity'] = [False]
all_args['constant_activation_P'] = [False]
all_args['constant_P_lui'] = [False]
all_args['constant_weight_P'] = [True]
all_args['threshold_mode'] = [
    {
        "weight": 'softplus',
        "activation": 'tanh',
    },
]

all_args = dict_cross(all_args)
#
for idx, args in enumerate(all_args):

    if args["kernel_size"] == "adapt":
        # args["kernel_size"] = args["morp_operation"].selems[0][0][0].shape[0]
        args["kernel_size"] = int(max(args['morp_operation'].max_selem_shape))

    args['loss_data'] = loss_dict[args['loss_data_str']](border=np.array([args['kernel_size'] // 2, args['kernel_size'] // 2]))
    args['experiment_subname'] = f"{args['threshold_mode']['weight']}/{args['dataset_type']}/{args['morp_operation'].name}"

    if args['channels'] == 'adapt':
        args['channels'] = args['morp_operation'].in_channels + [args['morp_operation'].out_channels[-1]]

    if args["n_atoms"] == 'adapt':
        args['n_atoms'] = len(args['morp_operation'])
        if args['atomic_element'] in ['cobise', 'cobisec']:
            args['n_atoms'] = max(args['n_atoms'] // 2, 1)


    args["random_gen_args"] = args["random_gen_args"].copy()
    args["random_gen_args"]["border"] = (args["kernel_size"]//2 + 1, args["kernel_size"]//2 + 1)
    args['random_gen_args']['size'] = args['random_gen_args']['size'] + (args["morp_operation"].in_channels[0],)

    args['loss'] = {"loss_data": args['loss_data']}
