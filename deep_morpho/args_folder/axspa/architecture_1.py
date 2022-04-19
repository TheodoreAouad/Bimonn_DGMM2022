import torch.optim as optim

from deep_morpho.loss import MaskedMSELoss, MaskedDiceLoss, MaskedBCELoss
from general.utils import dict_cross

loss_dict = {
    "MaskedMSELoss": MaskedMSELoss,
    "MaskedDiceLoss": MaskedDiceLoss,
    "MaskedBCELoss": MaskedBCELoss,
}

all_args = {}

all_args['batch_seed'] = [2112050611]

all_args['n_try'] = [0]
# all_args['n_try'] = range(1, 11)

all_args['experiment_name'] = ["axspa_architecture_3"]


# DATA ARGS
all_args['dataset_path'] = ["/hdd/aouadt/these/projets/3d_segm/data/deep_morpho/axspa_roi/axspa_roi.csv"]
all_args['dataset_type'] = ['axspa_roi']
all_args['preprocessing'] = [None]
all_args['train_test_split'] = [(0.8, 0.2, 0)]


# TRAINING ARGS
all_args['learning_rate'] = [1e-1]

# if max_plus, then the loss is MSELoss
all_args['loss_data_str'] = ["MaskedDiceLoss"]
all_args['optimizer'] = [optim.Adam]
all_args['batch_size'] = [64]
all_args['num_workers'] = [20]
all_args['freq_imgs'] = [10]
all_args['n_epochs'] = [6]


# MODEL ARGS
all_args['atomic_element'] = [
    "bisel",
]
all_args['kernel_size'] = [
    41,
]
all_args['channels'] = [
    [
        2, 1
    ]
]
all_args['init_weight_mode'] = [
    "conv_0.5"
]
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

    args['morp_operation'] = []
    args['experiment_subname'] = 'axspa_roi'
    args['n_atoms'] = len(args['channels']) - 1
    args['loss_data'] = loss_dict[args['loss_data_str']](border=(0, 0))

    args['loss'] = {"loss_data": args['loss_data']}
