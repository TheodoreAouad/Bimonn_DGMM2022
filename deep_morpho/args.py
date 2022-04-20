""" File to choose the type of training to do. Choose either the classical morphological operators
or the axSpA ROI detection. Comment the 'arg_path' you do not want to use.

Each `all_args` variables is a list of arguments that will be loaded by deep_morpho/train_net.py.
If you want to launch multiple training at the same time, add the all_args together.

Ex:
>>> all_args1 = importlib.import_module(arg_path1).all_args
>>> all_args2 = importlib.import_module(arg_path2).all_args
>>> all_args = all_args1 + all_args2
"""

import importlib


# Classical Morphological Operators

dataset = "mnist"  # "diskorect"  # "mnist" # "inverted_mnist"
operation = "black_tophat"  # "dilation"  # "erosion" # "opening" # "closing" # "white_tophat" # "black_tophat"
selem = "disk"  # "dcross" # "hstick"

arg_path = f"deep_morpho.args_folder.{dataset}.{operation}.{selem}"

# axSpA ROI detection

architecture = 1  # 2 # 3
# arg_path = f"deep_morpho.args_folder.axspa.architecture_{architecture}"


#######################
all_args = importlib.import_module(arg_path).all_args
#######################
