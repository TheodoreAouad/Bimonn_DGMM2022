import importlib


# Classical Morphological Operators

dataset = "mnist"  # "diskorect"  # "mnist" # "inverted_mnist"
operation = "black_tophat"  # "dilation"  # "erosion" # "opening" # "closing" # "white_tophat" # "black_tophat"
selem = "disk"  # "dcross" # "hstick"

# arg_path = f"deep_morpho.args_folder.{dataset}.{operation}.{selem}"

# axSpA ROI detection

architecture = 1  # 2 # 3
arg_path = f"deep_morpho.args_folder.axspa.architecture_{architecture}"


#######################
all_args = importlib.import_module(arg_path).all_args
#######################
