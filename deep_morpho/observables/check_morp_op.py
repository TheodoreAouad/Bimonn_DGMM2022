import pathlib
from os.path import join

from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.tensorboard.summary import custom_scalars
import matplotlib.pyplot as plt
import numpy as np

from .observable_layers import ObservableLayers, ObservableLayersChans
from general.utils import save_json



class CheckMorpOperation(ObservableLayers):

    def __init__(self, selems, operations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selems = selems
        self.operations = operations


    def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        layers = self._get_layers(pl_module)
        default_layout = self.get_layout(layers)
        layout = {"default": default_layout}
        trainer.logger.experiment._get_file_writer().add_summary(custom_scalars(layout))


    @staticmethod
    def get_layout(layers):
        default_layout = {}
        for layer_idx, layer in enumerate(layers):
            for bise_idx, bise_layer in enumerate(layer.bises):
                tags_dilation = [
                    f'comparative/weights/bias_{layer_idx}_{bise_idx}/bias',
                    f'comparative/weights/bias_{layer_idx}_{bise_idx}/dilation_lb',
                    f'comparative/weights/bias_{layer_idx}_{bise_idx}/dilation_ub'
                ]
                tags_erosion = [
                    f'comparative/weights/bias_{layer_idx}_{bise_idx}/bias',
                    f'comparative/weights/bias_{layer_idx}_{bise_idx}/erosion_lb',
                    f'comparative/weights/bias_{layer_idx}_{bise_idx}/erosion_ub'
                ]
                default_layout.update({
                    f"dilation_{layer_idx}_{bise_idx}": ['Margin', tags_dilation],
                    f"erosion_{layer_idx}_{bise_idx}": ['Margin', tags_erosion],
                })


        return default_layout

    def on_train_batch_end_layers(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
    ):
        self.write_scalars_and_metrics(trainer, layer, layer_idx, layer_idx)


    def write_scalars_and_metrics(self, trainer, layer, current_name, op_idx):
        erosion_lb, erosion_ub = layer.bias_bounds_erosion(self.selems[op_idx])
        dilation_lb, dilation_ub = layer.bias_bounds_dilation(self.selems[op_idx])

        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/bias", -layer.bias, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/erosion_lb", erosion_lb, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/erosion_ub", erosion_ub, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/dilation_lb", dilation_lb, trainer.global_step)
        trainer.logger.experiment.add_scalar(f"comparative/weights/bias_{current_name}/dilation_ub", dilation_ub, trainer.global_step)

        if self.operations[op_idx] == 'dilation':
            metrics = {
                f"metrics/bias - lb(op)_{current_name}": -layer.bias - dilation_lb,
                f"metrics/ub(op) - bias_{current_name}": dilation_ub - (-layer.bias),
            }
        elif self.operations[op_idx] == 'erosion':
            metrics = {
                f"metrics/bias - lb(op)_{current_name}": -layer.bias - erosion_lb,
                f"metrics/ub(op) - bias_{current_name}": erosion_ub - (-layer.bias),
            }
        else:
            raise NotImplementedError('operation must be dilation or erosion.')

        trainer.logger.log_metrics(metrics, trainer.global_step)


class ShowSelemBinary(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_selem_and_op = {}

    def on_train_batch_end_layers_chans(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        with torch.no_grad():
            selem, operation = layer.bises[chan_input].find_selem_and_operation_chan(chan_output, v1=0, v2=1)
        if selem is None:
            return

        fig = self.selem_fig(selem, operation)
        trainer.logger.experiment.add_figure(f"learned_selem/binary/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig, trainer.global_step)
        self.last_selem_and_op[(layer_idx, chan_input, chan_output)] = (selem, operation)


    @staticmethod
    def selem_fig(selem, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest", vmin=0, vmax=1)
        plt.title(operation)
        return fig


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_input, chan_output), (selem, operation) in self.last_selem_and_op.items():
            fig = self.selem_fig(selem, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}.png"))
            saved.append(fig)

        return saved


class ShowLUISetBinary(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_set_and_op = {}

    def on_train_batch_end_layers_chan_output(
        self,
        trainer='pl.Trainer',
        pl_module='pl.LightningModule',
        outputs="STEP_OUTPUT",
        batch="Any",
        batch_idx=int,
        dataloader_idx=int,
        layer="nn.Module",
        layer_idx=int,
        chan_output=int,
    ):
        with torch.no_grad():
            C, operation = layer.luis[chan_output].find_set_and_operation_chan(0, v1=None, v2=None)
        if C is None:
            return

        fig = self.set_fig(C, operation)
        trainer.logger.experiment.add_figure(f"learned_set_lui/binary/layer_{layer_idx}_chout_{chan_output}", fig, trainer.global_step)
        self.last_set_and_op[(layer_idx, chan_output)] = (C, operation)


    @staticmethod
    def set_fig(C, operation):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(C[:, None].astype(int), interpolation="nearest", vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks(range(len(C)))
        plt.title(operation)
        return fig


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        pathlib.Path(final_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_output), (C, operation) in self.last_set_and_op.items():
            fig = self.set_fig(C, operation)
            fig.savefig(join(final_dir, f"layer_{layer_idx}_chout_{chan_output}.png"))
            saved.append(fig)

        return saved


class ShowClosestSelemBinary(ObservableLayersChans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_elts = {}
        self.last_selems = {}
        self.freq_idx2 = 0

    def on_train_batch_end_layers_chans(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        outputs: "STEP_OUTPUT",
        batch: "Any",
        batch_idx: int,
        dataloader_idx: int,
        layer: "nn.Module",
        layer_idx: int,
        chan_input: int,
        chan_output: int,
    ):
        with torch.no_grad():
            selem, operation, distance = layer.bises[chan_input].find_closest_selem_and_operation_chan(chan_output, v1=0, v2=1)

        trainer.logger.experiment.add_scalar(f"comparative/closest_binary_dist/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", distance, trainer.global_step)

        fig = self.selem_fig(selem, f"{operation} dist {distance:.2f}")
        trainer.logger.experiment.add_figure(f"closest_selem/binary/layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}", fig, trainer.global_step)
        self.last_elts[str((layer_idx, chan_input, chan_output))] = {"operation": operation, "distance": str(distance)}
        self.last_selems[(layer_idx, chan_input, chan_output)] = selem


    @staticmethod
    def selem_fig(selem, title):
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(selem, interpolation="nearest", vmin=0, vmax=1)
        plt.title(title)
        return fig


    def save(self, save_path: str):
        final_dir = join(save_path, self.__class__.__name__)
        numpy_dir = join(final_dir, "selem_npy")
        png_dir = join(final_dir, "selem_png")
        pathlib.Path(numpy_dir).mkdir(exist_ok=True, parents=True)
        pathlib.Path(png_dir).mkdir(exist_ok=True, parents=True)

        saved = []

        for (layer_idx, chan_input, chan_output), selem in self.last_selems.items():
            filename = f"layer_{layer_idx}_chin_{chan_input}_chout_{chan_output}"

            elts = self.last_elts[str((layer_idx, chan_input, chan_output))]
            operation, distance = elts['operation'], elts['distance']

            fig = self.selem_fig(selem, f"{operation} dist {distance}")
            fig.savefig(join(png_dir, f"{filename}.png"))
            saved.append(fig)
            np.save(join(numpy_dir, f'{filename}.npy'), selem.astype(np.uint8))

        save_json(self.last_elts, join(final_dir, "operation_distance.json"))
        saved.append(self.last_elts)

        return saved
