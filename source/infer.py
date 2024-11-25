# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os
from module_infer import DenoisedModule
from datamodule_infer import DataModule
from pytorch_lightning.cli import LightningCLI

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=DenoisedModule,
        datamodule_class=DataModule,
        seed_everything_default=42,
        run=False,
    )
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="/home/vndata/trung/Thesis/source/checkpoint/DiffInpant/checkpoints/epoch_021.ckpt")

