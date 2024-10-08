# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import os
from module import MaskedPredictModule
from datamodule import DataModule
from pytorch_lightning.cli import LightningCLI

if __name__ == "__main__":
    cli = LightningCLI(
        model_class=MaskedPredictModule,
        datamodule_class=DataModule,
        seed_everything_default=42,
        run=False,
    )
    cli.trainer.fit(model = cli.model,datamodule = cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="last")