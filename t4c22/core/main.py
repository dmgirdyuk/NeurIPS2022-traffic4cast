# Copyright 2022 STIL at home.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from os.path import join as pjoin
from pathlib import Path
from shutil import rmtree
from typing import Any, Callable

import hydra
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler  # noqa
from torch.utils.data.dataloader import Dataset

from t4c22.core.dataset import get_train_val_dataloaders, get_city_class_weights
from t4c22.core.train import train
from t4c22.core.utils import get_project_root, seed_everything


def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    rmtree(cfg.dataset.cachedir, ignore_errors=True)

    accelerator: Accelerator = instantiate(cfg.accelerator)

    city_class_weights = get_city_class_weights(city=cfg.city)
    city_class_weights = city_class_weights.to(accelerator.device)

    # TODO: plz someone solve the issue with pathlib.Path from hydra config
    cfg.dataset.root = Path(cfg.dataset.root)
    cfg.dataset.cachedir = Path(cfg.dataset.cachedir)
    dataset: Dataset = instantiate(cfg.dataset)

    model: nn.Module = instantiate(cfg.model)
    optimizer: optim.Optimizer = instantiate(
        cfg.optimizer, params=filter(lambda p: p.requires_grad, model.parameters())
    )
    loss_function: Callable[[Any, Any], Any] = instantiate(
        cfg.loss_function, weight=city_class_weights, ignore_index=-1
    )
    lr_scheduler: _LRScheduler = instantiate(
        cfg.lr_scheduler, optimizer=optimizer
    )

    train_dataloader, val_dataloader = get_train_val_dataloaders(dataset)

    accelerator.init_trackers("example_project", config={})

    model = accelerator.prepare(model)
    optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
        epoch_num=cfg.epoch_num,
        checkpoint_save_folder=cfg.checkpoint_save_folder,
        checkpoint_save_prefix=f"{cfg.experiment}_model",
    )

    # evaluate()


if __name__ == "__main__":
    CONFIG_PATH = pjoin(get_project_root(), "t4c22", "core", "config")
    CONFIG_NAME = sys.argv[1] if len(sys.argv) > 1 else "config_example"
    hydra.main(
        config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base="1.2"
    )(main)()
