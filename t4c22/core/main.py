import sys
from os.path import join as pjoin
from typing import Any, Callable

import hydra
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader

from t4c22.core.train import train
from t4c22.core.utils import get_project_root, seed_everything


def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    accelerator: Accelerator = instantiate(cfg.accelerator)
    model: nn.Module = instantiate(cfg.model)
    optimizer: optim.Optimizer = instantiate(cfg.optimizer)
    train_dataloader: DataLoader = instantiate(cfg.dataset.train_dataloader)
    val_dataloader: DataLoader = instantiate(cfg.dataset.val_dataloader)
    loss_function: Callable[[Any, Any], Any] = instantiate(cfg.loss_function)
    lr_scheduler: optim.lr_scheduler._LRScheduler = instantiate(cfg.lr_scheduler)

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
    CONFIG_NAME = sys.argv[1]
    hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)(main)()
